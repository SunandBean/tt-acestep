#!/usr/bin/env python3
"""Golden data for the ACE-Step 1.5 Turbo DiT port to the Tenstorrent P100a.

Runs the real pipeline (the same handler and GenerationParams the port is driven with) on the CPU in float32
and records every call to model.decoder: its inputs (noisy latents, timestep, encoder states, context latents) and
its output velocity, plus the final audio. The TT decoder is checked against these tensors step by step.

Run with the ACE-Step environment (CPU only, never touches the GPU or the P100a):
  CUDA_VISIBLE_DEVICES= vendor/ACE-Step-1.5/.venv/bin/python experiments/tt-acestep/dump_reference.py \
      --duration 30 --out experiments/tt-acestep/golden/d30
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runners"))

PROMPT = "warm lo-fi hip hop, mellow electric piano, soft vinyl crackle, relaxed female vocal, 80 bpm"
LYRICS = """[Verse]
City lights are fading slow
Coffee warm and radio low
[Chorus]
Stay a while, the night is long
Hum along to our old song"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instrumental", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tt", action="store_true", help="run the decoder on the P100a (tt_acestep/remote.py) instead")
    parser.add_argument("--keep-lm-hints", action="store_true", help="with --tt: keep upstream's discarded LM hints")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    acestep_root = Path(os.environ.get("ACESTEP_ROOT", ROOT / "vendor/ACE-Step-1.5")).resolve()
    sys.path.insert(0, str(acestep_root))
    os.chdir(acestep_root)
    os.environ.update(ACESTEP_DISABLE_TQDM="1", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1")
    import torch
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler
    from acestep_runner import local_model_preflight

    class LocalDirectConditioningHandler(AceStepHandler):
        def _ensure_models_present(self, *, checkpoint_path, config_path, prefer_source, vae_variant=None):
            return local_model_preflight(checkpoint_path, config_path, vae_variant)

    torch.manual_seed(args.seed)
    worker = None
    if args.tt:  # the container loads the DiT while ACE-Step loads on the CPU
        sys.path.insert(0, str(ROOT))
        from tt.remote import RemoteDecoder, RemoteVae, TTWorker, skip_unused_lm_hints
        worker = TTWorker(acestep_root / "checkpoints/acestep-v15-turbo", log=open(out / "worker.log", "w")).start()
    handler = LocalDirectConditioningHandler()
    status, ok = handler.initialize_service(
        project_root=str(acestep_root), config_path=os.environ.get("ACESTEP_MODEL", "acestep-v15-turbo"),
        device="cpu", use_flash_attention=False, compile_model=False,
        offload_to_cpu=False, offload_dit_to_cpu=False, quantization=None)
    if not ok:
        raise RuntimeError(status)
    if worker is not None:
        worker.wait_ready()
        handler.model.decoder = RemoteDecoder(worker)
        handler.tiled_decode = RemoteVae(worker)
        if not args.keep_lm_hints:
            skip_unused_lm_hints(handler.model)
    decoder = handler.model.decoder
    calls = []

    def before(module, args_, kwargs):
        rec = {k: kwargs[k].detach().float().clone() for k in
               ("hidden_states", "timestep", "timestep_r", "encoder_hidden_states", "context_latents")}
        rec["started"] = time.monotonic()
        calls.append(rec)

    def after(module, args_, kwargs, output):
        calls[-1]["output"] = output[0].detach().float().clone()
        calls[-1]["seconds"] = time.monotonic() - calls[-1].pop("started")

    decoder.register_forward_pre_hook(before, with_kwargs=True)
    decoder.register_forward_hook(after, with_kwargs=True)

    params = GenerationParams(
        caption=PROMPT, lyrics="[Instrumental]" if args.instrumental else LYRICS, instrumental=args.instrumental,
        vocal_language="en", duration=args.duration, inference_steps=8, seed=args.seed,
        thinking=False, use_cot_metas=False, use_cot_caption=False, use_cot_language=False)
    config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=False, seeds=[args.seed])
    started = time.monotonic()
    try:
        result = generate_music(handler, LLMHandler(), params, config, save_dir=str(out / "raw"))
    finally:
        if worker is not None:
            worker.close()
    if not result.success or not result.audios:
        raise RuntimeError(result.error or result.status_message)
    shutil.move(result.audios[0]["path"], out / "reference.wav")

    first = calls[0]
    steps = [{"x": c["hidden_states"], "t": c["timestep"], "t_r": c["timestep_r"], "v": c["output"]} for c in calls]
    for c in calls[1:]:  # the pipeline conditions every step on the same encoder states and context
        assert torch.equal(c["encoder_hidden_states"], first["encoder_hidden_states"])
        assert torch.equal(c["context_latents"], first["context_latents"])
    torch.save({"encoder_hidden_states": first["encoder_hidden_states"], "context_latents": first["context_latents"],
                "steps": steps}, out / "decoder_calls.pt")
    meta = {"prompt": PROMPT, "instrumental": args.instrumental, "duration": args.duration, "seed": args.seed,
            "steps": len(calls), "timesteps": [float(c["timestep"][0]) for c in calls],
            "latent_shape": list(first["hidden_states"].shape),
            "encoder_shape": list(first["encoder_hidden_states"].shape),
            "context_shape": list(first["context_latents"].shape),
            "decoder_seconds": [round(c["seconds"], 3) for c in calls],
            "total_seconds": round(time.monotonic() - started, 1), "torch": torch.__version__,
            "decoder": "p100a" if worker is not None else "cpu", "worker": worker.stats if worker is not None else None,
            "lm_hints": "kept" if (worker is None or args.keep_lm_hints) else "skipped",
            "time_costs": getattr(result, "extra_outputs", {}).get("time_costs") if hasattr(result, "extra_outputs") else None}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
