#!/usr/bin/env python3
"""TT Oobleck VAE decoder (tt_acestep/oobleck_vae.py) against diffusers' AutoencoderOobleck on the CPU (float32).

Latents: x0 = x - v * t of the last recorded decoder call of a golden run (a real denoised song). Runs inside the TT
image with the P100a (diffusers is installed there):

  python /work/experiments/tt-acestep/vae_check.py /golden --frames 256 [--timing 256 1024 2048]
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tt import acestep_dit, oobleck_vae  # noqa: E402


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def spectral_db(a, b, n_fft=2048, hop=512):
    """Mean |dB difference| of log-magnitude spectrograms, all bins and bins below -60 dB of the peak (quiet parts,
    where a low-precision noise floor shows)."""
    spec = lambda w: torch.stft(w.mean(0), n_fft, hop, window=torch.hann_window(n_fft), return_complex=True).abs()
    sa, sb = spec(a), spec(b)
    da, db = 20 * torch.log10(sa + 1e-7), 20 * torch.log10(sb + 1e-7)
    quiet = da < da.max() - 60
    return {"all_db": float((da - db).abs().mean()), "quiet_db": float((da - db)[quiet].abs().mean()),
            "quiet_fraction": float(quiet.float().mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("golden")
    parser.add_argument("--vae", default="/vae")
    parser.add_argument("--frames", type=int, default=256, help="latent frames compared with the CPU decoder")
    parser.add_argument("--timing", type=int, nargs="*", default=[], help="extra lengths, timing only")
    parser.add_argument("--out", default="vae_check.json")
    parser.add_argument("--act", choices=["bf16", "fp32"], default="bf16", help="activation dtype on the card")
    args = parser.parse_args()
    golden_dir = Path(args.golden)
    last = torch.load(golden_dir / "decoder_calls.pt")["steps"][-1]
    x0 = (last["x"] - last["v"] * last["t"].view(-1, 1, 1))[0].t().contiguous()  # [64, T]
    lat = x0[:, : args.frames]
    result = {"frames": lat.shape[-1]}

    from diffusers import AutoencoderOobleck

    ref = AutoencoderOobleck.from_pretrained(args.vae, torch_dtype=torch.float32).eval()
    ref_taps = []
    hooks = [ref.decoder.conv1.register_forward_hook(lambda m, i, o: ref_taps.append(o[0].t().clone()))]
    hooks += [b.register_forward_hook(lambda m, i, o: ref_taps.append(o[0].t().clone())) for b in ref.decoder.block]
    t0 = time.monotonic()
    with torch.inference_mode():
        expected = ref.decode(lat[None]).sample[0]
    result["cpu_s"] = round(time.monotonic() - t0, 2)
    for h in hooks:
        h.remove()
    # what a bf16 VAE does anyway (the GPU runner decodes in bf16): the same decoder on the CPU in bf16
    with torch.inference_mode():
        bf16 = ref.to(torch.bfloat16).decode(lat[None].to(torch.bfloat16)).sample[0].float()
    ref.float()
    result["cpu_bf16"] = {"audio_pcc": pcc(bf16, expected), "spectral": spectral_db(bf16, expected)}
    print(json.dumps({"cpu_bf16": result["cpu_bf16"]}), flush=True)

    dev = acestep_dit.open_device()
    try:
        t0 = time.monotonic()
        prec = oobleck_vae.Precision(act_dtype={"bf16": oobleck_vae.ttnn.bfloat16, "fp32": oobleck_vae.ttnn.float32}[args.act])
        result["act"] = args.act
        vae = oobleck_vae.OobleckTT(dev, args.vae, prec)
        result["load_s"] = round(time.monotonic() - t0, 2)
        runs = []
        for p in range(2):  # the first pass compiles
            taps = []
            t0 = time.monotonic()
            got = vae.decode(lat, taps=taps if p == 1 else None)
            runs.append(round(time.monotonic() - t0, 2))
        result["tt_s"] = runs
        n = min(got.shape[-1], expected.shape[-1])
        result.update(shape_tt=list(got.shape), shape_cpu=list(expected.shape), audio_pcc=pcc(got[:, :n], expected[:, :n]),
                      max_abs=float((got[:, :n] - expected[:, :n]).abs().max()),
                      rms_cpu=float(expected.pow(2).mean().sqrt()), rms_tt=float(got.pow(2).mean().sqrt()),
                      stage_pcc=[round(pcc(a, b[: a.shape[0]]), 6) for a, b in zip(taps, ref_taps)],
                      spectral=spectral_db(got[:, :n], expected[:, :n]))
        print(json.dumps({k: result[k] for k in ("act", "frames", "cpu_s", "tt_s", "audio_pcc", "max_abs", "spectral",
                                                  "stage_pcc")}), flush=True)
        result["timing"] = []
        for frames in args.timing:
            dev.clear_program_cache()  # conv L1_SMALL configs accumulate per input length
            z = x0[:, :frames] if frames <= x0.shape[-1] else x0.repeat(1, frames // x0.shape[-1] + 1)[:, :frames]
            times = []
            for p in range(2):
                t0 = time.monotonic()
                vae.decode(z)
                times.append(round(time.monotonic() - t0, 2))
            rec = {"frames": frames, "seconds_of_audio": frames / 25, "tt_s": times, "dram": acestep_dit.dram_stats(dev)}
            result["timing"].append(rec)
            print(json.dumps(rec), flush=True)
    finally:
        acestep_dit.close_device(dev)
        (golden_dir / args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
