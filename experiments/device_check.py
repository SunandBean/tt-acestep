#!/usr/bin/env python3
"""TT decoder (tt_acestep/acestep_dit.py) against the upstream decoder calls recorded by dump_reference.py, step by step
with the recorded inputs (teacher forcing). Runs inside the TT image with the P100a:

  python experiments/tt-acestep/device_check.py /golden/d30 [--repeat 2] [--taps]

Start it with run_on_card.sh. Writes device_check.json next to the golden data.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tt import acestep_dit, acestep_host as host  # noqa: E402


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("golden")
    parser.add_argument("--checkpoint", default="/checkpoint")
    parser.add_argument("--repeat", type=int, default=2, help="passes over the steps (the first one compiles)")
    parser.add_argument("--taps", action="store_true", help="compare per-layer hidden states with RefDecoder")
    parser.add_argument("--mm-fidelity", choices=["LoFi", "HiFi2", "HiFi3", "HiFi4"], default="HiFi2")
    parser.add_argument("--weights", choices=["bf16", "bfp8"], default="bf16")
    parser.add_argument("--out", default="device_check.json")
    parser.add_argument("--frames", type=int, nargs="*", help="timing only: random latents of these lengths "
                                                                 "(25 frames per second), golden encoder states")
    args = parser.parse_args()
    golden_dir = Path(args.golden)
    golden = torch.load(golden_dir / "decoder_calls.pt")
    enc, ctx = golden["encoder_hidden_states"][0], golden["context_latents"][0]
    result = {"golden": str(golden_dir), "passes": []}
    t0 = time.monotonic()
    dev = acestep_dit.open_device()
    try:
        result["open_s"] = round(time.monotonic() - t0, 2)
        t0 = time.monotonic()
        prec = acestep_dit.Precision(mm_fidelity=getattr(acestep_dit.ttnn.MathFidelity, args.mm_fidelity),
                                     weight_dtype={"bf16": acestep_dit.ttnn.bfloat16,
                                                   "bfp8": acestep_dit.ttnn.bfloat8_b}[args.weights])
        result["precision"] = {"mm_fidelity": args.mm_fidelity, "weights": args.weights}
        dit = acestep_dit.AceStepDiT(dev, host.Checkpoint(args.checkpoint), prec)
        result["load_s"] = round(time.monotonic() - t0, 2)
        result["dram_after_load"] = acestep_dit.dram_stats(dev)
        if args.frames:
            result["timing"] = timing_only(dit, dev, golden, enc, args.frames)
            return
        t0 = time.monotonic()
        song = dit.prepare(enc, ctx.shape[0])
        g = song.geo
        result.update(prepare_s=round(time.monotonic() - t0, 2),
                      geometry={"frames": g.frames, "seq": g.seq, "seq_pad": g.seq_pad, "enc_len": g.enc_len,
                                "enc_pad": g.enc_pad})
        for p in range(args.repeat):
            steps = []
            for s in golden["steps"]:
                t, t_r = float(s["t"][0]), float(s["t_r"][0])
                started = time.monotonic()
                v = dit.forward(s["x"][0], t, t_r, ctx, song)
                steps.append({"t": t, "pcc": pcc(v, s["v"][0]), "max_abs": float((v - s["v"][0]).abs().max()),
                              "seconds": round(time.monotonic() - started, 3)})
                print(json.dumps({"pass": p, **steps[-1]}), flush=True)
            result["passes"].append({"min_pcc": min(x["pcc"] for x in steps), "steps": steps,
                                     "total_s": round(sum(x["seconds"] for x in steps), 2)})
        if args.taps:  # where does the error enter: per-layer hidden states of the first step against RefDecoder
            s = golden["steps"][0]
            taps = []
            dit.forward(s["x"][0], float(s["t"][0]), float(s["t_r"][0]), ctx, song, taps=taps)
            ref_taps = []
            dit.ref.forward(s["x"][0], float(s["t"][0]), float(s["t_r"][0]), ctx, dit.ref.encode(enc, g), g,
                            taps=ref_taps)
            result["layer_pcc"] = [round(pcc(a[: g.seq], b[: g.seq]), 6) for a, b in zip(taps, ref_taps)]
            print(json.dumps({"layer_pcc": result["layer_pcc"]}), flush=True)
        result["dram_after_steps"] = acestep_dit.dram_stats(dev)
        dit.release(song)
    finally:
        acestep_dit.close_device(dev)
        (golden_dir / args.out).write_text(json.dumps(result, indent=2))
    if result.get("passes"):
        print(json.dumps({k: result[k] for k in ("geometry", "load_s")} | {"min_pcc": [x["min_pcc"] for x in result["passes"]],
                                                                             "total_s": [x["total_s"] for x in result["passes"]]}))


def timing_only(dit, dev, golden, enc, frame_counts):
    out = []
    schedule = [(float(s["t"][0]), float(s["t_r"][0])) for s in golden["steps"]]
    for frames in frame_counts:
        g = torch.Generator().manual_seed(frames)
        x, ctx = torch.randn(frames, 64, generator=g), torch.randn(frames, 128, generator=g)
        t0 = time.monotonic()
        song = dit.prepare(enc, frames)
        rec = {"frames": frames, "seconds_of_audio": frames / 25, "seq_pad": song.geo.seq_pad,
               "prepare_s": round(time.monotonic() - t0, 2), "steps_s": []}
        for p in range(2):  # the first pass compiles this length
            for t, t_r in schedule:
                t0 = time.monotonic()
                v = dit.forward(x, t, t_r, ctx, song)
                rec["steps_s"].append(round(time.monotonic() - t0, 3))
        rec["finite"] = bool(torch.isfinite(v).all())
        rec["dram"] = acestep_dit.dram_stats(dev)
        rec["compiled_8_steps_s"] = round(sum(rec["steps_s"][len(schedule):]), 2)
        dit.release(song)
        out.append(rec)
        print(json.dumps({k: rec[k] for k in ("frames", "seq_pad", "compiled_8_steps_s", "finite", "dram")}), flush=True)
    return out


if __name__ == "__main__":
    main()
