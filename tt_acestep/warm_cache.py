#!/usr/bin/env python3
"""Compile every P100a kernel shape this port can hit, ahead of the first songs.

The card compiles kernels per shape and keeps them in TT_METAL_CACHE (data/tt-cache). Without this, the first song
of each new length bucket pays that compilation (tens of seconds). Shapes, per model (--model):
  acestep: every DiT sequence bucket of a 30-480 s song x every encoder bucket (acestep_host.SEQ_BUCKET /
           ENC_BUCKET), and both VAE windows (remote.py RemoteVae)
           both VAE windows
Random inputs: only the shapes matter. Run it again after changing this package or the TT image.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from . import acestep_dit, acestep_host as host, oobleck_vae
from .shapes import VAE_WINDOWS

MIN_FRAMES, MAX_FRAMES = 30 * 25, 480 * 25  # ACE-Step's 30-480 s at 25 latent frames per second


def buckets(lo, hi, step):
    return sorted({host.round_up(n, step) for n in range(lo, hi + 1)})


def warm_vae(vae, report):
    for window in VAE_WINDOWS:  # exactly as the worker does right after loading (worker.py)
        t0 = time.monotonic()
        audio = vae.decode(torch.zeros(64, window))
        rec = {"vae_window": window, "s": round(time.monotonic() - t0, 2), "finite": bool(torch.isfinite(audio).all())}
        report["vae"].append(rec)
        print(json.dumps(rec), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--vae")
    parser.add_argument("--max-enc", type=int, default=2048, help="largest encoder length to cover (ACE-Step)")
    args = parser.parse_args()
    args.checkpoint, args.vae = args.checkpoint or "/checkpoint", args.vae or "/vae"
    seq_buckets = buckets((MIN_FRAMES + 1) // 2, (MAX_FRAMES + 1) // 2, host.SEQ_BUCKET)
    enc_buckets = buckets(1, args.max_enc, host.ENC_BUCKET)
    g = torch.Generator().manual_seed(0)
    started = time.monotonic()
    dev = acestep_dit.open_device()
    report = {"seq_buckets": seq_buckets, "enc_buckets": enc_buckets, "dit": [], "vae": []}
    try:
        dit = acestep_dit.AceStepDiT(dev, host.Checkpoint(args.checkpoint))
        vae = oobleck_vae.OobleckTT(dev, args.vae)
        report["load_s"] = round(time.monotonic() - started, 1)
        warm_vae(vae, report)
        for seq_pad in seq_buckets:
            frames = min(2 * seq_pad, MAX_FRAMES)  # any length in the bucket gives the same shapes
            x, ctx = torch.randn(frames, 64, generator=g), torch.randn(frames, 128, generator=g)
            for enc_pad in enc_buckets:
                t0 = time.monotonic()
                song = dit.prepare(torch.randn(enc_pad, 2048, generator=g), frames)
                assert (song.geo.seq_pad, song.geo.enc_pad) == (seq_pad, enc_pad)
                v = dit.forward(x, 1.0, 1.0, ctx, song)
                dit.release(song)
                rec = {"seq_pad": seq_pad, "enc_pad": enc_pad, "s": round(time.monotonic() - t0, 2),
                       "finite": bool(torch.isfinite(v).all())}
                report["dit"].append(rec)
                print(json.dumps(rec), flush=True)
    finally:
        acestep_dit.close_device(dev)
    report["total_s"] = round(time.monotonic() - started, 1)
    print(json.dumps({k: report[k] for k in ("seq_buckets", "enc_buckets", "load_s", "total_s")}), flush=True)


if __name__ == "__main__":
    main()
