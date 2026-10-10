#!/usr/bin/env python3
"""Where the P100a VAE decode spends its time: per-op timings (synchronized) for windows of the given lengths.

  run_on_card.sh vae_profile.py golden/d30 --frames 512 1024
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tt import acestep_dit, oobleck_vae  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("golden")
    parser.add_argument("--vae", default="/vae")
    parser.add_argument("--frames", type=int, nargs="+", default=[512])
    parser.add_argument("--out", default="vae_profile.json")
    parser.add_argument("--act-block-h", type=int, default=None, help="override Precision.act_block_h (0: default)")
    parser.add_argument("--transpose-act-block-h", type=int, default=None)
    parser.add_argument("--config-in-l1", action="store_true", help="keep conv config tensors in L1_SMALL")
    parser.add_argument("--no-profile", action="store_true", help="only first-call and steady decode times")
    parser.add_argument("--profile-first", action="store_true", help="per-op times of the first (building) call")
    args = parser.parse_args()
    last = torch.load(Path(args.golden) / "decoder_calls.pt")["steps"][-1]
    x0 = (last["x"] - last["v"] * last["t"].view(-1, 1, 1))[0].t().contiguous()
    dev = acestep_dit.open_device()
    result = []
    try:
        prec = oobleck_vae.Precision()
        if args.act_block_h is not None:
            prec.act_block_h = args.act_block_h
        if args.transpose_act_block_h is not None:
            prec.transpose_act_block_h = args.transpose_act_block_h
        prec.config_tensors_in_dram = not args.config_in_l1
        vae = oobleck_vae.OobleckTT(dev, args.vae, prec)
        for frames in args.frames:
            dev.clear_program_cache()
            z = x0.repeat(1, frames // x0.shape[-1] + 1)[:, :frames]
            if args.profile_first:
                vae.profile = []
            t0 = time.monotonic()
            vae.decode(z)  # first call in this process: programs and conv configs are built here
            first = time.monotonic() - t0
            if args.profile_first:
                ops, vae.profile = vae.profile, None
                agg = defaultdict(float)
                for where, op, length, ch, sec in ops:
                    agg[f"{where}:{op}"] += sec
                print(json.dumps({"frames": frames, "first_s": round(first, 2),
                                  "first_by_op": {k: round(v, 2) for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:14]}}), flush=True)
            t0 = time.monotonic()
            vae.decode(z)
            plain = time.monotonic() - t0
            if args.no_profile:
                rec = {"frames": frames, "first_s": round(first, 2), "decode_s": round(plain, 3)}
                result.append(rec)
                print(json.dumps(rec), flush=True)
                continue
            vae.profile = []
            vae.decode(z)
            ops, vae.profile = vae.profile, None
            by_kind = defaultdict(float)
            for where, op, length, ch, sec in ops:
                by_kind[op.split("_d")[0] if op.startswith("conv_k7") else op] += sec
            top = sorted(ops, key=lambda r: -r[4])[:12]
            rec = {"frames": frames, "first_s": round(first, 2), "decode_s": round(plain, 3), "profiled_sum_s": round(sum(r[4] for r in ops), 3),
                   "by_kind_s": {k: round(v, 3) for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1])},
                   "top_ops": [[w, op, length, ch, round(sec, 4)] for w, op, length, ch, sec in top],
                   "ops": [[w, op, length, ch, round(sec, 5)] for w, op, length, ch, sec in ops]}
            result.append(rec)
            print(json.dumps({k: rec[k] for k in ("frames", "decode_s", "profiled_sum_s", "by_kind_s", "top_ops")}), flush=True)
    finally:
        acestep_dit.close_device(dev)
        (Path(args.golden) / args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
