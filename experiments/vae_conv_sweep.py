#!/usr/bin/env python3
"""Conv settings for the VAE's hottest layers on the P100a: speed and error against torch (CPU, float32).

Layers from a 512-frame window: block 4's dilated k7 convs (128 channels, 983040 samples) and the output conv
(128 -> 2). Real weights; activations drawn with the VAE's scale. Variants: math fidelity, DRAM width slices,
activation block height, double buffering, bfp8 weights.

  run_on_card.sh vae_conv_sweep.py golden/d30
"""
import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tt import acestep_dit, oobleck_vae  # noqa: E402
import ttnn  # noqa: E402

MEM = ttnn.DRAM_MEMORY_CONFIG


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def run(dev, x_host, w, b, length, k, pad, dilation, fidelity, slices, act_h, double, weights_dtype):
    ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=fidelity, math_approx_mode=False,
                                                fp32_dest_acc_en=True, packer_l1_acc=False)
    cfg = ttnn.Conv2dConfig(weights_dtype=weights_dtype, enable_act_double_buffer=double,
                            enable_weights_double_buffer=double)
    if act_h:
        cfg.act_block_h_override = act_h
    slice_cfg = None if slices is None else ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=slices)
    weight = ttnn.from_torch(w.unsqueeze(2).contiguous(), dtype=ttnn.float32)
    bias = ttnn.from_torch(b.reshape(1, 1, 1, -1), dtype=ttnn.float32) if b is not None else None
    times, out_host = [], None
    for _ in range(3):
        x = ttnn.from_torch(x_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=MEM)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        out, (weight, bias) = ttnn.conv1d(
            input_tensor=x, weight_tensor=weight, bias_tensor=bias, device=dev, in_channels=w.shape[1],
            out_channels=w.shape[0], batch_size=1, input_length=length, kernel_size=k, stride=1, padding=pad,
            dilation=dilation, groups=1, dtype=ttnn.bfloat16, conv_config=cfg, compute_config=ck,
            slice_config=slice_cfg, return_weights_and_bias=True)
        ttnn.synchronize_device(dev)
        times.append(time.perf_counter() - t0)
        out_host = ttnn.to_torch(out).reshape(length, -1)[:, : w.shape[0]].float()
        ttnn.deallocate(out)
        ttnn.deallocate(x)
    return min(times[1:]), out_host


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("golden")
    parser.add_argument("--vae", default="/vae")
    parser.add_argument("--length", type=int, default=983040)
    parser.add_argument("--out", default="vae_conv_sweep.json")
    args = parser.parse_args()
    weights = oobleck_vae.VaeWeights(args.vae)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(args.length, 128, generator=g) * 0.5
    layers = []
    for name, dilation in (("block.4.res_unit1.conv1", 1), ("block.4.res_unit3.conv1", 9)):
        w, b = weights.conv(name)
        layers.append((name, w, b, 7, 3 * dilation, dilation))
    w, b = weights.conv("conv2")
    layers.append(("conv2", w, b, 7, 3, 1))
    fid = ttnn.MathFidelity
    variants = [dict(fidelity=f, slices=None, act_h=0, double=False, weights_dtype=ttnn.bfloat16)
                for f in (fid.HiFi4, fid.HiFi2, fid.LoFi)]
    variants += [dict(fidelity=fid.HiFi2, slices=s, act_h=0, double=False, weights_dtype=ttnn.bfloat16) for s in (4, 8, 16, 32)]
    variants += [dict(fidelity=fid.HiFi2, slices=None, act_h=h, double=d, weights_dtype=ttnn.bfloat16)
                 for h, d in itertools.product((0, 64, 128, 256), (False, True)) if h or d]
    variants += [dict(fidelity=fid.HiFi2, slices=None, act_h=0, double=True, weights_dtype=ttnn.bfloat8_b)]
    dev = acestep_dit.open_device()
    results = []
    try:
        for name, w, b, k, pad, dilation in layers:
            ref = torch.nn.functional.conv1d(x.t()[None], w, b, padding=pad, dilation=dilation)[0].t()
            x_host = x.reshape(1, 1, args.length, 128)
            for v in variants:
                dev.clear_program_cache()
                label = {"layer": name, "fidelity": str(v["fidelity"]).split(".")[-1], "slices": v["slices"],
                         "act_h": v["act_h"], "double": v["double"], "weights": str(v["weights_dtype"]).split(".")[-1]}
                try:
                    sec, out = run(dev, x_host, w, b, args.length, k, pad, dilation, **v)
                    rec = label | {"s": round(sec, 4), "pcc": round(pcc(out, ref), 6),
                                   "rel_err": round(float((out - ref).norm() / ref.norm()), 5)}
                except Exception as exc:  # an invalid combination: record and go on
                    rec = label | {"error": str(exc).splitlines()[0][:160]}
                results.append(rec)
                print(json.dumps(rec), flush=True)
    finally:
        acestep_dit.close_device(dev)
        (Path(args.golden) / args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
