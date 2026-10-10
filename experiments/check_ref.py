#!/usr/bin/env python3
"""RefDecoder (tt_acestep/acestep_host.py, float32 CPU, TT formulation) against the upstream decoder calls recorded by
dump_reference.py. Every step is fed the recorded input (teacher forcing), so errors do not accumulate.

  vendor/ACE-Step-1.5/.venv/bin/python experiments/tt-acestep/check_ref.py experiments/tt-acestep/golden/d30
"""
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tt import acestep_host as host  # noqa: E402

CHECKPOINT = ROOT / "vendor/ACE-Step-1.5/checkpoints/acestep-v15-turbo"


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    golden_dir = Path(sys.argv[1])
    golden = torch.load(golden_dir / "decoder_calls.pt")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    ref = host.RefDecoder(host.Checkpoint(CHECKPOINT))
    enc, ctx = golden["encoder_hidden_states"][0], golden["context_latents"][0]
    geo = host.Geometry(frames=ctx.shape[0], enc_len=enc.shape[0])
    kv = ref.encode(enc, geo)
    steps = []
    with torch.inference_mode():
        for s in golden["steps"]:
            started = time.monotonic()
            v = ref.forward(s["x"][0], float(s["t"][0]), float(s["t_r"][0]), ctx, kv, geo)
            steps.append({"t": float(s["t"][0]), "pcc": pcc(v, s["v"][0]),
                          "max_abs": float((v - s["v"][0]).abs().max()), "seconds": round(time.monotonic() - started, 2)})
            print(json.dumps(steps[-1]), flush=True)
    result = {"geometry": {"frames": geo.frames, "seq": geo.seq, "seq_pad": geo.seq_pad, "enc_len": geo.enc_len},
              "min_pcc": min(s["pcc"] for s in steps), "steps": steps}
    (golden_dir / "check_ref.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("geometry", "min_pcc")}))


if __name__ == "__main__":
    main()
