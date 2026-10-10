#!/usr/bin/env python3
"""Compare a P100a run (dump_reference.py --tt) with the CPU golden run of the same request.

Free-running, unlike check_ref/device_check: every step starts from the P100a's own previous result, so this shows
how the per-step error accumulates into the latents and the audio.

  vendor/ACE-Step-1.5/.venv/bin/python experiments/tt-acestep/compare_runs.py golden/d30 golden/d30-tt
"""
import json
from pathlib import Path
import sys

import torch
import torchaudio


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def log_mel(wav, sr):
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=2048, hop_length=512, n_mels=128)(wav.mean(0))
    return torch.log(mel + 1e-5)


def main():
    ref_dir, tt_dir = Path(sys.argv[1]), Path(sys.argv[2])
    ref, tt = torch.load(ref_dir / "decoder_calls.pt"), torch.load(tt_dir / "decoder_calls.pt")
    steps = []
    for a, b in zip(ref["steps"], tt["steps"]):
        steps.append({"t": float(a["t"][0]), "input_pcc": pcc(a["x"], b["x"]), "velocity_pcc": pcc(a["v"], b["v"])})
    wa, sra = torchaudio.load(str(ref_dir / "reference.wav"))
    wb, srb = torchaudio.load(str(tt_dir / "reference.wav"))
    n = min(wa.shape[1], wb.shape[1])
    ma, mb = log_mel(wa[:, :n], sra), log_mel(wb[:, :n], srb)
    result = {"encoder_equal": torch.equal(ref["encoder_hidden_states"], tt["encoder_hidden_states"]),
              "steps": steps, "final_step_input_pcc": steps[-1]["input_pcc"],
              "audio": {"waveform_pcc": pcc(wa[:, :n], wb[:, :n]), "log_mel_pcc": pcc(ma, mb),
                        "log_mel_mae_db": float((ma - mb).abs().mean() * 10 / torch.log(torch.tensor(10.0))),
                        "rms_ref": float(wa.pow(2).mean().sqrt()), "rms_tt": float(wb.pow(2).mean().sqrt())},
              "meta_ref": json.loads((ref_dir / "meta.json").read_text()),
              "meta_tt": json.loads((tt_dir / "meta.json").read_text())}
    (tt_dir / "compare.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("encoder_equal", "final_step_input_pcc", "audio")}, indent=1))
    for s in steps:
        print(json.dumps(s))


if __name__ == "__main__":
    main()
