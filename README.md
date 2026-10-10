# tt-acestep

**ACE-Step 1.5 Turbo** ported to a single **Tenstorrent Blackhole p100a**. The DiT decoder and the
Oobleck audio VAE decoder run on the card in TTNN.

Model card and the full numbers:
**[sunandbean/acestep-1.5-turbo-p100a](https://huggingface.co/sunandbean/acestep-1.5-turbo-p100a)**

**What this port adds** — the DiT decoder and the Oobleck audio VAE decoder on the card, a transport that works whether or not ACE-Step's environment and the tt-metal runtime can share an interpreter, and the shape bucketing that stops a song length from compiling kernels of its own.
**What it builds on** — ACE-Step's own pipeline (MIT), which runs unchanged on the CPU; only those two stages move.

**A 30-second song in 3.3 s** in one process (DiT 0.37 s, audio VAE decode 0.89 s), or 6.8 s split
across a socket. An 8-minute song takes about 19 s split, the VAE decode growing to 10.8 s.

On an RTX 5070 Ti the same 30-second song takes 3.5–3.6 s through ACE-Step's own GPU path, so the
two are interchangeable at this length. Numbers and method on the model card.

## One transport, two homes for it

`RemoteDecoder` and `RemoteVae` reach the card through exactly one method, `worker.call(msg)`, and
there are two things that answer it. `TTWorker` runs the card in its own container behind a Unix
socket, for when ACE-Step's environment and the tt-metal runtime cannot share an interpreter — the
deployment this port came from. `LocalWorker` answers in-process, for when they can: ACE-Step's
dependency set resolves beside ttnn and the pipeline runs, so its `torch==2.10.0+cu128` pin turns
out to be a convenience rather than an API requirement. The published image lands on
`torch 2.12.1+cpu` with `numpy 2.5.3`. The shims are the same code either way.

```
  ACE-Step venv (CPU)                      TT image (p100a)
  ───────────────────                      ────────────────
  text + lyric encoders        ──socket──► worker.py
  sampler loop                             AceStepDiT     (on the card)
  post-processing                          OobleckTT      (on the card)
```

`remote.py` swaps `model.decoder` for a `RemoteDecoder` and `tiled_decode` for a `RemoteVae`, so the
upstream pipeline runs **unmodified** and simply calls across the socket. Tensors cross as raw
little-endian bytes with a dtype and shape (`wire.py`) — a pickled torch tensor would not survive the
version gap.

## Accuracy

The torch reference in this port against upstream ACE-Step is **PCC 0.99999999997** per diffusion step —
the reimplementation is exact, so everything below is the device's numerics alone.

On the card against the CPU float32 pipeline (30 s test song): audio **log-mel PCC 0.9966**, MAE under
1 dB. The audio VAE decoder is PCC 0.99994 against the CPU, which is the same level a CPU bf16 VAE
reaches (0.99996) — and bf16 is what the GPU path uses too.

Seeds do **not** reproduce across backends. A song made here cannot be reproduced bit-for-bit on a GPU.

## Install

```bash
pip install -e .
```

Host side, inside the ACE-Step environment:

```python
from tt_acestep import TTWorker, RemoteDecoder, RemoteVae
from tt_acestep.remote import skip_unused_lm_hints

worker = TTWorker(checkpoint_dir).start()
worker.wait_ready()
handler.model.decoder = RemoteDecoder(worker)
handler.tiled_decode = RemoteVae(worker)
skip_unused_lm_hints(handler.model)
```

Worker side, inside the tt-metal image:

```bash
python -m tt_acestep.worker --socket /ipc/worker.sock --checkpoint /checkpoint --vae /vae
python -m tt_acestep.warm_cache     # compile every bucket ahead of time
```

See [`PYTHON.md`](PYTHON.md) for the protocol and the full API.

## Layout

| Path | |
|---|---|
| `tt_acestep/` | the port — host side, worker side, DiT, VAE, transport |
| `tt-model.yaml` | the container manifest the published image is built from — `tt-model package --container tt-model.yaml` |
| `PYTHON.md` | API reference and the socket protocol |
| `experiments/` | the verification scripts behind the published numbers — most import this port under the name it had in the private tree it was written in, so read [`experiments/README.md`](experiments/README.md) before running them |

## Related

`oobleck_vae.py` implements both ACE-Step's diffusers Oobleck VAE and DiffRhythm's stable-audio-tools
variant. The DiffRhythm port is [tt-diffrhythm](https://github.com/SunandBean/tt-diffrhythm), which
shares this worker and transport.

## Licence

Port code Apache-2.0; the ACE-Step 1.5 weights are MIT and are not redistributed here.
