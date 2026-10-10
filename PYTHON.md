# tt_acestep — Python reference

ACE-Step 1.5 Turbo on one Tenstorrent Blackhole p100a. The DiT decoder and the Oobleck audio VAE
decoder run on the card; everything else stays on the CPU in ACE-Step's own environment.

## Two processes

ACE-Step's environment and the tt-metal runtime need different torch and numpy builds, so they run as
two processes talking over a Unix socket in a directory bind-mounted into both. Tensors cross as raw
little-endian bytes with a dtype and a shape (`wire.py`) — a pickled torch tensor would not survive
the version gap.

## Host side (inside the ACE-Step environment)

### `TTWorker(checkpoint, image=None, log=None, reply_timeout_s=120.0, model="acestep", vae=None)`

Starts the worker container on the card. `image` is the tt-metal / ttnn image the port was built
against; it has no portable default, so set it here or through `MUSIC_TT_IMAGE`. `.start()` returns self,
`.wait_ready(timeout_s)` blocks until the weights are on the device, `.close()` stops the container and
waits for it to exit — a process that has closed the device can still hold the handle until it exits, so
the card is free only after that. `.died()` reports an unexpected container exit.

### `RemoteDecoder(worker)`

A `torch.nn.Module` drop-in for `model.decoder`. Assign it and the upstream pipeline runs unchanged.

### `RemoteVae(worker, upsample=1920)`

A drop-in for `tiled_decode`: `[B, 64, T]` latents to `[B, 2, T * upsample]` float32 audio on the CPU.
It picks the largest VAE window from `shapes.VAE_WINDOWS` that fits the song.

### `skip_unused_lm_hints(model)`

Drops the LM hints `text2music` builds over the whole song on the CPU and then discards. Output is
bit-identical; this is a pure saving. It patches the upstream model object.

## Worker side (inside the tt-metal image)

```bash
python -m tt_acestep.worker --socket /ipc/worker.sock --checkpoint /checkpoint --vae /vae
```

Protocol (plain dicts, tensors via `wire`):

| | |
|---|---|
| `-> {"op": "ready", ...}` | once the weights are on the device |
| `<- {"op": "prepare", "enc": [L, 2048], "ctx": [T, 128]}` | `-> {"op": "prepared", ...}` |
| `<- {"op": "forward", "x": [T, 64], "t": float, "t_r": float}` | `-> {"op": "velocity", "v": [T, 64]}` |
| `<- {"op": "decode", ...}` | `-> audio` |

## Direct use on the card

```python
from tt_acestep import AceStepDiT, OobleckTT, open_device, close_device
from tt_acestep.acestep_host import Checkpoint

dev = open_device()
dit = AceStepDiT(dev, Checkpoint(checkpoint_dir))
vae = OobleckTT(dev, vae_dir)
...
close_device(dev)
```

## Kernel cache

```bash
python -m tt_acestep.warm_cache
```

Song lengths fall into 12 DiT buckets and 4 encoder buckets (padding is masked out) plus the 2 VAE
windows in `shapes.VAE_WINDOWS`. Warming compiles all of them: 3 minutes from empty, 1 minute
otherwise. Run it after changing the code or the image, or the first song of each shape pays for it.

## Modules

| Module | Role |
|---|---|
| `remote.py` | host side: `TTWorker`, `RemoteDecoder`, `RemoteVae`, the worker container |
| `worker.py` | worker side: loads the DiT and the VAE and answers one host pipeline |
| `wire.py` | the tensor transport between the two torch builds |
| `shapes.py` | shapes both sides need, with no torch or ttnn import |
| `acestep_dit.py` | the DiT decoder on the card |
| `acestep_host.py` | checkpoint access and the host tensor helpers |
| `oobleck_vae.py` | the audio VAE decoder on the card |
| `warm_cache.py` | ahead-of-time kernel compilation for every bucket |
