# SPDX-License-Identifier: Apache-2.0
"""The card in THIS process, behind the same `.call(msg)` surface `TTWorker` gives.

`RemoteDecoder` and `RemoteVae` reach the card through exactly one method -- `worker.call(msg)` --
so swapping the transport is enough to run everything in one process: the shims, their window
arithmetic and their drop-in signatures are reused unchanged rather than reimplemented here.

Why there are two transports at all: ACE-Step pins `torch==2.10.0+cu128` and resolves `numpy>=2`,
while ttnn's extension modules are built against tt-metal's own `torch 2.11.0+cpu` and `numpy 1.x`.
Two environments cannot be one process, so the deployment this port came from ran ACE-Step on the
host and the card in a container, joined by a Unix socket (`remote.py` / `worker.py`). A container
built for this model alone has no such split -- one environment satisfies both -- and then the
socket is pure cost. Use `LocalWorker` there, `TTWorker` when the two environments must stay apart.

    from tt_acestep import LocalWorker, RemoteDecoder, RemoteVae
    from tt_acestep.remote import skip_unused_lm_hints

    worker = LocalWorker(checkpoint_dir).start()
    handler.model.decoder = RemoteDecoder(worker)
    handler.tiled_decode = RemoteVae(worker, upsample=worker.upsample)
    skip_unused_lm_hints(handler.model)
    ...
    worker.close()
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch

from . import acestep_dit, oobleck_vae, wire
from .shapes import VAE_WINDOWS
from .worker import AceStep

MODELS = {"acestep": AceStep}


class LocalWorker:
    """`TTWorker`'s surface, answered in-process. `checkpoint` is the DiT checkpoint directory;
    `vae` defaults to its sibling `vae/`, which is the layout the upstream checkpoint ships."""

    def __init__(self, checkpoint, vae=None, model: str = "acestep", warm: bool = True):
        self.checkpoint = Path(checkpoint)
        self.vae_path = Path(vae) if vae is not None else self.checkpoint.parent / "vae"
        self.model_name = model
        self.warm = warm
        self.dev = None
        self.model = self.vae = None
        self.song = None
        self.stats: dict = {}
        self.failed = False
        self.upsample: Optional[int] = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "LocalWorker":
        started = time.monotonic()
        self.dev = acestep_dit.open_device()
        try:
            self.model = MODELS[self.model_name](self.dev, str(self.checkpoint))
            self.vae = oobleck_vae.OobleckTT(self.dev, str(self.vae_path))
            self.upsample = self.vae.upsample
            if self.warm:
                # Same reason worker.main() warms here: built now, the VAE's conv kernels compile
                # in one device state and every later decode reuses the in-process programs. Built
                # after a song's DiT steps instead, the state -- and so the kernels -- depends on
                # the song length and recompiles for 20-40 s.
                t0 = time.monotonic()
                for window in VAE_WINDOWS:
                    self.vae.decode(torch.zeros(64, window))
                self.stats["vae_warm_s"] = round(time.monotonic() - t0, 2)
            self.stats["load_s"] = round(time.monotonic() - started, 2)
            self.stats["dram"] = acestep_dit.dram_stats(self.dev)
        except BaseException:
            self.close()
            raise
        return self

    def wait_ready(self, timeout_s: float = 0.0) -> dict:
        """`TTWorker.wait_ready`'s shape. Loading already finished in `start`, so this only reports."""
        if self.model is None:
            raise RuntimeError("LocalWorker.start() has not run")
        return {"op": "ready", "model": self.model_name, "upsample": self.upsample, **self.stats}

    def died(self) -> bool:
        """There is no child process to lose; a failure here raised in the caller's own stack."""
        return False

    def close(self) -> None:
        if self.song is not None and self.model is not None:
            self.model.dit.release(self.song)
            self.song = None
        self.model = self.vae = None
        if self.dev is not None:
            acestep_dit.close_device(self.dev)
            self.dev = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ calls
    def call(self, msg: dict) -> dict:
        """`worker.py:serve`'s dispatch with the socket taken out and nothing else changed.

        Messages still carry `wire`-encoded tensors even though both ends are now one process.
        That is deliberate: it is what lets `RemoteDecoder` and `RemoteVae` run against this
        worker unmodified, and the encoding is a float32 copy of a few MB against card work
        measured in seconds."""
        if self.model is None:
            raise RuntimeError("LocalWorker.start() has not run")
        op = msg.get("op")
        started = time.monotonic()
        try:
            if op == "prepare":
                if self.song is not None:
                    self.model.dit.release(self.song)
                    self.song = None
                self.song, geometry = self.model.prepare(msg)
                return {"op": "prepared", "seconds": time.monotonic() - started, "geometry": geometry}
            if op == "forward":
                if self.song is None:
                    raise RuntimeError("forward before prepare")
                v = self.model.forward(msg, self.song)
                return {"op": "velocity", "v": wire.encode(v), "seconds": time.monotonic() - started}
            if op == "vae":
                if self.song is not None:
                    # Sampling is over; freeing the song's tensors first gives the VAE the same
                    # device allocation state for every song length, so its compiled conv kernels
                    # are reused instead of recompiled.
                    self.model.dit.release(self.song)
                    self.song = None
                audio = self.vae.decode(wire.decode(msg["z"]))
                return {"op": "audio", "audio": wire.encode(audio), "seconds": time.monotonic() - started}
            if op == "close":
                self.close()
                return {"op": "closed", "seconds": time.monotonic() - started}
        except BaseException:
            self.failed = True  # the card may be unhealthy, same contract as TTWorker
            raise
        raise RuntimeError(f"unknown op {op!r}")
