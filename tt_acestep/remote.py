"""Host side of the P100a models: start the TT worker container on the card and stand in for the DiT and
the audio VAE decoder of the host pipeline.

Used from the ACE-Step host pipeline:
  worker = TTWorker(checkpoint_dir).start()   # starts the worker container on the card
  ...                                          # load the rest of ACE-Step on the CPU meanwhile
  worker.wait_ready()
  handler.model.decoder = RemoteDecoder(worker)
  handler.tiled_decode = RemoteVae(worker)
  ...
  worker.close()                               # the card is free only after the container has exited

The DiffRhythm port is the sibling repo, tt-diffrhythm: it shares this worker and transport.

Runs in the model's own environment; needs only the standard library and torch.
"""
from __future__ import annotations

from multiprocessing.connection import Client
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import tempfile
import time

import torch

from . import wire
from .shapes import VAE_WINDOWS

ROOT = Path(__file__).resolve().parents[1]
# The tt-metal image the worker runs in. There is no portable default: set MUSIC_TT_IMAGE,
# or pass image= , to the tt-metal / ttnn image you built the port against.
DEFAULT_IMAGE = ""


class WorkerError(RuntimeError):
    """The P100a worker failed (died, hung or reported an error)."""


class TTWorker:
    """checkpoint is the ACE-Step DiT checkpoint directory; its sibling vae/ holds the audio VAE."""

    def __init__(self, checkpoint: Path, image: str = None, log=None,
                 reply_timeout_s: float = 120.0,  # the slowest call (an 8-minute song's DiT step or VAE window) is < 1 s
                 model: str = "acestep", vae: Path = None):
        self.checkpoint = Path(checkpoint)
        self.model = model
        self.vae = Path(vae) if vae is not None else None
        self.image = image or os.environ.get("MUSIC_TT_IMAGE") or DEFAULT_IMAGE
        if not self.image:
            raise ValueError("set MUSIC_TT_IMAGE (or pass image=) to the tt-metal image the worker runs in")
        self.log = log
        self.reply_timeout_s = reply_timeout_s
        self.proc = self.conn = self.ipc = None
        self.name = f"music-tt-{os.getpid()}-{secrets.token_hex(3)}"
        self.stats = {}
        self.failed = False  # set when the worker died, hung or reported an error: the card may be unhealthy

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "TTWorker":
        try:
            cache = ROOT / "data/tt-cache"
            cache.mkdir(parents=True, exist_ok=True)
            run_dir = ROOT / "data/tt-run"
            run_dir.mkdir(parents=True, exist_ok=True)
            self.ipc = Path(tempfile.mkdtemp(prefix="dit-", dir=run_dir))
            self.authkey = secrets.token_hex(16)
            mounts = [("/dev/hugepages-1G", "/dev/hugepages-1G", False), (ROOT, "/work", True),
                      *self._model_mounts(), (cache, "/cache", False), (self.ipc, "/ipc", False)]
            cmd = ["docker", "run", "--rm", "--name", self.name, "--ipc", "host", "--device", "/dev/tenstorrent"]
            for src, dst, ro in mounts:
                cmd += ["--mount", f"type=bind,src={src},dst={dst}" + (",readonly" if ro else "")]
            for k, v in {"TT_METAL_VISIBLE_DEVICES": "0", "MESH_DEVICE": "P100", "PYTHONUNBUFFERED": "1",
                         "TT_METAL_OPERATION_TIMEOUT_SECONDS": "90", "MUSIC_TT_AUTHKEY": self.authkey}.items():
                cmd += ["-e", f"{k}={v}"]
            cmd += [self.image, "python", "/work/tt_acestep/worker.py", "--socket", "/ipc/worker.sock", "--model", self.model]
            # own session: a supervisor may stop the host process by signalling its process group, and a signal reaching
            # `docker run` directly would stop the container while this process still has to wait for it
            self.proc = subprocess.Popen(cmd, stdout=self.log or subprocess.DEVNULL, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            self._started = time.monotonic()
        except BaseException:
            self.close()
            raise
        return self

    def _model_mounts(self):
        if self.model == "acestep":
            return [(self.checkpoint, "/checkpoint", True), (self.checkpoint.parent / "vae", "/vae", True)]
        raise ValueError(f"unknown model {self.model!r}")

    def died(self) -> bool:
        """The worker container exited on its own (not through close())."""
        return self.proc is not None and self.proc.poll() is not None

    def _fail(self, message):
        self.failed = True
        raise WorkerError(message)

    def wait_ready(self, timeout_s: float = 300.0):
        sock = self.ipc / "worker.sock"
        deadline = time.monotonic() + timeout_s
        while not sock.exists():
            self._check_alive()
            if time.monotonic() > deadline:
                self._fail("TT worker did not open its socket")
            time.sleep(0.2)
        try:
            self.conn = Client(str(sock), family="AF_UNIX", authkey=self.authkey.encode())
        except (OSError, EOFError) as exc:  # the worker died while loading
            self._fail(f"TT worker connection failed: {exc}")
        try:
            ready = self._recv(timeout_s)
        except (OSError, EOFError) as exc:
            self._fail(f"TT worker connection lost: {exc}")
        self.stats.update(worker_load_s=round(ready["load_s"], 2), start_to_ready_s=round(time.monotonic() - self._started, 2))
        return ready

    def close(self):
        """Stop the worker and wait for the container to exit; only then give the card back."""
        try:
            if self.conn is not None:
                try:
                    self.conn.send({"op": "close"})
                except OSError:
                    pass
                self.conn.close()
            if self.proc is not None:
                try:
                    self.proc.wait(60)
                except subprocess.TimeoutExpired:
                    subprocess.run(["docker", "stop", "-t", "20", self.name], capture_output=True, timeout=60)
                    self.proc.wait(60)
        finally:
            self.conn = None
            if self.ipc is not None:
                shutil.rmtree(self.ipc, ignore_errors=True)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ calls
    def _check_alive(self):
        if self.proc is not None and self.proc.poll() is not None:
            self._fail(f"TT worker exited ({self.proc.returncode}); see the worker log")

    def _recv(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        while not self.conn.poll(1.0):
            self._check_alive()
            if time.monotonic() > deadline:
                self._fail("TT worker did not answer in time")
        msg = self.conn.recv()
        if msg.get("op") == "error":
            self._fail(msg["error"])
        return msg

    def call(self, msg: dict) -> dict:
        try:
            self.conn.send(msg)
            return self._recv(self.reply_timeout_s)
        except (OSError, EOFError) as exc:  # the socket broke: the worker is gone
            self._fail(f"TT worker connection lost: {exc}")


class RemoteDecoder(torch.nn.Module):
    """Drop-in for AceStepDiTModel in generate_audio (batch 1): same call, output (velocity, past_key_values)."""

    def __init__(self, worker: TTWorker):
        super().__init__()
        self.worker = worker
        self._enc = self._ctx = None
        self.seconds = []

    def reset(self) -> None:
        """Forget which song is prepared on the card.

        The worker releases the song's tensors when the VAE runs, so that every song length meets
        the same device allocation state and the VAE's compiled kernels are reused. After that the
        card holds no song, and this cache -- which exists to skip re-sending an unchanged
        conditioning -- would otherwise send a `forward` for a song that is no longer there.

        A caller that builds a decoder per song never notices. One that keeps a decoder across
        songs, as a server does, has to call this between them.
        """
        self._enc = self._ctx = None

    def forward(self, hidden_states, timestep, timestep_r, attention_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, context_latents=None, use_cache=None, past_key_values=None, **kwargs):
        if hidden_states.shape[0] != 1:
            raise ValueError("The P100a decoder runs batch 1")
        enc, ctx = encoder_hidden_states[0].float(), context_latents[0].float()
        if self._enc is None or not (torch.equal(enc, self._enc) and torch.equal(ctx, self._ctx)):
            self.worker.call({"op": "prepare", "enc": wire.encode(enc), "ctx": wire.encode(ctx)})
            self._enc, self._ctx = enc.clone(), ctx.clone()
        reply = self.worker.call({"op": "forward", "x": wire.encode(hidden_states[0]), "t": float(timestep[0]),
                                  "t_r": float(timestep_r[0])})
        self.seconds.append(reply["seconds"])
        v = wire.decode(reply["v"]).to(hidden_states.dtype)[None]
        return v, past_key_values


class RemoteVae:
    """Stands in for handler.tiled_decode: overlap-discard tiling like upstream's, but every window of a song has the
    same length (the last one is moved back to end at the song's end), so the P100a compiles only a few shapes.

    The window is the largest of `chunks` that fits the song: 1024 frames (41 s) decode at 0.61 ms per frame with
    12.5 % overlap, 512 frames at the same rate with 25 % overlap (vae_profile.py); longer windows get slower per
    frame. Songs of 30-41 s use 512, so no song compiles a shape of its own.

    latents [B, 64, T] -> audio [B, 2, T * upsample] float32 on the CPU (1920 for ACE-Step, 2048 for DiffRhythm)."""

    def __init__(self, worker: TTWorker, chunks=VAE_WINDOWS, overlap: int = 64, upsample: int = 1920):
        self.worker, self.chunks, self.overlap, self.upsample = worker, tuple(sorted(chunks, reverse=True)), overlap, upsample
        self.seconds = []

    def chunk_for(self, frames: int) -> int:
        return next((c for c in self.chunks if c <= frames), self.chunks[-1])

    @staticmethod
    def windows(frames: int, chunk: int, overlap: int):
        """(window_start, core_start, core_end) covering [0, frames); windows are exactly `chunk` frames long
        (or the whole song when it is shorter) and keep >= overlap frames of context around each core."""
        if frames <= chunk:
            return [(0, 0, frames)]
        stride = chunk - 2 * overlap
        out = []
        for core_start in range(0, frames, stride):
            core_end = min(core_start + stride, frames)
            out.append((min(max(core_start - overlap, 0), frames - chunk), core_start, core_end))
        return out

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        reply = self.worker.call({"op": "vae", "z": wire.encode(z)})
        self.seconds.append(reply["seconds"])
        return wire.decode(reply["audio"])

    def __call__(self, latents, chunk_size=None, overlap=None, offload_wav_to_cpu=None):
        out = []
        for z in latents.detach().float().cpu():
            frames = z.shape[-1]
            chunk = self.chunk_for(frames)
            pieces = []
            for win_start, core_start, core_end in self.windows(frames, chunk, self.overlap):
                win_end = min(win_start + chunk, frames)
                audio = self._decode(z[:, win_start:win_end].contiguous())
                pieces.append(audio[:, (core_start - win_start) * self.upsample:(core_end - win_start) * self.upsample])
            out.append(torch.cat(pieces, dim=-1))
        return torch.stack(out)


def skip_unused_lm_hints(model):
    """Make model.prepare_condition skip work whose result text2music throws away.

    Upstream always runs the audio tokenizer and detokenizer over the whole song to build LM hints, then keeps
    them only where is_covers > 0 (torch.where). Without covers, precomputed hints or audio codes the hints are
    discarded, yet on the CPU they cost about 6 s for an 8-minute song. The outputs are unchanged: the encoder
    states and context latents are computed exactly as upstream does. Returns the model."""
    original = model.prepare_condition

    def prepare_condition(*, text_hidden_states, text_attention_mask, lyric_hidden_states, lyric_attention_mask,
                          refer_audio_acoustic_hidden_states_packed, refer_audio_order_mask, hidden_states,
                          attention_mask, silence_latent, src_latents, chunk_masks, is_covers,
                          precomputed_lm_hints_25Hz=None, audio_codes=None):
        if precomputed_lm_hints_25Hz is not None or audio_codes is not None or bool((is_covers > 0).any()):
            return original(text_hidden_states=text_hidden_states, text_attention_mask=text_attention_mask,
                            lyric_hidden_states=lyric_hidden_states, lyric_attention_mask=lyric_attention_mask,
                            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
                            refer_audio_order_mask=refer_audio_order_mask, hidden_states=hidden_states,
                            attention_mask=attention_mask, silence_latent=silence_latent, src_latents=src_latents,
                            chunk_masks=chunk_masks, is_covers=is_covers,
                            precomputed_lm_hints_25Hz=precomputed_lm_hints_25Hz, audio_codes=audio_codes)
        encoder_hidden_states, encoder_attention_mask = model.encoder(
            text_hidden_states=text_hidden_states, text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states, lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask=refer_audio_order_mask)
        context_latents = torch.cat([src_latents, chunk_masks.to(hidden_states.dtype)], dim=-1)
        return encoder_hidden_states, encoder_attention_mask, context_latents

    model.prepare_condition = prepare_condition
    return model
