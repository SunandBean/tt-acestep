# SPDX-License-Identifier: Apache-2.0
"""HTTP server for ACE-Step 1.5 Turbo on one Blackhole p100a, everything in one process.

    uvicorn tt_acestep.server:app --host 0.0.0.0 --port 20000

ACE-Step's own pipeline runs here on the CPU and the DiT decoder and Oobleck audio VAE decoder
run on the card, joined by `LocalWorker` instead of the Unix socket `remote.py` uses. That split
existed because ACE-Step and tt-metal could not share an interpreter in the deployment this port
came from; an image built for this model alone resolves one environment that satisfies both, so
the socket is unnecessary there. See `local.py`.

Requires ACE-Step itself (the `acestep` package) importable beside this one -- it is the model,
and this package only moves two of its stages onto the card.
"""
from __future__ import annotations

import base64
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .local import LocalWorker
from .remote import RemoteDecoder, RemoteVae, skip_unused_lm_hints

LICENSE = "mit"
WEIGHTS_REPO = "ACE-Step/Ace-Step1.5"
STEPS = 8  # the turbo schedule this port compiles its shapes for
MIN_DURATION, MAX_DURATION = 30.0, 480.0
TURN_WAIT_S = float(os.environ.get("MUSIC_TURN_WAIT_S", "1800"))

# The weights, and which checkpoint inside them to load. ACE-Step's Hub repo IS the layout its
# own `checkpoints/` directory has, so one snapshot carries the DiT, the audio VAE and the text
# encoder it expects to find beside each other.
WEIGHTS_REVISION = os.environ.get("ACESTEP_REVISION", "19671f406d603126926c1b7e2adc169acbcade22")
CHECKPOINT = os.environ.get("ACESTEP_CHECKPOINT", "acestep-v15-turbo")
# ACE-Step resolves everything under <project_root>/checkpoints and writes beside it, so it
# cannot be the read-only snapshot itself; this directory holds a link to it.
ACESTEP_ROOT = os.environ.get("ACESTEP_ROOT", "/tmp/ace-step")

STATE: dict = {"status": "loading", "error": None, "generating": False, "load_s": None, "dram": None}
WORKER: Optional[LocalWorker] = None
HANDLER = None
LM = None
LOCK = Lock()


def _project_root() -> Path:
    """A writable ACE-Step project root whose `checkpoints/` is the downloaded snapshot.

    `initialize_service` takes a project root and a checkpoint name under it, and upstream's
    own layout puts the snapshot's contents there verbatim -- so the link costs nothing and
    keeps ACE-Step's path handling exactly as it ships."""
    from huggingface_hub import snapshot_download

    explicit = os.environ.get("ACESTEP_CHECKPOINTS")
    snapshot = Path(explicit) if explicit else Path(snapshot_download(WEIGHTS_REPO, revision=WEIGHTS_REVISION))
    root = Path(ACESTEP_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    link = root / "checkpoints"
    if link.is_symlink() or link.exists():
        if link.is_symlink() and link.readlink() == snapshot:
            return root
        link.unlink()
    link.symlink_to(snapshot, target_is_directory=True)
    return root


def _handler_class():
    """AceStepHandler with its bundle preflight narrowed to what this port actually loads.

    Upstream's preflight also requires the planning LM, which text2music never reaches here
    (`thinking=False`), and it downloads at runtime when something is missing. Both are wrong
    for a server whose weights arrived with the image's pull."""
    from acestep.handler import AceStepHandler

    required = {CHECKPOINT: ("config.json", "silence_latent.pt"), "vae": ("config.json",),
                "Qwen3-Embedding-0.6B": ("config.json", "tokenizer_config.json", "tokenizer.json")}

    class LocalCheckpointHandler(AceStepHandler):
        def _ensure_models_present(self, *, checkpoint_path, config_path, prefer_source, vae_variant=None):
            missing = [f"{component}/{name}"
                       for component, names in required.items() for name in names
                       if not (Path(checkpoint_path) / component / name).is_file()]
            weightless = [component for component in required
                          if not any((Path(checkpoint_path) / component).glob("*.safetensors"))]
            problems = missing + [f"{c}/weights" for c in weightless]
            if problems:
                return "ERROR: these ACE-Step assets are missing: " + ", ".join(problems), False
            return None

    return LocalCheckpointHandler


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global WORKER, HANDLER, LM
    started = time.monotonic()
    try:
        os.environ.setdefault("ACESTEP_DISABLE_TQDM", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        from acestep.llm_inference import LLMHandler

        root = _project_root()
        ckpt = root / "checkpoints" / CHECKPOINT
        WORKER = LocalWorker(ckpt).start()

        HANDLER = _handler_class()()
        status, ok = HANDLER.initialize_service(
            project_root=str(root), config_path=CHECKPOINT, device="cpu",
            use_flash_attention=False, compile_model=False,
            offload_to_cpu=False, offload_dit_to_cpu=False, quantization=None,
        )
        if not ok:
            raise RuntimeError(status)

        # The two stages that move to the card. Everything else is ACE-Step's own CPU code.
        HANDLER.model.decoder = RemoteDecoder(WORKER)
        HANDLER.tiled_decode = RemoteVae(WORKER, upsample=WORKER.upsample)
        skip_unused_lm_hints(HANDLER.model)  # text2music discards them; ~6 s of CPU for 8 minutes
        LM = LLMHandler()

        STATE.update(status="ok", load_s=round(time.monotonic() - started, 2),
                     dram=WORKER.stats.get("dram"), vae_warm_s=WORKER.stats.get("vae_warm_s"))
    except Exception as exc:
        STATE.update(status="error", error=f"{type(exc).__name__}: {exc}")
    try:
        yield
    finally:
        HANDLER = LM = None
        if WORKER is not None:
            WORKER.close()
            WORKER = None


app = FastAPI(title="ACE-Step 1.5 Turbo on p100a", lifespan=lifespan, docs_url=None, redoc_url=None)


class Request(BaseModel):
    model_config = ConfigDict(extra="ignore")
    prompt: str = Field(min_length=1, max_length=4000, description="the style caption")
    lyrics: str = Field(default="", max_length=20000)
    instrumental: bool = False
    duration: float = 120.0
    seed: Optional[int] = Field(default=None, ge=0, le=2147483647)
    inference_steps: Optional[int] = None


def _check(request: Request) -> None:
    if request.inference_steps is not None and request.inference_steps != STEPS:
        raise ValueError(f"this port runs {STEPS} steps (the turbo schedule its shapes are compiled for)")
    if not MIN_DURATION <= request.duration <= MAX_DURATION:
        raise ValueError(f"duration must be between {MIN_DURATION} and {MAX_DURATION} seconds")
    if not request.instrumental and not request.lyrics.strip():
        raise ValueError("vocals need lyrics; set instrumental=true for music without them")


@app.get("/health")
def health():
    return dict(STATE)


@app.get("/info")
def info():
    return dict(STATE) | {
        "model": WEIGHTS_REPO,
        "license": LICENSE,
        "device": "Tenstorrent Blackhole p100a",
        "on_card": ["DiT decoder", "Oobleck audio VAE decoder"],
        "on_cpu": ["text encoder", "lyric encoder", "sampler", "post-processing"],
        "inference_steps": STEPS,
        "task_modes": ["text-to-music"],
        "duration_limits": {"min": MIN_DURATION, "max": MAX_DURATION},
    }


@app.post("/predict")
def predict(request: Request):
    if not request.prompt.strip():
        raise HTTPException(400, "Empty prompt")
    try:
        _check(request)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if STATE["status"] != "ok" or HANDLER is None:
        raise HTTPException(503, STATE["error"] or "Model is not ready")
    if not LOCK.acquire(timeout=TURN_WAIT_S):
        raise HTTPException(409, "Another song is in progress")

    import tempfile

    try:
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        STATE["generating"] = True
        t0 = time.perf_counter()
        params = GenerationParams(
            caption=request.prompt,
            lyrics="[Instrumental]" if request.instrumental else request.lyrics,
            instrumental=request.instrumental, vocal_language="en",
            duration=float(request.duration), inference_steps=STEPS,
            seed=request.seed if request.seed is not None else -1,
            thinking=False, use_cot_metas=False, use_cot_caption=False, use_cot_language=False,
        )
        config = GenerationConfig(
            batch_size=1, audio_format="wav", use_random_seed=request.seed is None,
            seeds=[request.seed] if request.seed is not None else None,
        )
        with tempfile.TemporaryDirectory() as out:
            result = generate_music(HANDLER, LM, params, config, save_dir=out)
            if not result.success or not result.audios:
                raise RuntimeError(result.error or result.status_message or "generation failed")
            audio = Path(result.audios[0]["path"])
            if not audio.is_file():
                raise RuntimeError("generation produced no audio file")
            wav = audio.read_bytes()

        decoder, vae = HANDLER.model.decoder, HANDLER.tiled_decode
        # The card holds no song once the VAE has run, so the decoder's "already prepared" cache
        # has to go with it -- the next request's conditioning differs and would otherwise be
        # skipped. A per-song decoder would not need this; a server keeps one across songs.
        decoder.reset()
        timing = {
            "dit_s": round(sum(decoder.seconds), 2),
            "dit_calls": len(decoder.seconds),
            "vae_s": round(sum(vae.seconds), 2),
            "total_s": round(time.perf_counter() - t0, 2),
        }
        decoder.seconds.clear()
        vae.seconds.clear()
        return {
            "audio": base64.b64encode(wav).decode(),
            "format": "wav",
            "model": WEIGHTS_REPO,
            "license": LICENSE,
            "seed": request.seed,
            "duration": request.duration,
            "inference_steps": STEPS,
            "timing_s": timing,
        }
    except Exception as exc:
        if HANDLER is not None:
            HANDLER.model.decoder.reset()
        if WORKER is not None and WORKER.failed:
            STATE.update(status="error", error=str(exc))
        raise HTTPException(503, str(exc)) from exc
    finally:
        STATE["generating"] = False
        LOCK.release()
