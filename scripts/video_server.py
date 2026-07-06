#!/usr/bin/env python3
"""Minimal OpenAI-style text/image-to-video API server using diffusers (LTX-Video).

Serves POST /v1/videos/generations and GET /health, mirroring the shape of
scripts/diffusion_server.py so the video MCP server (mcp_servers/video_gen_server.py)
can talk to it exactly like the image one. Returns the clip as base64-encoded mp4
in {"data": [{"b64_json": ...}]}.

Model: LTX-Video (Lightricks). Chosen over Wan 2.1 for the single-GPU 12GB
(RTX 3060) path — LTX has a first-class diffusers LTXPipeline + a distilled
variant, and enable_model_cpu_offload() keeps peak VRAM within 12GB. Favor the
distilled model for speed:
    Lightricks/LTX-Video-0.9.7-distilled

Usage:
    python3 scripts/video_server.py --model Lightricks/LTX-Video-0.9.7-distilled --port 8102 --cpu-offload

The script is mounted at runtime by docker/video (not copied), so edits are
picked up on container restart — same convention as the diffusion sidecar.
"""
import argparse
import base64
import io
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("video_server")

_pipe = None
_i2v_pipe = None
_model_id = ""
_args = None
DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@asynccontextmanager
async def lifespan(application):
    load_model()
    yield


app = FastAPI(title="Video Server", lifespan=lifespan)

# Same conservative, server-to-server security posture as diffusion_server.py:
# a Host-header allowlist (DNS-rebinding protection) and default-deny CORS,
# both extendable via CLI flags.
_DEFAULT_ALLOWED_HOSTS = ["127.0.0.1", "localhost", "::1"]
_DEFAULT_CORS_ORIGINS: list = []


def _compute_allowed_hosts(bind_host: str, extras=None) -> list:
    seen = []
    for h in (bind_host, *_DEFAULT_ALLOWED_HOSTS, *(extras or [])):
        h = (h or "").strip()
        if h and h not in seen:
            seen.append(h)
    return seen


def _compute_cors_origins(extras=None) -> list:
    seen = []
    for o in (*_DEFAULT_CORS_ORIGINS, *(extras or [])):
        o = (o or "").strip()
        if o and o not in seen:
            seen.append(o)
    return seen


def _configure_security_middleware(application, allowed_hosts, allowed_origins):
    if application.middleware_stack is not None:
        raise RuntimeError("security middleware must be configured before the app starts serving")
    application.user_middleware.clear()
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))
    if allowed_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(allowed_origins),
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )


_configure_security_middleware(app, _DEFAULT_ALLOWED_HOSTS, _DEFAULT_CORS_ORIGINS)


class VideoRequest(BaseModel):
    model: str = ""
    prompt: str
    size: str = "1024x576"
    num_frames: int = 121          # ~5s at 24fps
    num_inference_steps: int = 0   # 0 -> pipeline/CLI default
    fps: int = 24
    image_url: str = ""            # optional source image for image-to-video
    response_format: str = "b64_json"


def load_model():
    global _pipe, _model_id
    from diffusers import LTXPipeline

    model_path = _args.model
    _model_id = Path(model_path).name
    torch_dtype = DTYPE_MAP.get(_args.dtype, torch.bfloat16)

    _hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if _hf_token:
        try:
            from huggingface_hub import login
            login(token=_hf_token, add_to_git_credential=False)
            logger.info("Logged in to HuggingFace Hub")
        except Exception as e:
            logger.warning(f"HF login failed: {e}")

    logger.info(f"Loading LTX-Video from {model_path} (dtype={_args.dtype}, offload={_args.cpu_offload})...")
    _pipe = LTXPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)

    # On a 12GB card, model CPU offload is what keeps LTX-Video inside VRAM.
    if _args.cpu_offload:
        _pipe.enable_model_cpu_offload()
        logger.info("Loaded LTX-Video with model CPU offload")
    else:
        _pipe = _pipe.to("cuda")
        logger.info("Loaded LTX-Video on CUDA")

    if _args.vae_tiling:
        try:
            _pipe.vae.enable_tiling()
            logger.info("VAE tiling enabled")
        except Exception:
            pass

    logger.info(f"Model loaded: {_model_id}")


def _get_i2v_pipe():
    """Lazy-load the image-to-video pipeline sharing the txt2vid components."""
    global _i2v_pipe
    if _i2v_pipe is not None:
        return _i2v_pipe
    try:
        from diffusers import LTXImageToVideoPipeline
        _i2v_pipe = LTXImageToVideoPipeline.from_pipe(_pipe)
        logger.info("Loaded LTX image-to-video pipeline")
    except Exception as e:
        logger.warning(f"Could not load image-to-video pipeline: {e}")
        _i2v_pipe = None
    return _i2v_pipe


def _load_image(image_ref: str):
    """Load a PIL image from a URL or local path for image-to-video."""
    from PIL import Image
    if image_ref.startswith(("http://", "https://")):
        import httpx
        r = httpx.get(image_ref, timeout=30.0, follow_redirects=True)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    return Image.open(image_ref).convert("RGB")


@app.get("/v1/models")
def list_models():
    return {"data": [{"id": _model_id, "object": "model", "owned_by": "local"}]}


@app.post("/v1/videos/generations")
def generate_video(req: VideoRequest):
    if _pipe is None:
        return {"error": "Model not loaded"}
    from diffusers.utils import export_to_video

    try:
        w, h = req.size.split("x")
        width, height = int(w), int(h)
    except Exception:
        width, height = 1024, 576
    # LTX requires dims divisible by 32 and frames of form 8*k + 1.
    width = max(256, (width // 32) * 32)
    height = max(256, (height // 32) * 32)
    num_frames = max(9, req.num_frames)
    num_frames = ((num_frames - 1) // 8) * 8 + 1
    steps = req.num_inference_steps or (_args.steps or 30)

    logger.info(f"Generating video: {req.prompt[:80]}... ({width}x{height}, {num_frames} frames, {steps} steps)")
    start = time.time()

    call_kwargs = dict(
        prompt=req.prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=steps,
    )

    if req.image_url:
        i2v = _get_i2v_pipe()
        if i2v is not None:
            try:
                call_kwargs["image"] = _load_image(req.image_url)
                result = i2v(**call_kwargs)
            except Exception as e:
                logger.warning(f"image-to-video failed ({e}); falling back to text-to-video")
                call_kwargs.pop("image", None)
                result = _pipe(**call_kwargs)
        else:
            result = _pipe(**call_kwargs)
    else:
        result = _pipe(**call_kwargs)

    frames = result.frames[0]

    # export_to_video needs a file path; write to a temp mp4, read bytes back.
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        export_to_video(frames, tmp_path, fps=req.fps or 24)
        video_bytes = Path(tmp_path).read_bytes()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    b64 = base64.b64encode(video_bytes).decode()
    elapsed = time.time() - start
    logger.info(f"Generated video in {elapsed:.1f}s ({len(video_bytes)} bytes)")
    return {"created": int(time.time()), "model": _model_id, "data": [{"b64_json": b64}]}


@app.get("/health")
def health():
    return {"status": "ok", "model": _model_id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Lightricks/LTX-Video-0.9.7-distilled",
                        help="Path or HF repo of an LTX-Video model")
    parser.add_argument("--port", type=int, default=8102)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--steps", type=int, default=0, help="Default inference steps (0=auto)")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Enable model CPU offload (needed to fit a 12GB card)")
    parser.add_argument("--vae-tiling", action="store_true", help="Enable VAE tiling")
    parser.add_argument("--allowed-host", action="append", default=[],
                        help="Additional Host header value to accept (DNS-rebinding allowlist).")
    parser.add_argument("--allowed-origin", action="append", default=[],
                        help="Additional CORS origin to allow.")
    _args = parser.parse_args()

    final_hosts = _compute_allowed_hosts(_args.host, _args.allowed_host)
    final_origins = _compute_cors_origins(_args.allowed_origin)
    _configure_security_middleware(app, final_hosts, final_origins)
    logger.info("security middleware: allowed_hosts=%s allowed_origins=%s",
                final_hosts, final_origins or "(none — default-deny)")

    uvicorn.run(app, host=_args.host, port=_args.port)
