# Local LTX-Video text/image-to-video server for Odysseus.
# Runs scripts/video_server.py (OpenAI-style /v1/videos/generations) on the GPU
# inside a container, so no CUDA/torch install is needed on the Windows host.
# The script itself is mounted at runtime (not copied), so edits to
# video_server.py are picked up on container restart — same convention as the
# diffusion sidecar (docker/diffusion/Dockerfile).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Diffusers stack + the web-server deps video_server.py imports.
# imageio-ffmpeg supplies the ffmpeg binary diffusers' export_to_video needs to
# encode the mp4; httpx is used to fetch a source image for image-to-video.
RUN pip install --no-cache-dir \
        "diffusers>=0.32" \
        "transformers>=4.44" \
        accelerate \
        safetensors \
        sentencepiece \
        protobuf \
        ftfy \
        Pillow \
        imageio \
        imageio-ffmpeg \
        httpx \
        fastapi \
        "uvicorn[standard]" \
        pydantic

WORKDIR /workspace
EXPOSE 8102
