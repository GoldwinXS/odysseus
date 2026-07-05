# Local Stable Diffusion image server for Odysseus.
# Runs scripts/diffusion_server.py (OpenAI-compatible /v1/images/generations)
# on the GPU inside a container, so no CUDA/torch install is needed on the
# Windows host. The script itself is mounted at runtime (not copied), so edits
# to diffusion_server.py are picked up on container restart.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Diffusers stack + the web-server deps diffusion_server.py imports.
RUN pip install --no-cache-dir \
        "diffusers>=0.30" \
        "transformers>=4.44" \
        accelerate \
        safetensors \
        sentencepiece \
        protobuf \
        ftfy \
        Pillow \
        fastapi \
        "uvicorn[standard]" \
        pydantic

WORKDIR /workspace
EXPOSE 8100
