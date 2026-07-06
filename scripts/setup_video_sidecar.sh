#!/usr/bin/env bash
# Set up the local LTX-Video sidecar for Odysseus's generate_video tool.
#
# This is the ONE command the user runs to enable local (offline, no-API-key)
# video generation. It builds the docker/video image and starts the "video"
# service defined under the `video` compose profile on host port 8102. The LTX-
# Video weights (~a few GB) download into ./data/huggingface on first request.
#
# Nothing here runs automatically — the sidecar is deferred so a normal
# `docker compose up` stays lightweight. Run this only when you want local video.
#
# Usage:
#   scripts/setup_video_sidecar.sh            # NVIDIA (default)
#   COMPOSE=docker-compose.gpu-amd.yml scripts/setup_video_sidecar.sh   # AMD ROCm
#
# After it is up, in Odysseus Settings set video_gen_enabled = true. The tool's
# backend=auto/local path health-checks http://host.docker.internal:8102 and
# uses this sidecar; no API key is needed.
set -euo pipefail

COMPOSE="${COMPOSE:-docker-compose.gpu-nvidia.yml}"
cd "$(dirname "$0")/.."

echo "Building + starting the LTX-Video sidecar via ${COMPOSE} (profile: video)..."
docker compose -f "${COMPOSE}" --profile video up -d --build video

echo
echo "Done. The sidecar is starting on host port 8102."
echo "First video request will download the LTX-Video weights into ./data/huggingface."
echo "Check health:   curl http://127.0.0.1:8102/health"
echo "Then in Odysseus Settings, set video_gen_enabled = true."
