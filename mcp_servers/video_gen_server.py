"""
video_gen_server.py

MCP server exposing short text-to-video (and optional image-to-video) clip
generation. Mirrors mcp_servers/image_gen_server.py: it is a stdio MCP server
that saves an mp4 into GENERATED_IMAGES_DIR (the same dir the gallery serves
from, so the clip shows up in the gallery and via /api/generated-image/<file>)
and records a GalleryImage row, then returns a "Direct link:" the agent loop
promotes and the chat linkifies.

Two backends (see generate_video):

  * LOCAL  — a GPU sidecar container (docker/video, host port 8102) that serves
    an OpenAI-image-style /v1/videos/generations endpoint. Preferred when it is
    healthy; needs no API key. The container spec ships but its heavy weights /
    image are NOT built by default (see scripts/setup_video_sidecar.sh).
  * CLOUD  — a provider-agnostic fal.ai-style queue backend (submit -> poll ->
    download mp4). Provider/model/endpoint are configurable in Settings; the API
    key (video_cloud_api_key) is left empty, so an unconfigured cloud call fails
    with a clear, actionable message.

backend=auto (default): try LOCAL if its health check passes, else fall back to
CLOUD if configured, else return a clear error explaining both options.
"""

import asyncio
import sys
import uuid
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import GENERATED_IMAGES_DIR

# Local sidecar defaults. Host port 8102 mirrors the diffusion sidecar's 8101/
# 8100 convention (see docker/video/Dockerfile + the gpu compose entries). The
# Odysseus app runs in Docker and reaches host services via host.docker.internal
# (see docker-compose.yml extra_hosts), so that is the default host.
_DEFAULT_LOCAL_URL = "http://host.docker.internal:8102"

# Resolution presets -> WxH. Kept small/16:9-ish for single-GPU (12GB) local
# inference; the cloud provider clamps to whatever it supports.
_RESOLUTION_PRESETS = {
    "480p": "854x480",
    "512": "512x512",
    "576p": "1024x576",
    "720p": "1280x720",
}
_DEFAULT_RESOLUTION = "576p"

_MAX_DURATION = 6  # hard cap on requested seconds (short clips only)

server = Server("video_gen")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_video",
            description=(
                "Generate a short video clip (a few seconds) from a text prompt "
                "and optional source image. Takes minutes; returns a gallery link."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Video description prompt"},
                    "duration_seconds": {
                        "type": "number",
                        "description": f"Clip length in seconds (default 5, max {_MAX_DURATION})",
                    },
                    "resolution": {
                        "type": "string",
                        "description": "Resolution preset: " + ", ".join(_RESOLUTION_PRESETS) + " (default 576p)",
                    },
                    "image_url": {
                        "type": "string",
                        "description": "Optional source image URL/path for image-to-video",
                    },
                    "backend": {
                        "type": "string",
                        "description": "auto (default), local, or cloud",
                    },
                    "model": {"type": "string", "description": "Model override (else uses Settings)"},
                },
                "required": ["prompt"],
            },
        )
    ]


def _resolve_resolution(resolution: str) -> str:
    if not resolution:
        resolution = _DEFAULT_RESOLUTION
    resolution = str(resolution).strip().lower()
    if resolution in _RESOLUTION_PRESETS:
        return _RESOLUTION_PRESETS[resolution]
    # Accept an explicit WxH too, else fall back to the default preset.
    if "x" in resolution:
        parts = resolution.split("x")
        if len(parts) == 2 and all(p.strip().isdigit() for p in parts):
            return resolution
    return _RESOLUTION_PRESETS[_DEFAULT_RESOLUTION]


def _clamp_duration(duration) -> int:
    try:
        d = int(round(float(duration)))
    except (TypeError, ValueError):
        d = 5
    return max(1, min(_MAX_DURATION, d))


def _save_video_and_gallery(video_bytes: bytes, prompt: str, model_id: str,
                            size: str, backend: str, ext: str = "mp4") -> str:
    """Write the clip into GENERATED_IMAGES_DIR (served + gallery-scanned) and
    insert a GalleryImage row, mirroring image_gen_server. Returns the relative
    /api/generated-image/<filename> path (public base prefixed by the caller)."""
    from src.settings import get_setting

    out_dir = Path(GENERATED_IMAGES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex[:12]}.{ext}"
    (out_dir / filename).write_bytes(video_bytes)

    _pub_base = (get_setting("app_public_url", "") or "").rstrip("/")
    video_url = f"{_pub_base}/api/generated-image/{filename}"

    # Gallery row. The gallery frontend renders <video controls muted playsinline>
    # for mp4/webm entries purely off the filename extension (_isVideoUrl), so no
    # gallery-side change is needed — a normal GalleryImage row is enough.
    try:
        from src.database import SessionLocal, GalleryImage
        db = SessionLocal()
        db.add(GalleryImage(
            id=str(uuid.uuid4()),
            filename=filename,
            prompt=prompt,
            model=f"{model_id} ({backend})" if model_id else backend,
            size=size,
            quality="video",
        ))
        db.commit()
        db.close()
    except Exception:
        pass

    return video_url


async def _local_health(base_url: str) -> bool:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
            resp = await client.get(base_url.rstrip("/") + "/health")
            return resp.status_code == 200
    except Exception:
        return False


async def _run_local(base_url: str, prompt: str, size: str, duration: int,
                     model_spec: str, image_url: str) -> tuple[bytes | None, str, str]:
    """Call the local sidecar's OpenAI-image-style /v1/videos/generations.
    Returns (video_bytes | None, model_id, error). Mirrors the diffusion server's
    b64_json response shape."""
    import base64
    import httpx

    payload = {
        "prompt": prompt,
        "size": size,
        "num_frames": max(9, duration * 24),
        "response_format": "b64_json",
    }
    if model_spec:
        payload["model"] = model_spec
    if image_url:
        payload["image_url"] = image_url

    videos_url = base_url.rstrip("/") + "/v1/videos/generations"
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=900.0, write=30.0, pool=30.0)
        ) as client:
            resp = await client.post(videos_url, json=payload)
    except httpx.TimeoutException:
        return None, "", "local video generation timed out (900s)"
    except Exception as e:
        return None, "", f"local sidecar unreachable: {e}"

    if resp.status_code != 200:
        return None, "", f"local sidecar error ({resp.status_code}): {resp.text[:300]}"
    data = resp.json()
    if data.get("error"):
        return None, "", f"local sidecar error: {data['error']}"
    items = data.get("data", [])
    if not items or not items[0].get("b64_json"):
        return None, "", "local sidecar returned no video"
    model_id = data.get("model") or model_spec or "local-video"
    try:
        return base64.b64decode(items[0]["b64_json"]), model_id, ""
    except Exception as e:
        return None, "", f"local sidecar returned undecodable video: {e}"


def _extract_video_url(result: dict) -> str:
    """Pull the output mp4 URL from a fal.ai-style result. Video models return
    {"video": {"url": ...}} or {"videos": [{"url": ...}]}; be liberal."""
    if not isinstance(result, dict):
        return ""
    vid = result.get("video")
    if isinstance(vid, dict) and vid.get("url"):
        return vid["url"]
    vids = result.get("videos")
    if isinstance(vids, list) and vids and isinstance(vids[0], dict) and vids[0].get("url"):
        return vids[0]["url"]
    # Some providers nest under output / put a bare url.
    out = result.get("output")
    if isinstance(out, dict):
        return _extract_video_url(out)
    if isinstance(result.get("url"), str):
        return result["url"]
    return ""


async def _run_cloud(prompt: str, size: str, duration: int, model_spec: str,
                     image_url: str) -> tuple[bytes | None, str, str]:
    """Provider-agnostic queue backend, targeting fal.ai's queue API by default:
    submit -> poll status -> fetch result -> download mp4. Returns
    (video_bytes | None, model_id, error)."""
    import httpx
    from src.settings import get_setting

    api_key = (get_setting("video_cloud_api_key", "") or "").strip()
    if not api_key:
        return None, "", (
            "Cloud video generation is not configured. Set video_cloud_api_key in "
            "Settings (and optionally video_cloud_model / video_cloud_base_url), or "
            "set up the local video sidecar (see the generate-video skill)."
        )

    model_id = (model_spec or get_setting("video_cloud_model", "") or "fal-ai/ltx-video").strip()
    base_url = (get_setting("video_cloud_base_url", "") or "https://queue.fal.run").rstrip("/")
    auth_scheme = (get_setting("video_cloud_auth_scheme", "") or "Key").strip()
    headers = {"Authorization": f"{auth_scheme} {api_key}", "Content-Type": "application/json"}

    submit_url = f"{base_url}/{model_id}"
    body: dict = {"prompt": prompt}
    if image_url:
        body["image_url"] = image_url
    # Best-effort hints; providers ignore unknown fields.
    body["duration"] = duration
    try:
        w, h = size.split("x")
        body["resolution"] = h  # fal video models take a short-side like "480"/"720"
    except Exception:
        pass

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=30.0)
    ) as client:
        # 1) submit
        try:
            resp = await client.post(submit_url, json=body, headers=headers)
        except httpx.TimeoutException:
            return None, model_id, "cloud submit timed out"
        if resp.status_code not in (200, 201, 202):
            return None, model_id, f"cloud submit failed ({resp.status_code}): {resp.text[:300]}"
        submit = resp.json()
        request_id = submit.get("request_id")
        status_url = submit.get("status_url")
        response_url = submit.get("response_url")
        if not status_url and request_id:
            status_url = f"{submit_url}/requests/{request_id}/status"
        if not response_url and request_id:
            response_url = f"{submit_url}/requests/{request_id}"
        if not status_url or not response_url:
            return None, model_id, "cloud submit returned no request handle"

        # 2) poll status until COMPLETED (cap ~15 min at 3s intervals)
        for _ in range(300):
            await asyncio.sleep(3)
            try:
                s = await client.get(status_url, headers=headers)
            except httpx.TimeoutException:
                continue
            if s.status_code != 200:
                return None, model_id, f"cloud status failed ({s.status_code}): {s.text[:200]}"
            st = s.json().get("status", "")
            if st == "COMPLETED":
                break
            if st in ("FAILED", "ERROR", "CANCELLED"):
                return None, model_id, f"cloud job {st.lower()}"
        else:
            return None, model_id, "cloud job did not complete in time (15 min)"

        # 3) fetch result + download mp4
        try:
            r = await client.get(response_url, headers=headers)
        except httpx.TimeoutException:
            return None, model_id, "cloud result fetch timed out"
        if r.status_code != 200:
            return None, model_id, f"cloud result failed ({r.status_code}): {r.text[:200]}"
        video_url = _extract_video_url(r.json())
        if not video_url:
            return None, model_id, "cloud result contained no video url"

    # Download the mp4 (separate client with a long read timeout).
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=300.0, write=30.0, pool=30.0),
            follow_redirects=True,
        ) as dl:
            dresp = await dl.get(video_url)
    except httpx.TimeoutException:
        return None, model_id, "cloud video download timed out"
    if dresp.status_code != 200:
        return None, model_id, f"cloud video download failed ({dresp.status_code})"
    return dresp.content, model_id, ""


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "generate_video":
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    prompt = (arguments.get("prompt") or "").strip()
    if not prompt:
        return [TextContent(type="text", text="Error: Video prompt is required")]

    from src.settings import get_setting

    if not get_setting("video_gen_enabled", False):
        return [TextContent(type="text", text="Error: Video generation is disabled by the administrator.")]

    duration = _clamp_duration(arguments.get("duration_seconds", 5))
    size = _resolve_resolution(arguments.get("resolution", ""))
    image_url = (arguments.get("image_url") or "").strip()
    model_spec = (arguments.get("model") or "").strip()
    backend = (arguments.get("backend") or "auto").strip().lower()
    if backend not in ("auto", "local", "cloud"):
        backend = "auto"

    local_url = (get_setting("video_local_url", "") or _DEFAULT_LOCAL_URL).rstrip("/")

    video_bytes = None
    used_backend = ""
    model_id = ""
    errors: list[str] = []

    # LOCAL first for auto/local (no API cost, private).
    if backend in ("auto", "local"):
        if await _local_health(local_url):
            video_bytes, model_id, err = await _run_local(
                local_url, prompt, size, duration, model_spec, image_url
            )
            if video_bytes:
                used_backend = "local"
            elif err:
                errors.append(f"local: {err}")
        else:
            errors.append("local: sidecar not running (start it with the video setup script)")
            if backend == "local":
                return [TextContent(type="text", text=(
                    "Error: Local video sidecar is not running. Start it (docker/video, "
                    "host port 8102) via scripts/setup_video_sidecar.sh, or use "
                    "backend='cloud' with video_cloud_api_key set in Settings."
                ))]

    # CLOUD fallback for auto, or explicit cloud.
    if video_bytes is None and backend in ("auto", "cloud"):
        video_bytes, model_id, err = await _run_cloud(
            prompt, size, duration, model_spec, image_url
        )
        if video_bytes:
            used_backend = "cloud"
        elif err:
            errors.append(f"cloud: {err}")

    if video_bytes is None:
        return [TextContent(type="text", text="Error: video generation failed — " + "; ".join(errors))]

    try:
        video_url = _save_video_and_gallery(video_bytes, prompt, model_id, size, used_backend)
    except Exception as e:
        return [TextContent(type="text", text=f"Error: could not save generated video: {e}")]

    _prompt_echo = " ".join(str(prompt).split())
    result = (
        f"Generated video for: {_prompt_echo}\n"
        f"Direct link: {video_url}\n"
        f"model: {model_id}\nbackend: {used_backend}\nduration: {duration}s\nsize: {size}"
    )
    return [TextContent(type="text", text=result)]


async def run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(run())
