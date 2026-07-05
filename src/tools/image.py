"""Image-domain tool implementations.

Extracted from tool_implementations.py as part of slice 1 (#4082/#4071).
Holds the edit_image (gallery) tool.
``src.tool_implementations`` re-exports these for backward compatibility.
``_INTERNAL_BASE`` still lives in tool_implementations.py and is pulled back
function-locally here.
"""
import base64
from pathlib import Path
from typing import Dict, Optional

from src.tools._common import _parse_tool_args

# Extensions we hand to a vision model, mapped to their MIME types.
_VIEW_IMAGE_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
}
_VIEW_IMAGE_MAX_BYTES = 20 * 1024 * 1024  # 20 MB — vision inputs are pricey.


async def do_view_image(content: str, owner: Optional[str] = None) -> Dict:
    """Load an image so a vision-capable model can actually see it.

    Accepts either a gallery/generated image id (``image_id`` — e.g. the id
    returned by generate_image) or a workspace file ``path``. Returns the pixels
    under ``images``; the agent loop forwards that to the model's vision input,
    but only for models that can accept images (the loop already gates this).
    """
    try:
        args = _parse_tool_args(content)
    except ValueError:
        return {"error": "Invalid JSON arguments", "exit_code": 1}

    image_id = str(args.get("image_id") or "").strip()
    path = str(args.get("path") or "").strip()
    if not image_id and not path:
        return {"error": "Provide either image_id (a gallery/generated image) or path (a workspace file).", "exit_code": 1}

    # Resolve to a filesystem path.
    if image_id:
        from src.generated_images import resolve_generated_image_path
        fid = image_id.lstrip("#")
        if fid.startswith("image-"):
            fid = fid[len("image-"):]
        if not any(fid.lower().endswith(ext) for ext in _VIEW_IMAGE_MIME):
            fid += ".png"
        try:
            fpath = Path(resolve_generated_image_path(fid))
        except Exception:
            return {"error": f"Gallery image not found: {image_id}", "exit_code": 1}
    else:
        from src.tool_execution import _resolve_tool_path
        try:
            fpath = Path(_resolve_tool_path(path))
        except ValueError as e:
            return {"error": f"Path not allowed: {e}", "exit_code": 1}
        if not fpath.is_file():
            return {"error": f"File not found: {path}", "exit_code": 1}

    mime = _VIEW_IMAGE_MIME.get(fpath.suffix.lower())
    if not mime:
        return {"error": f"Not a supported image type: {fpath.suffix or '(none)'}. Supported: {', '.join(sorted(_VIEW_IMAGE_MIME))}.", "exit_code": 1}
    try:
        data = fpath.read_bytes()
    except Exception as e:
        return {"error": f"Could not read image: {e}", "exit_code": 1}
    if len(data) > _VIEW_IMAGE_MAX_BYTES:
        return {"error": f"Image too large ({len(data) // 1024 // 1024} MB, max 20 MB).", "exit_code": 1}

    b64 = base64.b64encode(data).decode("ascii")
    return {
        "images": [{"data": b64, "mimeType": mime}],
        "output": (
            f"Loaded image '{fpath.name}' ({mime}, {len(data) // 1024} KB) — it is now "
            f"attached to your view. If you can see images, describe or analyze it as needed; "
            f"if you cannot, say so."
        ),
        "exit_code": 0,
    }


async def do_edit_image(content: str, owner: Optional[str] = None) -> Dict:
    """Edit a gallery image (upscale, rembg, inpaint, harmonize)."""
    import httpx
    from src.tool_implementations import _INTERNAL_BASE  # shared constant, still lives in the facade
    try:
        args = _parse_tool_args(content)
    except ValueError:
        return {"error": "Invalid JSON arguments", "exit_code": 1}
    image_id = args.get("image_id", "")
    action = args.get("action", "")
    if not image_id or not action:
        return {"error": "image_id and action are required", "exit_code": 1}
    payload = {"image_id": image_id}
    if args.get("prompt"):
        payload["prompt"] = args["prompt"]
    if args.get("scale"):
        payload["scale"] = args["scale"]
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{_INTERNAL_BASE}/api/gallery/{action}", json=payload)
            data = resp.json()
        if data.get("success") or data.get("id"):
            return {"output": f"Image edited ({action}). New image ID: {data.get('id', '?')}", "exit_code": 0}
        return {"error": data.get("error", f"{action} failed"), "exit_code": 1}
    except Exception as e:
        return {"error": str(e), "exit_code": 1}
