"""Tests for the generate_video tool: wiring/registration consistency, the MCP
arg parser, and the video_gen MCP server's cloud request-build/poll/download
logic (mocked httpx — no real API calls, no key) plus its failure messages.

Mirrors the repo's existing mock style (fake httpx.AsyncClient, monkeypatched
src.settings) from tests/test_ai_image_url_safety.py."""

import base64
import json

import pytest


# ── Registration consistency (the "silently unreachable" trap) ──

def test_generate_video_is_wired_everywhere():
    from src.tool_execution import _MCP_TOOL_MAP, _MCP_ARG_PARSERS, _MCP_JSON_PRIMARY_KEYS
    from src.agent_tools import TOOL_TAGS
    from src.tool_index import BUILTIN_TOOL_DESCRIPTIONS
    from src.tool_policy import _COMMON_TOOL_NAMES
    from src.builtin_mcp import _BUILTIN_SERVERS
    from src.settings import DEFAULT_SETTINGS

    assert _MCP_TOOL_MAP.get("generate_video") == ("video_gen", "generate_video")
    assert "generate_video" in _MCP_ARG_PARSERS
    assert "generate_video" in _MCP_JSON_PRIMARY_KEYS
    assert "generate_video" in TOOL_TAGS
    assert "generate_video" in BUILTIN_TOOL_DESCRIPTIONS
    assert "generate_video" in _COMMON_TOOL_NAMES
    assert "video_gen" in _BUILTIN_SERVERS
    # Settings gate + cloud config keys present.
    for k in ("video_gen_enabled", "video_cloud_api_key", "video_cloud_model",
              "video_cloud_base_url"):
        assert k in DEFAULT_SETTINGS


def test_video_cloud_api_key_is_secret_only():
    """The key ends in _key/contains api_key, so admin_tools classifies it as a
    credential that can't be set from chat — only via Settings. Guard that the
    name keeps that property."""
    key = "video_cloud_api_key"
    is_secret = key.endswith("token") or any(t in key for t in ("api_key", "_key", "secret", "password"))
    assert is_secret


# ── Arg parsing (JSON form + bare-text form) ──

def test_parse_generate_video_json_and_text():
    from src.tool_execution import _parse_generate_video, _build_mcp_args

    parsed = _parse_generate_video('{"prompt": "a cat surfing", "duration_seconds": 4, "resolution": "480p"}')
    assert parsed == {"prompt": "a cat surfing", "duration_seconds": 4, "resolution": "480p"}

    assert _parse_generate_video("a dog running\nignored second line") == {"prompt": "a dog running"}

    # Inline-JSON with the primary key routes through _build_mcp_args untouched.
    built = _build_mcp_args("generate_video", '{"prompt": "waves", "backend": "cloud"}')
    assert built == {"prompt": "waves", "backend": "cloud"}


def test_promote_video_fields_lifts_url():
    from src.tool_execution import _promote_video_fields

    result = {
        "exit_code": 0,
        "stdout": (
            "Generated video for: a neon city flythrough\n"
            "Direct link: https://x.example/api/generated-image/abc123def456.mp4\n"
            "model: fal-ai/ltx-video\nbackend: cloud\nduration: 5s\nsize: 1024x576"
        ),
    }
    _promote_video_fields(result)
    assert result["video_url"] == "https://x.example/api/generated-image/abc123def456.mp4"
    assert result["video_prompt"] == "a neon city flythrough"
    assert result["video_model"] == "fal-ai/ltx-video"


# ── Cloud backend: request build / poll / download (mocked httpx) ──

class _Resp:
    def __init__(self, status_code=200, payload=None, content=b""):
        self.status_code = status_code
        self._payload = payload or {}
        self.content = content
        self.text = ""

    def json(self):
        return self._payload


def _install_settings(monkeypatch, **overrides):
    import src.settings as settings
    base = {
        "video_gen_enabled": True,
        "video_cloud_api_key": "",
        "video_cloud_model": "fal-ai/ltx-video",
        "video_cloud_base_url": "https://queue.fal.run",
        "video_cloud_auth_scheme": "Key",
        "video_local_url": "",
        "app_public_url": "",
    }
    base.update(overrides)
    monkeypatch.setattr(settings, "get_setting", lambda k, d=None: base.get(k, d))


async def test_cloud_unconfigured_returns_clear_error(monkeypatch):
    """No key set -> actionable 'Set video_cloud_api_key in Settings' message,
    and no HTTP is attempted."""
    import mcp_servers.video_gen_server as vs
    _install_settings(monkeypatch)  # key empty

    video, model, err = await vs._run_cloud("a clip", "1024x576", 5, "", "")
    assert video is None
    assert "video_cloud_api_key" in err


async def test_cloud_submit_poll_download_happy_path(monkeypatch):
    """Drive the full fal.ai queue flow with a fake client: POST submit ->
    GET status (IN_PROGRESS then COMPLETED) -> GET result -> download mp4.
    Asserts the request is built correctly (URL, auth header, prompt body)."""
    import httpx
    import mcp_servers.video_gen_server as vs
    _install_settings(monkeypatch, video_cloud_api_key="secret-key")

    seen = {"submit": None, "auth": None, "statuses": 0}
    fake_mp4 = b"\x00\x00\x00\x18ftypmp42FAKEDATA"

    class _Client:
        def __init__(self, *a, **k):
            self._follow = k.get("follow_redirects", False)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None):
            seen["submit"] = (url, json)
            seen["auth"] = (headers or {}).get("Authorization")
            return _Resp(202, {
                "request_id": "req-1",
                "status_url": "https://queue.fal.run/fal-ai/ltx-video/requests/req-1/status",
                "response_url": "https://queue.fal.run/fal-ai/ltx-video/requests/req-1",
            })

        async def get(self, url, headers=None):
            if url.endswith("/status"):
                seen["statuses"] += 1
                st = "IN_PROGRESS" if seen["statuses"] == 1 else "COMPLETED"
                return _Resp(200, {"status": st})
            if url.endswith("/requests/req-1"):
                return _Resp(200, {"video": {"url": "https://v3.fal.media/out.mp4"}})
            # mp4 download
            return _Resp(200, content=fake_mp4)

    # Skip the real 3s poll sleeps.
    async def _no_sleep(*_a, **_k):
        return None

    monkeypatch.setattr(vs.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    video, model, err = await vs._run_cloud("a neon city", "1024x576", 5, "", "")

    assert err == ""
    assert video == fake_mp4
    assert model == "fal-ai/ltx-video"
    # Submit hit the right model endpoint with the prompt and Key auth.
    submit_url, submit_body = seen["submit"]
    assert submit_url == "https://queue.fal.run/fal-ai/ltx-video"
    assert submit_body["prompt"] == "a neon city"
    assert seen["auth"] == "Key secret-key"
    assert seen["statuses"] >= 2  # polled through IN_PROGRESS -> COMPLETED


async def test_cloud_job_failure_reported(monkeypatch):
    import httpx
    import mcp_servers.video_gen_server as vs
    _install_settings(monkeypatch, video_cloud_api_key="k")

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None):
            return _Resp(202, {"request_id": "r", "status_url": "u/status", "response_url": "u"})

        async def get(self, url, headers=None):
            return _Resp(200, {"status": "FAILED"})

    async def _no_sleep(*_a, **_k):
        return None

    monkeypatch.setattr(vs.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    video, model, err = await vs._run_cloud("x", "1024x576", 5, "", "")
    assert video is None
    assert "failed" in err.lower()


def test_extract_video_url_shapes():
    from mcp_servers.video_gen_server import _extract_video_url
    assert _extract_video_url({"video": {"url": "a"}}) == "a"
    assert _extract_video_url({"videos": [{"url": "b"}]}) == "b"
    assert _extract_video_url({"output": {"video": {"url": "c"}}}) == "c"
    assert _extract_video_url({"url": "d"}) == "d"
    assert _extract_video_url({"nope": 1}) == ""


def test_resolution_and_duration_helpers():
    from mcp_servers.video_gen_server import _resolve_resolution, _clamp_duration
    assert _resolve_resolution("720p") == "1280x720"
    assert _resolve_resolution("") == "1024x576"        # default 576p
    assert _resolve_resolution("640x360") == "640x360"  # explicit WxH accepted
    assert _resolve_resolution("garbage") == "1024x576"
    assert _clamp_duration(100) == 6      # capped
    assert _clamp_duration(0) == 1        # floored
    assert _clamp_duration("abc") == 5    # default
