import asyncio
from unittest.mock import patch

from src.mcp_manager import _format_mcp_connection_error, McpManager


def test_playwright_mcp_connection_error_includes_install_hint():
    msg = _format_mcp_connection_error(
        "Browser (Playwright)",
        "npx",
        ["-y", "@playwright/mcp@latest", "--headless"],
        RuntimeError("package not found"),
    )

    assert "package not found" in msg
    assert "Browser MCP could not start" in msg
    assert "npx -y @playwright/mcp@latest --version" in msg
    assert "restart Odysseus" in msg


def test_generic_mcp_connection_error_preserves_original_error():
    msg = _format_mcp_connection_error(
        "Custom MCP",
        "python",
        ["server.py"],
        RuntimeError("boom"),
    )

    assert msg == "boom"


def test_http_transport_routes_to_start_http_connect():
    mgr = McpManager()

    async def fake_start(server_id, name, url, headers=None):
        return "ROUTED"

    with patch.object(McpManager, "_start_http_connect", side_effect=fake_start) as m:
        result = asyncio.run(mgr.connect_server("id1", "n", "http", url="https://x/mcp"))
    assert result == "ROUTED"
    m.assert_called_once()


def test_sse_transport_forwards_static_auth_headers():
    """A token-authed SSE server (e.g. Home Assistant) must have its headers
    threaded through connect_server -> _connect_sse so the Bearer token reaches
    the remote. Without this, sse_client() is called with no auth and 401s."""
    mgr = McpManager()
    seen = {}

    async def fake_sse(server_id, name, url, headers=None):
        seen["headers"] = headers
        return True

    hdrs = {"Authorization": "Bearer tok123"}
    with patch.object(McpManager, "_connect_sse", side_effect=fake_sse):
        asyncio.run(mgr.connect_server("id1", "HA", "sse", url="http://ha/sse", headers=hdrs))
    assert seen["headers"] == hdrs
