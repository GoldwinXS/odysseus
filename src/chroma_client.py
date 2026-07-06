"""
chroma_client.py

Singleton ChromaDB HTTP client.
Connects to a ChromaDB instance running as a standalone service.
"""

import os
import socket
import logging

logger = logging.getLogger(__name__)

_client = None

# A short connect probe so an unreachable ChromaDB fails fast instead of
# blocking on the OS connection timeout (~30-60s, WinError 10060 on Windows),
# which otherwise stalls app startup. Tunable via CHROMADB_CONNECT_TIMEOUT.
_CONNECT_TIMEOUT = float(os.getenv("CHROMADB_CONNECT_TIMEOUT", "2.0"))


def _port_open(host: str, port: int, timeout: float = None) -> bool:
    """Return True if a TCP connection to host:port succeeds within timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout or _CONNECT_TIMEOUT):
            return True
    except OSError:
        return False


def get_chroma_client():
    """Get or create the singleton ChromaDB HTTP client.

    Raises RuntimeError with a clear install hint if the `chromadb` package
    is not installed — it's an optional dependency (RAG + memory vectors).
    """
    global _client
    if _client is not None:
        return _client

    try:
        import chromadb
    except ImportError as e:
        raise RuntimeError(
            "ChromaDB integration is not installed. Install the optional "
            "dependency with: pip install chromadb-client"
        ) from e

    host = os.getenv("CHROMADB_HOST", "localhost")
    port = int(os.getenv("CHROMADB_PORT", "8100"))

    if not _port_open(host, port):
        raise RuntimeError(
            f"ChromaDB is not reachable at {host}:{port}. Start the ChromaDB "
            f"service (e.g. `docker compose up chromadb`) or set CHROMADB_HOST / "
            f"CHROMADB_PORT to point at a running instance."
        )

    # The TCP probe above can pass while the service is a black hole: Docker's
    # port proxy accepts connections even when the engine/container behind it
    # is wedged, and the chromadb client's own heartbeat() has no request
    # timeout — observed freezing app startup at import indefinitely with the
    # process at 0 CPU. Probe the HTTP layer with a hard deadline before
    # constructing the client (v2 heartbeat, falling back to v1 for older
    # servers).
    _hb_timeout = _CONNECT_TIMEOUT + 3.0
    _hb_ok = False
    _hb_err = None
    try:
        import httpx
        for _path in ("/api/v2/heartbeat", "/api/v1/heartbeat"):
            try:
                _r = httpx.get(f"http://{host}:{port}{_path}", timeout=_hb_timeout)
                if _r.status_code == 200:
                    _hb_ok = True
                    break
            except Exception as e:
                _hb_err = e
    except ImportError:
        _hb_ok = True  # httpx missing: skip the probe rather than block RAG
    if not _hb_ok:
        raise RuntimeError(
            f"ChromaDB at {host}:{port} accepted the TCP connection but did "
            f"not answer a heartbeat within {_hb_timeout:.0f}s — the service "
            f"(or the Docker engine behind it) is unhealthy."
        ) from _hb_err

    client = chromadb.HttpClient(host=host, port=port)

    # Health check before caching — if the port is open but the service isn't
    # healthy yet (e.g. still starting), don't poison the singleton with a dead
    # client; leave _client unset so the next call retries.
    client.heartbeat()
    _client = client
    logger.info(f"ChromaDB connected: {host}:{port}")
    return _client


def reset_client():
    """Reset the singleton (e.g. after config change)."""
    global _client
    _client = None
