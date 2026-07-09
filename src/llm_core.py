# src/llm_core.py
import httpx
import asyncio
import time
import json
import logging
import hashlib
import threading
import re
import os
from fastapi import HTTPException
from typing import Optional, Dict, List, Tuple
from src.model_context import get_context_length, DEFAULT_CONTEXT
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class LLMConfig:
    """Configuration constants for LLM operations."""
    DEFAULT_TIMEOUT = 30
    DEFAULT_TEMPERATURE = 1.0
    DEFAULT_MAX_TOKENS = 0
    MAX_RETRIES = 3
    RETRY_DELAY = 0.5
    STREAM_TIMEOUT = 300
    # TCP+TLS connect budget for a SINGLE attempt. The old hard-coded 3.0s
    # assumed LAN/Tailscale peers ('SYN in <100ms'); it is too tight for public
    # cloud endpoints (offshore APIs take ~0.5-1.5s cold, with jitter), so a
    # brief blip on the first connect of an idle chat surfaced as a 503 on the
    # streaming path (which, unlike llm_call, does not retry the connect). A
    # genuinely dead upstream stays bounded by the dead-host cooldown. Override
    # with env LLM_CONNECT_TIMEOUT (seconds).
    CONNECT_TIMEOUT = float(os.getenv('LLM_CONNECT_TIMEOUT', '10') or '10')


def _call_timeout(read_timeout) -> httpx.Timeout:
    """Per-request timeout for non-streaming LLM calls (connect from config)."""
    return httpx.Timeout(connect=LLMConfig.CONNECT_TIMEOUT, read=float(read_timeout), write=10.0, pool=5.0)


def _stream_timeout(read_timeout) -> httpx.Timeout:
    """Per-request timeout for streaming LLM calls (connect from config)."""
    return httpx.Timeout(connect=LLMConfig.CONNECT_TIMEOUT, read=float(read_timeout), write=30.0, pool=5.0)


# Cache for LLM responses
def _get_cache_key(url: str, model: str, messages: List[Dict], 
                   temperature: float, max_tokens: int) -> str:
    """Generate cache key for LLM requests."""
    hashable_messages = []
    for msg in messages:
        sorted_items = tuple(sorted(msg.items()))
        hashable_messages.append(sorted_items)
    
    content = json.dumps({
        'url': url,
        'model': model, 
        'messages': hashable_messages,
        'temp': temperature,
        'max_tokens': max_tokens
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()

_response_cache = {}

# Dead-host cooldown: maps host (scheme://host:port) -> unix ts when cooldown expires.
# When a connect to a host fails, we mark it dead for DEAD_HOST_COOLDOWN seconds so
# subsequent calls fail instantly instead of waiting on the connect timeout. Keeps
# one unreachable upstream from jamming chat across the rest of the app.
#
# But a SINGLE transient blip (local model briefly busy, a momentary
# Tailscale hiccup) used to trip a full 60s lockout — the user saw a
# 503 and thought the model died when it was fine a second later. So:
#   - require FAIL_THRESHOLD consecutive failures before cooling
#   - shorter cooldown so recovery is quick
#   - any success resets the failure counter immediately
DEAD_HOST_COOLDOWN = 20.0
_HOST_FAIL_THRESHOLD = 2
_dead_hosts: Dict[str, float] = {}
_host_fails: Dict[str, int] = {}

# ── Provider circuit-breaker (HTTP-status based) ──
# Separate from the connect-error dead-host mechanism above: an endpoint can be
# perfectly reachable (TCP connects fine) yet return a fatal HTTP status every
# request — e.g. 401 after API credits die. That case used to keep the dead
# primary first in the fallback chain and eat a failed round-trip EVERY agent
# round (observed: 12 wasted 400s over ~30 min after Anthropic credits ran out).
# So we trip a per-endpoint breaker keyed by (host, model): a single auth-style
# status (401/402/403) cools immediately; repeated transient/other pre-content
# 4xx/5xx cool after _ENDPOINT_FAIL_THRESHOLD consecutive failures. A success
# clears the state. Cooldown length is settings-configurable
# (`provider_circuit_breaker_cooldown_seconds`, default below).
_ENDPOINT_CIRCUIT_DEFAULT_COOLDOWN = 180.0
_ENDPOINT_FAIL_THRESHOLD = 3
# Statuses that cool an endpoint on the FIRST occurrence (auth/permission/quota
# — retrying is pointless until the operator fixes credentials or billing).
_ENDPOINT_HARD_STATUSES = frozenset({401, 402, 403})
_unhealthy_endpoints: Dict[str, float] = {}   # "host|model" -> unix ts cooldown expiry
_endpoint_status_fails: Dict[str, int] = {}   # "host|model" -> consecutive pre-content HTTP failures
# Guards the two maps above. The synchronous llm_call() runs inside FastAPI's
# threadpool (sync routes such as /sessions/auto-sort) while llm_call_async()
# runs on the event loop, so these maps are mutated from multiple OS threads.
# Without the lock the get()+1+set on _host_fails is a read-modify-write that
# loses failure counts under concurrent connect errors (issue #659).
_host_health_lock = threading.Lock()
_model_activity: Dict[str, float] = {}

_HARMONY_MARKER_RE = re.compile(
    r"<\|channel\|>(analysis|commentary|final)"
    r"|<\|start\|>(?:assistant|system|user|tool)?"
    r"|<\|message\|>"
    r"|<\|end\|>"
    r"|<\|return\|>"
    r"|<\|call\|>"
)
_HARMONY_MARKERS = (
    "<|channel|>analysis",
    "<|channel|>commentary",
    "<|channel|>final",
    "<|start|>assistant",
    "<|start|>system",
    "<|start|>user",
    "<|start|>tool",
    "<|start|>",
    "<|message|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
)
_HARMONY_MAX_MARKER_LEN = max(len(marker) for marker in _HARMONY_MARKERS)

_VISIBLE_CHAT_TEMPLATE_ARTIFACT_RE = re.compile(
    r"(?:\|end\|)+\|?assistan(?:t)?\|?"
    r"|\|assistan(?:t)?\|"
    r"|<\|im_start\|>\s*assistant"
    r"|<\|im_end\|>",
    re.IGNORECASE,
)


def _strip_visible_chat_template_artifacts(text: str) -> str:
    return _VISIBLE_CHAT_TEMPLATE_ARTIFACT_RE.sub("", text or "")


def _harmony_suffix_hold_len(text: str) -> int:
    """Return how many trailing chars could be the start of a harmony marker."""
    limit = min(len(text), _HARMONY_MAX_MARKER_LEN - 1)
    for n in range(limit, 0, -1):
        suffix = text[-n:]
        if any(marker.startswith(suffix) for marker in _HARMONY_MARKERS):
            return n
    return 0


class _HarmonyStreamRouter:
    """Route OpenAI harmony analysis/final channels without leaking markers."""

    def __init__(self) -> None:
        self._buf = ""
        self._seen_harmony = False
        self._channel: Optional[str] = None
        self._in_message = False

    def feed(self, text: str) -> List[Tuple[str, bool]]:
        if not text:
            return []
        self._buf += text
        return self._drain(final=False)

    def flush(self) -> List[Tuple[str, bool]]:
        return self._drain(final=True)

    def _append_text(self, out: List[Tuple[str, bool]], text: str) -> None:
        if not text:
            return
        if not self._seen_harmony:
            out.append((text, False))
            return
        if self._in_message:
            # analysis + commentary (tool-call preambles / function-arg bodies)
            # are internal, not user-facing — route them to thinking so they
            # don't leak into the visible answer; only `final` is visible.
            out.append((text, self._channel in ("analysis", "commentary")))

    def _handle_marker(self, match: re.Match[str]) -> None:
        marker = match.group(0)
        self._seen_harmony = True
        if marker.startswith("<|channel|>"):
            self._channel = match.group(1)
            self._in_message = False
        elif marker == "<|message|>":
            self._in_message = True
        else:
            self._in_message = False
            if marker in {"<|end|>", "<|return|>", "<|call|>"}:
                self._channel = None

    def _drain(self, *, final: bool) -> List[Tuple[str, bool]]:
        out: List[Tuple[str, bool]] = []
        while True:
            match = _HARMONY_MARKER_RE.search(self._buf)
            if not match:
                break
            self._append_text(out, self._buf[:match.start()])
            self._handle_marker(match)
            self._buf = self._buf[match.end():]

        hold = 0 if final else _harmony_suffix_hold_len(self._buf)
        emit = self._buf if hold == 0 else self._buf[:-hold]
        self._buf = "" if hold == 0 else self._buf[-hold:]
        self._append_text(out, emit)
        return out


def _stream_delta_event(text: str, *, thinking: bool = False) -> str:
    payload = {"delta": text}
    if thinking:
        payload["thinking"] = True
    return f"data: {json.dumps(payload)}\n\n"

def _model_activity_key(url: str, model: str) -> str:
    return f"{(url or '').strip()}|{(model or '').strip()}"

def _same_model_identity(left: str, right: str) -> bool:
    return (left or "").strip().lower() == (right or "").strip().lower()

def note_model_activity(url: str, model: str):
    """Record that a real upstream request used this endpoint/model."""
    if not url or not model:
        return
    _model_activity[_model_activity_key(url, model)] = time.time()

def seconds_since_model_activity(url: str, model: str) -> Optional[float]:
    """Seconds since the endpoint/model was last used in this process."""
    ts = _model_activity.get(_model_activity_key(url, model))
    if not ts:
        return None
    return max(0.0, time.time() - ts)

def _host_key(url: str) -> str:
    from urllib.parse import urlsplit
    s = urlsplit(url)
    return f"{s.scheme}://{s.netloc}" if s.scheme and s.netloc else url

def _is_host_dead(url: str) -> bool:
    key = _host_key(url)
    with _host_health_lock:
        exp = _dead_hosts.get(key)
        if exp is None:
            return False
        if time.time() >= exp:
            _dead_hosts.pop(key, None)
            return False
        return True

def _mark_host_dead(url: str) -> bool:
    """Record a connect failure. Only actually cools the host after
    _HOST_FAIL_THRESHOLD consecutive failures. Returns True if the host
    is now cooled (so callers can log accurately), False if it's still
    within its allowed-failure grace."""
    key = _host_key(url)
    with _host_health_lock:
        n = _host_fails.get(key, 0) + 1
        _host_fails[key] = n
        if n >= _HOST_FAIL_THRESHOLD:
            _dead_hosts[key] = time.time() + DEAD_HOST_COOLDOWN
            return True
        return False

def _clear_host_dead(url: str) -> None:
    key = _host_key(url)
    with _host_health_lock:
        _dead_hosts.pop(key, None)
        _host_fails.pop(key, None)


def _endpoint_key(url: str, model: str) -> str:
    """Circuit-breaker key: an endpoint is a (host, model) pair — one model on a
    host can 401 (its key died) while another on the same host is fine."""
    return f"{_host_key(url)}|{model or ''}"


def _circuit_cooldown_seconds() -> float:
    """Settings-configurable breaker cooldown; falls back to the default."""
    try:
        from src.settings import get_setting
        val = get_setting("provider_circuit_breaker_cooldown_seconds",
                           _ENDPOINT_CIRCUIT_DEFAULT_COOLDOWN)
        val = float(val)
        # Bound to something sane: never below 30s (too churny to help) or
        # above 30 min (a genuinely-recovered endpoint would stay locked out).
        return max(30.0, min(val, 1800.0))
    except Exception:
        return _ENDPOINT_CIRCUIT_DEFAULT_COOLDOWN


def _is_endpoint_unhealthy(url: str, model: str) -> bool:
    key = _endpoint_key(url, model)
    with _host_health_lock:
        exp = _unhealthy_endpoints.get(key)
        if exp is None:
            return False
        if time.time() >= exp:
            _unhealthy_endpoints.pop(key, None)
            _endpoint_status_fails.pop(key, None)
            return False
        return True


def _mark_endpoint_status_failure(url: str, model: str, status: Optional[int]) -> bool:
    """Record a pre-content HTTP failure for (url, model). Auth-style statuses
    (401/402/403) trip the breaker immediately; other statuses trip it after
    _ENDPOINT_FAIL_THRESHOLD consecutive failures. Returns True when the
    endpoint is now cooled (so callers can log accurately)."""
    key = _endpoint_key(url, model)
    hard = status in _ENDPOINT_HARD_STATUSES
    with _host_health_lock:
        n = _endpoint_status_fails.get(key, 0) + 1
        _endpoint_status_fails[key] = n
        if hard or n >= _ENDPOINT_FAIL_THRESHOLD:
            _unhealthy_endpoints[key] = time.time() + _circuit_cooldown_seconds()
            return True
        return False


def _clear_endpoint_unhealthy(url: str, model: str) -> None:
    """A real success clears any breaker state for this endpoint."""
    key = _endpoint_key(url, model)
    with _host_health_lock:
        _unhealthy_endpoints.pop(key, None)
        _endpoint_status_fails.pop(key, None)


# Shared async HTTP client. Reusing one client keeps connections warm:
# repeat calls to api.anthropic.com / api.openai.com / openrouter skip the
# 100-500ms TCP+TLS handshake. Lazy init so we bind to the running event loop.
_http_client: Optional[httpx.AsyncClient] = None
_http_limits = httpx.Limits(max_connections=100, max_keepalive_connections=30, keepalive_expiry=30.0)

def _get_http_client() -> httpx.AsyncClient:
    """Return process-wide AsyncClient. Per-request timeout is passed at call time."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        from src.tls_overrides import llm_verify
        _http_client = httpx.AsyncClient(
            limits=_http_limits, http2=False, verify=llm_verify(),
        )
    return _http_client

def _get_cached_response(cache_key: str) -> Optional[str]:
    """Get cached response if it exists."""
    return _response_cache.get(cache_key)

def _set_cached_response(cache_key: str, response: str) -> None:
    """Store response in cache."""
    if len(_response_cache) > 128:
        keys_to_remove = list(_response_cache.keys())[:64]
        for key in keys_to_remove:
            # pop(), not del: another thread (sync llm_call runs in FastAPI's
            # threadpool) may have already evicted the same snapshotted key,
            # and del would raise KeyError mid-eviction (issue #659).
            _response_cache.pop(key, None)
    _response_cache[cache_key] = response

# ── Anthropic native API adapter ──

ANTHROPIC_MODELS = [
    "claude-opus-4-20250514", "claude-opus-4",
    "claude-sonnet-4-20250514", "claude-sonnet-4", "claude-sonnet-4-5-20250929", "claude-sonnet-4-5",
    "claude-haiku-4-20250514", "claude-haiku-4", "claude-haiku-3-5-20241022", "claude-haiku-3-5",
]


def _is_ollama_native_url(url: str) -> bool:
    """Return True for native Ollama API URLs, including Ollama Cloud."""
    try:
        parsed = urlparse(url or "")
    except Exception as e:
        logger.warning("Failed to parse URL for Ollama detection", exc_info=e)
        return False
    host = parsed.hostname or ""
    path = (parsed.path or "").rstrip("/")
    if _host_match(url, "ollama.com"):
        return True
    if path.startswith("/v1"):
        return False
    local_ollama_host = host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"} or parsed.port == 11434
    return local_ollama_host and (path == "" or path == "/api" or path.startswith("/api/"))


def _is_ollama_openai_compat_url(url: str) -> bool:
    """Return True for local Ollama's OpenAI-compatible /v1 surface.

    Mirrors the host detection used by ``_is_ollama_native_url`` so that the
    two helpers stay in lockstep: a localhost Ollama on a non-default port
    (custom ``OLLAMA_HOST``, reverse proxy, container port remap) is treated
    the same way here as it is on the native ``/api`` path.
    """
    try:
        parsed = urlparse(url or "")
    except Exception:
        return False
    host = parsed.hostname or ""
    path = (parsed.path or "").rstrip("/")
    local_ollama_host = host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"} or parsed.port == 11434
    return local_ollama_host and (path == "/v1" or path.startswith("/v1/"))


def _ollama_api_root(url: str) -> str:
    """Return a native Ollama API root such as https://ollama.com/api."""
    url = (url or "").strip().rstrip("/")
    parsed = urlparse(url)
    path = (parsed.path or "").rstrip("/")
    if path.endswith("/api/chat"):
        return url[: -len("/chat")]
    if path.endswith("/api/tags"):
        return url[: -len("/tags")]
    if path.endswith("/api/generate"):
        return url[: -len("/generate")]
    if path.endswith("/api"):
        return url
    if path == "":
        return url + "/api"
    if _host_match(url, "ollama.com"):
        root = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else "https://ollama.com"
        return root.rstrip("/") + "/api"
    return url


def _normalize_ollama_url(url: str) -> str:
    """Ensure a native Ollama URL points at /api/chat."""
    base = _ollama_api_root(url)
    return base.rstrip("/") + "/chat"


def _normalize_openai_chat_url(url: str) -> str:
    """Ensure an OpenAI-compatible base URL points at /chat/completions."""
    base = (url or "").strip().rstrip("/")
    if not base:
        return base
    if base.endswith("/chat/completions") or base.endswith("/completions"):
        return base
    if base.endswith("/models"):
        base = base[: -len("/models")].rstrip("/")
    return base + "/chat/completions"


def _ollama_normalize_messages(messages: List[Dict]) -> List[Dict]:
    """Adapt Odysseus' canonical OpenAI-style messages to native Ollama /api/chat.

    Two shape mismatches silently break requests:

    1. Tool calls: Odysseus carries `function.arguments` as a JSON *string*.
       Native Ollama expects a JSON *object* and rejects the string form with
       HTTP 400 ("Value looks like object, but can't find closing '}' symbol"),
       aborting every follow-up (tool-result) round. Parse the arguments back
       into an object here, on a shallow copy, leaving non-tool messages
       untouched. The opaque Gemini `extra_content` (thought_signature) is
       dropped — it is meaningless to Ollama and only matters when the
       conversation is replayed to Gemini.

    2. Images (issue #4723): Odysseus carries multimodal user content as an
       OpenAI-style list ``[{type: "text", ...}, {type: "image_url",
       image_url: {url: "data:image/...;base64,XXX"}}, ...]``. Native Ollama
       does not accept a list for ``content`` — it wants ``content`` as a
       string plus a separate ``images`` array of raw base64 strings (no
       ``data:`` prefix). Without this conversion the image blocks pass
       through untouched, the vision-capable model never sees the picture,
       and the user gets "I can't see any image" even though the request
       succeeded.
    """
    out: List[Dict] = []
    for m in messages or []:
        if not isinstance(m, dict):
            out.append(m)
            continue

        nm = dict(m)

        # 1. Tool-call argument strings -> objects.
        tcs = nm.get("tool_calls")
        if tcs:
            new_calls = []
            for tc in tcs:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                call: Dict = {"function": {"name": fn.get("name", ""), "arguments": args or {}}}
                if tc.get("id"):
                    call["id"] = tc["id"]
                new_calls.append(call)
            nm["tool_calls"] = new_calls

        # 2. Multimodal content list -> native content string + images array.
        content = nm.get("content")
        if isinstance(content, list):
            text_parts: List[str] = []
            images: List[str] = list(nm.get("images") or [])
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    t = block.get("text")
                    if t:
                        text_parts.append(str(t))
                elif btype == "image_url":
                    url = (block.get("image_url") or {}).get("url", "")
                    if not url:
                        continue
                    if url.startswith("data:"):
                        # Strip the ``data:[...];base64,`` prefix — native
                        # Ollama wants only the base64 bytes.
                        _, _, b64 = url.partition(",")
                        if b64:
                            images.append(b64)
                    else:
                        # Native Ollama images[] is base64-only; it does
                        # not fetch HTTP URLs.  Skip unsupported schemes
                        # rather than sending a non-base64 string that the
                        # model silently ignores.
                        logger.warning(
                            "Skipping non-data image_url (Ollama images[] "
                            "requires base64): %s",
                            url[:80],
                        )
            nm["content"] = "\n".join(text_parts).strip()
            if images:
                nm["images"] = images

        out.append(nm)
    return out


# Backward-compatible alias for callers/tests that imported the older name
# (it only handled tool messages originally — issue #4723 broadened scope).
_ollama_normalize_tool_messages = _ollama_normalize_messages


def _build_ollama_payload(
    model: str,
    messages: List[Dict],
    temperature: float,
    max_tokens: int,
    stream: bool = False,
    tools: Optional[List[Dict]] = None,
    num_ctx: Optional[int] = None,
) -> Dict:
    """Build the JSON payload for Ollama's /api/chat endpoint.

    ``num_ctx`` sets the input context window. Ollama defaults to 2048
    when the option is omitted, so a model with a larger advertised
    window is silently truncated there, and a model with a smaller one
    gets an oversized window it can't service. Pass the discovered
    context length through ``num_ctx``; this builder only emits it when
    the value is trusted (not the ``DEFAULT_CONTEXT`` fallback), so we
    don't guess for unknown models but do tell Ollama the real window
    when we know it — even if it's smaller than 2048.
    """
    payload: Dict = {
        "model": model,
        "messages": _ollama_normalize_messages(messages),
        "stream": stream,
    }
    options: Dict = {}
    if temperature is not None:
        options["temperature"] = temperature
    if max_tokens and max_tokens > 0:
        options["num_predict"] = max_tokens
    if num_ctx is not None and num_ctx > 0 and num_ctx != DEFAULT_CONTEXT:
        options["num_ctx"] = num_ctx
    if options:
        payload["options"] = options
    if tools:
        payload["tools"] = tools
    return payload


def _parse_ollama_response(data: dict) -> str:
    message = data.get("message") or {}
    return message.get("content") or data.get("response") or ""


def _host_match(url: str, *domains: str) -> bool:
    """Return True if url's hostname equals any of `domains` or is a subdomain of one.

    Used by helpers that want "is this Anthropic?" / "is this OpenRouter?"
    style checks. Prefer this over substring matching on the URL: the
    substring form gives wrong answers for unrelated paths or query strings
    that happen to contain the domain text.
    """
    if not url:
        return False
    try:
        # rstrip(".") so a fully-qualified host with a trailing dot
        # ("api.anthropic.com.") still matches "anthropic.com".
        host = (urlparse(url).hostname or "").lower().rstrip(".")
    except Exception:
        return False
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in domains)


# Kimi Code subscription keys (api.kimi.com/coding/v1) require a whitelisted
# coding-agent User-Agent; otherwise the API returns 403 access_terminated_error.
# Tried in order; first success is cached per base URL for later requests.
KIMI_CODE_USER_AGENTS: tuple[str, ...] = (
    "claude-code/0.1.0",
    "claude-code/1.0.0",
    "KimiCLI/1.0",
    "Kilo-Code/1.0",
    "Roo-Code/1.0",
    "Cursor/1.0",
)
KIMI_CODE_USER_AGENT = KIMI_CODE_USER_AGENTS[0]
_kimi_code_ua_cache: dict[str, str] = {}


def _is_kimi_code_url(url: str) -> bool:
    if not url or not _host_match(url, "kimi.com"):
        return False
    try:
        return "/coding" in (urlparse(url).path or "")
    except Exception:
        return False


def _kimi_code_base_key(url: str) -> str:
    """Normalize a Kimi Code chat/models URL to its OpenAI base (.../coding/v1)."""
    parsed = urlparse(url)
    path = (parsed.path or "").rstrip("/")
    for suffix in ("/chat/completions", "/models", "/completions"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
    path = path.rstrip("/") or "/coding/v1"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _is_kimi_code_access_denied(status: int, body: bytes | str) -> bool:
    if status != 403:
        return False
    text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else (body or "")
    lower = text.lower()
    return (
        "access_terminated_error" in lower
        or "coding agents" in lower
        or "only available for coding" in lower
    )


def _kimi_code_ua_candidates(url: str) -> list[str]:
    if not _is_kimi_code_url(url):
        return []
    base_key = _kimi_code_base_key(url)
    cached = _kimi_code_ua_cache.get(base_key)
    if cached:
        return [cached] + [ua for ua in KIMI_CODE_USER_AGENTS if ua != cached]
    return list(KIMI_CODE_USER_AGENTS)


def _remember_kimi_code_user_agent(url: str, user_agent: str) -> None:
    _kimi_code_ua_cache[_kimi_code_base_key(url)] = user_agent


def apply_kimi_code_headers(headers: Optional[Dict], url: str) -> Dict[str, str]:
    """Pick a Kimi Code User-Agent (cached probe when possible)."""
    h = dict(headers or {})
    if not _is_kimi_code_url(url):
        return h
    base_key = _kimi_code_base_key(url)
    cached = _kimi_code_ua_cache.get(base_key)
    if cached:
        h["User-Agent"] = cached
        return h
    models_url = base_key.rstrip("/") + "/models"
    from src.tls_overrides import llm_verify
    for ua in KIMI_CODE_USER_AGENTS:
        trial = dict(h)
        trial["User-Agent"] = ua
        try:
            r = httpx.get(models_url, headers=trial, timeout=8, verify=llm_verify())
        except Exception:
            continue
        if _is_kimi_code_access_denied(r.status_code, r.content):
            logger.debug("Kimi Code rejected User-Agent %s (403), trying next", ua)
            continue
        if r.status_code < 400:
            _remember_kimi_code_user_agent(url, ua)
            h["User-Agent"] = ua
            return h
        break
    h.setdefault("User-Agent", KIMI_CODE_USER_AGENT)
    return h


def httpx_get_kimi_aware(url: str, headers: Optional[Dict], **kwargs):
    h = apply_kimi_code_headers(headers, url)
    if not _is_kimi_code_url(url):
        return httpx.get(url, headers=h, **kwargs)
    last = None
    for ua in _kimi_code_ua_candidates(url):
        trial = dict(h)
        trial["User-Agent"] = ua
        last = httpx.get(url, headers=trial, **kwargs)
        if not _is_kimi_code_access_denied(last.status_code, last.content):
            if last.status_code < 400:
                _remember_kimi_code_user_agent(url, ua)
            return last
    return last


def httpx_post_kimi_aware(url: str, headers: Optional[Dict], **kwargs):
    h = apply_kimi_code_headers(headers, url)
    if not _is_kimi_code_url(url):
        return httpx.post(url, headers=h, **kwargs)
    last = None
    for ua in _kimi_code_ua_candidates(url):
        trial = dict(h)
        trial["User-Agent"] = ua
        last = httpx.post(url, headers=trial, **kwargs)
        if not _is_kimi_code_access_denied(last.status_code, last.content):
            if last.status_code < 400:
                _remember_kimi_code_user_agent(url, ua)
            return last
    return last


async def httpx_post_kimi_aware_async(client, url: str, headers: Optional[Dict], **kwargs):
    h = apply_kimi_code_headers(headers, url)
    if not _is_kimi_code_url(url):
        return await client.post(url, headers=h, **kwargs)
    last = None
    for ua in _kimi_code_ua_candidates(url):
        trial = dict(h)
        trial["User-Agent"] = ua
        last = await client.post(url, headers=trial, **kwargs)
        if not _is_kimi_code_access_denied(last.status_code, last.content):
            if last.status_code < 400:
                _remember_kimi_code_user_agent(url, ua)
            return last
    return last


def _detect_provider(url: str) -> str:
    """Detect the API provider from a configured endpoint URL.

    Matches on hostname (exact or subdomain) rather than substring, so a URL
    that merely contains a provider's domain in its path or query — or a
    look-alike host such as ``anthropic.com.example`` — is not misclassified.
    Unknown hosts fall back to the OpenAI-compatible default, which the
    majority of providers implement.
    """
    if _is_ollama_native_url(url):
        return "ollama"
    if _host_match(url, "anthropic.com"):
        return "anthropic"
    if _host_match(url, "opencode.ai/zen/go"):
        return "opencode-go"
    if _host_match(url, "opencode.ai/zen"):
        return "opencode-zen"
    if _host_match(url, "openrouter.ai"):
        return "openrouter"
    if _host_match(url, "groq.com"):
        return "groq"
    if _host_match(url, "nvidia.com"):
        return "nvidia"
    if _host_match(url, "moonshot.ai") or _host_match(url, "moonshot.cn"):
        return "moonshot"
    from src.chatgpt_subscription import is_chatgpt_subscription_base
    if is_chatgpt_subscription_base(url):
        return "chatgpt-subscription"
    from src.copilot import is_copilot_base
    if is_copilot_base(url):
        return "copilot"
    if _host_match(url, "cerebras.ai"):
        return "cerebras"
    if _host_match(url, "mistral.ai"):
        return "mistral"
    return "openai"


def _is_self_hosted_openai_compatible(url: str) -> bool:
    """True for custom/local OpenAI-compatible servers (llama.cpp, LM Studio,
    vLLM, text-generation-webui, etc.) as opposed to cloud APIs.

    Used to gate llama.cpp-server-specific payload extras (``session_id``,
    ``cache_prompt``) used for KV-cache slot affinity (issue #2927). Strict
    cloud providers reject unrecognized top-level fields (api.openai.com
    returns 400, Mistral returns 422 "extra_forbidden", issue #3793), and any
    unknown OpenAI-compatible host used to be treated as self-hosted, so those
    fields leaked to every strict provider added as a custom endpoint.

    A server only counts as self-hosted when it also resolves as local:
    loopback/private/tailscale host, or the endpoint explicitly configured
    with kind "local". A self-hosted server exposed via a public hostname
    loses the affinity hint unless its endpoint kind is set to "local" -
    a lost perf hint, versus a hard 4xx on every request the other way.
    """
    if _detect_provider(url) != "openai" or _host_match(url, "openai.com"):
        return False
    from src.model_context import is_local_endpoint
    return is_local_endpoint(url)


def _apply_local_cache_affinity(payload: Dict, url: str, session_id: Optional[str]) -> None:
    """Add llama.cpp-server slot-affinity hints to an outgoing payload, in place.

    As diagnosed in issue #2927, llama.cpp assigns requests to processing
    slots via LRU when no stable identifier is present ("session_id=<empty>
    server-selected (LCP/LRU)"), which means consecutive turns of the same
    chat can land on different slots and lose their cached prefix entirely.
    Sending a stable ``session_id`` (derived from the Odysseus session) lets
    the server keep routing the same conversation to the same slot, and
    ``cache_prompt: true`` asks it to retain/reuse the prefix it already has.

    Both fields are llama.cpp / LM Studio extensions to the OpenAI schema; we
    only set them for self-hosted OpenAI-compatible endpoints (never
    api.openai.com or other cloud providers, which reject unrecognized
    top-level request fields).
    """
    if not session_id:
        return
    if not _is_self_hosted_openai_compatible(url):
        return
    payload.setdefault("session_id", str(session_id))
    payload.setdefault("cache_prompt", True)


def _provider_headers(provider: str, headers: Optional[Dict] = None) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if isinstance(headers, dict):
        h.update(headers)
    if provider == "openrouter":
        h.setdefault("HTTP-Referer", "https://github.com/pewdiepie-archdaemon/odysseus")
        h.setdefault("X-OpenRouter-Title", "Odysseus")
    if provider == "copilot":
        # Ensure the Copilot-required headers are present even when the caller
        # didn't pass pre-built headers (e.g. model listing). build_headers()
        # already injects these for the live chat path; setdefault keeps any
        # request-specific values (x-initiator/vision) the caller set.
        from src.copilot import copilot_headers
        for k, v in copilot_headers(None).items():
            h.setdefault(k, v)
    return h


def _provider_label(url: str) -> str:
    """Human-friendly provider name for error messages."""
    if not url:
        return "provider"
    if _host_match(url, "anthropic.com"): return "Anthropic"
    if _host_match(url, "ollama.com"): return "Ollama Cloud"
    if _host_match(url, "x.ai"): return "xAI"
    if _host_match(url, "openai.com"): return "OpenAI"
    if _host_match(url, "openrouter.ai"): return "OpenRouter"
    if _host_match(url, "opencode.ai/zen/go"): return "OpenCode Go"
    if _host_match(url, "opencode.ai/zen"): return "OpenCode Zen"
    if _host_match(url, "groq.com"): return "Groq"
    from src.chatgpt_subscription import is_chatgpt_subscription_base
    if is_chatgpt_subscription_base(url): return "ChatGPT Subscription"
    from src.copilot import is_copilot_base
    if is_copilot_base(url): return "GitHub Copilot"
    if _host_match(url, "cerebras.ai"):
        return "cerebras"
    if _host_match(url, "mistral.ai"): return "Mistral"
    if _host_match(url, "deepseek.com"): return "DeepSeek"
    if _host_match(url, "nvidia.com"): return "NVIDIA"
    if _host_match(url, "googleapis.com"): return "Google"
    if _host_match(url, "together.xyz", "together.ai"): return "Together"
    if _host_match(url, "fireworks.ai"): return "Fireworks"
    if _host_match(url, "kimi.com"):
        try:
            if "/coding" in (urlparse(url).path or ""):
                return "Kimi Code"
        except Exception:
            pass
    if _is_ollama_native_url(url): return "Ollama"
    try:
        _parsed_local = urlparse(url)
        host = (_parsed_local.hostname or "").lower()
        port = _parsed_local.port
    except Exception:
        return "provider"
    if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
        # A port alone is not authoritative: vLLM, SGLang, llama.cpp and plain
        # OpenAI-compatible servers all routinely share 8000/8080, so naming the
        # serving tool from the port here would mislabel real setups. The tool is
        # identified by probing llama-server's native /props endpoint during
        # discovery (see ModelDiscovery._fingerprint_provider); this stays neutral.
        return "local endpoint"
    return host or "provider"


def _normalize_chatgpt_subscription_url(url: str) -> str:
    base = (url or "").strip().rstrip("/")
    if base.endswith("/responses"):
        return base
    return base + "/responses"


def _message_content_as_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                if part:
                    parts.append(str(part))
                continue
            if isinstance(part.get("text"), str):
                parts.append(part["text"])
                continue
            if isinstance(part.get("content"), str):
                parts.append(part["content"])
        return "\n".join(parts)
    return "" if content is None else str(content)


def _chatgpt_subscription_instructions(messages: List[Dict]) -> str:
    instructions = [
        _message_content_as_text(msg.get("content")).strip()
        for msg in messages or []
        if (msg.get("role") or "") == "system"
    ]
    instructions = [part for part in instructions if part]
    if instructions:
        return "\n\n".join(instructions)
    return "You are a helpful AI assistant."


def _build_chatgpt_responses_payload(
    model: str,
    messages: List[Dict],
    temperature: float,
    max_tokens: int,
    *,
    stream: bool = False,
) -> Dict:
    from src.chatgpt_subscription import build_responses_input

    conversation = [msg for msg in (messages or []) if (msg.get("role") or "") != "system"]
    payload: Dict = {
        "model": model,
        "instructions": _chatgpt_subscription_instructions(messages),
        "input": build_responses_input(conversation),
        "stream": stream,
        "store": False,
    }
    if not _restricts_temperature(model):
        payload["temperature"] = temperature
    # ChatGPT Subscription Codex API does not support max_output_tokens —
    # passing it returns HTTP 400 "Unsupported parameter: max_output_tokens".
    # Do not include it in the payload.
    return payload


def _format_chatgpt_subscription_error(status_code: int, text: str) -> str:
    if status_code in (401, 403):
        return "ChatGPT Subscription credentials expired or were rejected. Reconnect the provider."
    if status_code == 429:
        return "ChatGPT Subscription quota or rate limit was reached. Retry after the upstream limit resets."
    return _format_upstream_error(status_code, text, "https://chatgpt.com/backend-api/codex")


def _format_upstream_error(status: int, body: bytes | str, url: str) -> str:
    """Turn an upstream HTTP error into a user-readable sentence.

    Auth failures (401/403) become 'xAI rejected the API key' etc., so the UI
    stops showing raw JSON like '{"error":{"message":"User not found."}}'.
    """
    if isinstance(body, bytes):
        try:
            body = body.decode("utf-8", errors="replace")
        except Exception:
            body = str(body)
    provider = _provider_label(url)
    # Try to pull a message out of the body
    detail = ""
    try:
        j = json.loads(body) if body else {}
        if isinstance(j, dict):
            err = j.get("error") or j
            if isinstance(err, dict):
                detail = (err.get("message") or err.get("detail") or "").strip()
            elif isinstance(err, str):
                detail = err.strip()
    except Exception:
        detail = (body or "").strip()[:240]

    if status in (401, 403):
        msg = f"{provider} rejected the API key"
        if status == 403:
            msg = f"{provider} denied access (403)"
        if detail:
            msg += f" — {detail}"
        msg += ". Check Model Endpoints → {} and re-paste the key.".format(provider)
        return msg
    if status == 404:
        return f"{provider} returned 404 — check the base URL and model name." + (f" ({detail})" if detail else "")
    if status == 429:
        return f"{provider} rate-limited the request (429)." + (f" {detail}" if detail else "")
    if status >= 500:
        return f"{provider} is having an outage (HTTP {status})." + (f" {detail}" if detail else "")
    return f"{provider} returned HTTP {status}" + (f": {detail}" if detail else "")

# Models that require max_completion_tokens instead of max_tokens
_MAX_COMPLETION_TOKENS_MODELS = {"o1", "o3", "o4", "gpt-4.5", "gpt-5"}

def _uses_max_completion_tokens(model: str) -> bool:
    """Check if a model requires max_completion_tokens instead of max_tokens."""
    if not model:
        return False
    m = model.lower()
    return any(m.startswith(p) or f"/{p}" in m for p in _MAX_COMPLETION_TOKENS_MODELS)

# OpenAI reasoning models (o1, o3, o4, gpt-5 families) only accept the default
# temperature. Sending any explicit value — even 0.0 — returns HTTP 400
# ("Only the default (1) value is supported"). That otherwise breaks chat when a
# preset sets a non-default temperature, and makes endpoint probing report a
# perfectly good model as failing. For these models we omit the field and let
# the API use its required default. (gpt-4.5 is intentionally excluded — it is
# not a reasoning model and accepts temperature normally.)
_FIXED_TEMPERATURE_MODELS = ("o1", "o3", "o4", "gpt-5", "kimi-for-coding")

def _restricts_temperature(model: str) -> bool:
    """Check if a model rejects any non-default temperature."""
    if not model:
        return False
    m = model.lower()
    return any(m.startswith(p) or f"/{p}" in m for p in _FIXED_TEMPERATURE_MODELS)


# The official Moonshot API fixes temperature at 1.0 in thinking mode and 0.6
# when thinking is explicitly disabled for Kimi K2.5/K2.6. Any other explicit
# value returns HTTP 400. Odysseus does not currently send the `thinking` mode
# control, so omit temperature and let Moonshot use its default thinking mode.
# Keep the gate provider-specific: self-hosted Kimi deployments may accept
# custom sampling values, and older Moonshot models have different defaults.
def _moonshot_rejects_custom_temperature(provider: str, model: str) -> bool:
    """Check if the official Moonshot API fixes temperature for this model."""
    if provider != "moonshot" or not isinstance(model, str):
        return False
    model_id = model.lower().rsplit("/", 1)[-1]
    return bool(re.match(r"^kimi-k2\.(?:5|6)(?:$|[-_:])", model_id))


def _omit_temperature(provider: str, model: str) -> bool:
    """Check if a request should use the provider's default temperature."""
    return _restricts_temperature(model) or _moonshot_rejects_custom_temperature(
        provider, model
    )


# Anthropic removed the sampling parameters (temperature, top_p, top_k) starting
# with Claude Opus 4.7. On Opus 4.7 and later, sending `temperature` at all —
# even 0.0 — returns HTTP 400. Earlier Claude models (Opus 4.6 and below, every
# Sonnet/Haiku) still accept temperature in [0.0, 1.0], so the omission must be
# version-gated rather than applied to all `claude-*` models.
def _anthropic_rejects_temperature(model: str) -> bool:
    """Check if a native-Anthropic model rejects the temperature field (Opus 4.7+)."""
    if not isinstance(model, str) or not model:
        return False
    # `(?<![a-z])` anchors "opus" to a word boundary so a substring match like
    # `oct-opus`/`octopus-4-8` can't be read as Opus (it would otherwise strip
    # temperature). Cap the minor at 1-2 digits and forbid a trailing digit so a
    # dated id like `claude-opus-4-20250514` (Opus 4.0) parses as major-only (no
    # minor match, kept) instead of reading the date `20250514` as a giant minor
    # that would falsely test >= 4.7. Dated 4.7+ snapshots (`claude-opus-4-7-
    # 20260201`) keep their explicit minor and are still matched.
    match = re.search(r"(?<![a-z])opus[-_]?(\d+)[-_.](\d{1,2})(?!\d)", model.lower())
    if not match:
        return False
    return (int(match.group(1)), int(match.group(2))) >= (4, 7)

# Reasoning effort level sent to Mistral thinking-capable models. Mistral's
# API accepts "high", "medium", "low", "none" — see
# https://docs.mistral.ai/capabilities/reasoning/. Override via env var
# ODYSSEUS_MISTRAL_REASONING_EFFORT (e.g. set to "medium" for cheaper chat).
_MISTRAL_REASONING_EFFORT = os.getenv("ODYSSEUS_MISTRAL_REASONING_EFFORT", "high")


def _resolve_anthropic_effort(explicit=None):
    """Resolve the Anthropic reasoning-effort value.

    Precedence: an explicit per-call value (a caller may thread one later) >
    the `anthropic_effort` setting > None (omit). Returning None means "send
    no output_config" — the correct default, since some models (e.g.
    claude-fable-5) reject an explicit thinking/effort config. Lazy-imports
    settings to avoid a circular import at module load."""
    if explicit is not None:
        val = explicit
    else:
        try:
            from src.settings import get_setting
            val = get_setting("anthropic_effort", None)
        except Exception:
            val = None
    if val is None:
        return None
    val = str(val).strip()
    return val or None

# Reasoning-effort selector ("Default"/"Off"/"Low"/"Medium"/"High" in the UI) —
# a single conservative mapping from the UI's effort string to whatever knob
# (if any) each provider's API actually accepts. "default" always means "send
# nothing, let the provider/model pick" and is handled by the caller before
# this is even invoked (see apply_reasoning_effort's early return).
#
# Deliberately narrower than a full per-model matrix: when a provider/model's
# support for a param is uncertain, we omit rather than risk a 400 (project
# rule — see apply_reasoning_effort's docstring). This trades completeness for
# never breaking an existing working chat.
#
# Anthropic has no model-prefix list here — it is a deliberate no-op in
# apply_reasoning_effort (see that function's docstring); the UI selector
# reaches Anthropic through the pre-existing `effort=` kwarg / output_config
# mechanism instead (see stream_llm's Anthropic branch in this same module).
#
# Gemini tiers where the OpenAI-compat `reasoning_effort: "none"` value is
# documented as legal (thinking can be fully disabled). Pro-tier 2.5/3.x models
# require *some* thinking budget and reject "none" — so "off" is a no-op there
# rather than a guessed value that risks a 400.
_GEMINI_NONE_EFFORT_MODEL_PATTERNS = ("flash",)
_GEMINI_REASONING_MODEL_PATTERNS = ("gemini-2.5", "gemini-3")
_ZAI_THINKING_MODEL_PATTERNS = ("glm-4.5", "glm-4.6", "glm-5")
_QWEN3_MODEL_PATTERNS = ("qwen3", "qwen-3")


def _model_startswith(model: str, prefixes: tuple) -> bool:
    if not model:
        return False
    m = model.lower().rsplit("/", 1)[-1]
    return any(m.startswith(p) for p in prefixes)


def _model_contains(model: str, patterns: tuple) -> bool:
    if not model:
        return False
    m = model.lower()
    return any(p in m for p in patterns)


def apply_reasoning_effort(payload: Dict, model: str, endpoint_url: str, effort: Optional[str],
                            messages: Optional[List[Dict]] = None) -> None:
    """Apply the UI's reasoning-effort selector to an outgoing request, in place.

    `effort` is one of "off" / "low" / "medium" / "high", or None/"default" —
    "default"/None means "send nothing" and is a no-op (the caller may also
    skip calling this entirely in that case; handled here too so every call
    site doesn't need to duplicate the check).

    Provider mapping (conservative: omit over a param the provider might 400
    on — this project's stated rule for uncertain support):
      - Anthropic: this project already has an independent reasoning-effort
        mechanism (`_resolve_anthropic_effort` / `output_config={"effort":...}`,
        wired through `_build_anthropic_payload`'s `effort=` kwarg) which is
        NOT touched here — merging a second `thinking` mechanism into the same
        payload risks sending two conflicting reasoning configs to models that
        only accept one, and `_build_anthropic_payload` already documents that
        it deliberately never sends `thinking`. Anthropic is therefore a no-op
        in THIS helper by design; the UI selector reaches Anthropic via the
        `effort=` kwarg threaded separately (stream_llm's existing parameter),
        not through payload mutation here. See the call site in agent_loop.py.
      - OpenAI o-series / gpt-5 family: "reasoning_effort": low/medium/high;
        "off" -> "minimal" for gpt-5-family (the only family where that value
        is documented), omitted for o-series (no "minimal" tier there).
      - Gemini OpenAI-compat: "reasoning_effort": low/medium/high on 2.5+/3.x
        models; "off" -> "none" ONLY on flash-tier models where disabling
        thinking entirely is documented as legal, otherwise omitted.
      - Z.AI GLM (glm-4.5+/glm-5 only): "thinking": {"type": "enabled"} for
        low/medium/high (no granularity control), {"type": "disabled"} for off.
      - DeepSeek: no reasoning-effort param exists on the chat models -> always
        omitted, including "off".
      - Ollama / self-hosted OpenAI-compat, qwen3 family only: no payload
        field — "off" appends the "/no_think" soft-switch token to the last
        user message instead. Requires `messages` (the same list `payload`
        was/will be built from) to be passed so the mutation can be applied;
        a no-op if `messages` is not supplied.
      - Everything else (unrecognized host/model): no-op.
    """
    if not effort or effort == "default":
        return
    effort = effort.lower()
    if effort not in ("off", "low", "medium", "high"):
        return

    if _host_match(endpoint_url, "anthropic.com"):
        return  # handled via the separate `effort=` kwarg path — see docstring.

    if _host_match(endpoint_url, "openai.com"):
        is_gpt5 = _model_startswith(model, ("gpt-5",))
        is_o_series = _model_startswith(model, ("o1", "o3", "o4"))
        if effort == "off":
            if is_gpt5:
                payload["reasoning_effort"] = "minimal"
            # o-series has no "minimal"/off tier — omit rather than guess.
            return
        if is_gpt5 or is_o_series:
            payload["reasoning_effort"] = effort
        return

    if _host_match(endpoint_url, "generativelanguage.googleapis.com"):
        if not _model_contains(model, _GEMINI_REASONING_MODEL_PATTERNS):
            return
        if effort == "off":
            if _model_contains(model, _GEMINI_NONE_EFFORT_MODEL_PATTERNS):
                payload["reasoning_effort"] = "none"
            return  # pro-tier: disabling thinking entirely isn't legal — omit.
        payload["reasoning_effort"] = effort
        return

    if _host_match(endpoint_url, "z.ai") or _host_match(endpoint_url, "bigmodel.cn"):
        if not _model_contains(model, _ZAI_THINKING_MODEL_PATTERNS):
            return
        payload["thinking"] = {"type": "disabled" if effort == "off" else "enabled"}
        return

    if _host_match(endpoint_url, "deepseek.com"):
        return  # no effort param on any DeepSeek chat model — always omit.

    # Ollama (native or OpenAI-compat) and any other local/self-hosted
    # OpenAI-compat surface: only the qwen3 "/no_think" soft switch is
    # recognized, and only for "off". Everything else here is a no-op.
    if effort == "off" and messages and _model_contains(model, _QWEN3_MODEL_PATTERNS):
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str):
                    m["content"] = content + "\n/no_think"
                elif isinstance(content, list):
                    # Multimodal content list — append a trailing text block
                    # rather than guessing which block is "the text".
                    content.append({"type": "text", "text": "/no_think"})
                break


# Models that support structured thinking — may output </think> without opening tag
_THINKING_MODEL_PATTERNS = (
    "qwen3", "qwq", "deepseek-r1", "deepseek-reasoner", "minimax",
    "m2-reap", "gemma", "stepfun", "step-3", "step3",
    "magistral", "mistral-small", "mistral-medium",
)

def _supports_thinking(model: str) -> bool:
    """Check if model supports structured thinking output."""
    if not model:
        return False
    m = model.lower()
    return any(p in m for p in _THINKING_MODEL_PATTERNS)

def _normalize_mistral_content(content):
    """Mistral returns content as a structured array when reasoning is on:
        [{"type": "thinking", "thinking": [{"type": "text", "text": "..."}], "closed": true},
         {"type": "text", "text": "...final answer..."}]
    Convert to (text, thinking) tuple of plain strings. Pass through strings
    unchanged so non-Mistral OpenAI-compat endpoints are unaffected.
    """
    if isinstance(content, str):
        return content, ""
    if not isinstance(content, list):
        return "", ""
    text_parts = []
    thinking_parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            t = block.get("text", "")
            if t:
                text_parts.append(t)
        elif btype == "thinking":
            inner = block.get("thinking", [])
            if isinstance(inner, list):
                for tb in inner:
                    if isinstance(tb, dict) and tb.get("text"):
                        thinking_parts.append(tb["text"])
            elif isinstance(inner, str):
                thinking_parts.append(inner)
    return "".join(text_parts), "".join(thinking_parts)


def _convert_openai_content_to_anthropic(content):
    """Convert OpenAI multimodal content blocks to Anthropic format.

    Converts image_url blocks (data URI) → Anthropic image blocks.
    Passes text blocks through unchanged.
    """
    if not isinstance(content, list):
        return content
    converted = []
    for block in content:
        if not isinstance(block, dict):
            converted.append(block)
            continue
        if block.get("type") == "image_url":
            url = (block.get("image_url") or {}).get("url", "")
            # Parse data URI: data:image/<fmt>;base64,<data>
            if url.startswith("data:"):
                try:
                    header, b64_data = url.split(",", 1)
                    media_type = header.split(";")[0].replace("data:", "")
                except (ValueError, IndexError):
                    continue
                converted.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_data,
                    },
                })
            else:
                # External URL — use Anthropic's URL source
                converted.append({
                    "type": "image",
                    "source": {"type": "url", "url": url},
                })
        elif block.get("type") == "text":
            converted.append(block)
        else:
            converted.append(block)
    return converted


# Prompt-cache breakpoint TTL. Anthropic offers exactly TWO values — "5m" and "1h"
# (no 10-minute option). Holding a cache longer is FREE: you pay only to WRITE it
# (create) or READ it, never to retain it, so a longer TTL has no idle cost. The
# only price difference is the write itself — "1h" writes cost 2x the input rate,
# "5m" writes 1.25x. Where "1h" wins: human chat is bursty (you pause >5 min between
# messages, easy on mobile), so with "5m" the whole cached prefix (tools ~17k +
# system + history) expires in the gap and gets re-WRITTEN at 1.25x every turn
# instead of re-READ at 0.1x — a ~12x jump. Observed live: a short sonnet greeting
# where cache_write (145k) ≈ cache_read (197k), making writes ~86% of the bill. "1h"
# turns those re-writes into cheap reads (~4x cheaper on a slow chat). Agents that
# run a while in ONE turn are already covered by "5m" — their internal calls are
# seconds apart. Default "1h"; override per-deployment via the anthropic_cache_ttl
# setting ("5m" or "1h").
_VALID_CACHE_TTL = {"5m", "1h"}


def _resolve_cache_ctrl() -> dict:
    """Cache-control breakpoint dict, honouring the anthropic_cache_ttl setting.

    Falls back to "1h" on any missing/invalid value so a typo can't silently
    disable caching or send an unsupported ttl that would 400 the request."""
    ttl = "1h"
    try:
        from src.settings import get_setting
        val = str(get_setting("anthropic_cache_ttl", "1h") or "1h").strip().lower()
        if val in _VALID_CACHE_TTL:
            ttl = val
    except Exception:
        pass
    return {"type": "ephemeral", "ttl": ttl}


def _build_anthropic_payload(model, messages, temperature, max_tokens, stream=False, tools=None, effort=None):
    """Convert OpenAI-style messages to Anthropic format.

    `effort` (optional) threads a reasoning-effort level into the request as
    ``output_config={"effort": ...}``. Resolved from the `anthropic_effort`
    setting when not passed explicitly; None (the default) omits it entirely
    — required for models that reject an explicit config (e.g. claude-fable-5).
    Deliberately does NOT send a `thinking` param.
    """
    system_parts = []
    chat_messages = []
    cache_ctrl = _resolve_cache_ctrl()  # {"type":"ephemeral","ttl": 5m|1h}, per setting
    for m in messages:
        if m.get("role") == "system":
            system_parts.append(m.get("content") or "")
        elif m.get("role") == "tool":
            # Convert OpenAI tool result to Anthropic format
            chat_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id", ""),
                    "content": m.get("content", ""),
                }],
            })
        elif m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list):
            # Convert OpenAI assistant tool_calls to Anthropic format
            content = []
            if m.get("content"):
                content.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                args_str = fn.get("arguments") or "{}"
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, TypeError):
                    args = {}
                content.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": args,
                })
            chat_messages.append({"role": "assistant", "content": content})
        else:
            # Convert multimodal content (image_url → image) for Anthropic
            content = _convert_openai_content_to_anthropic(m["content"])
            chat_messages.append({"role": m["role"], "content": content})
    # Prompt-cache the conversation history too (third breakpoint alongside
    # system + tools; Anthropic allows 4). Caching is a prefix match, so marking
    # the last content block of the last message lets each agent round re-read
    # the ENTIRE prior conversation — including bulky tool results, which
    # dominate agent-loop input — at ~0.1x price instead of re-billing it.
    # Earlier breakpoints stay valid read points, so hits accrue every round.
    # Only for agentic calls (tools present): one-off chats rarely resend the
    # same history, and the cache-write premium wouldn't pay back.
    if tools and chat_messages:
        last_content = chat_messages[-1].get("content")
        if isinstance(last_content, str):
            # Normalize so the breakpoint has a block to attach to.
            chat_messages[-1]["content"] = [{"type": "text", "text": last_content}]
            last_content = chat_messages[-1]["content"]
        if isinstance(last_content, list) and last_content:
            last_block = last_content[-1]
            if isinstance(last_block, dict) and last_block.get("type") in (
                "text", "image", "tool_use", "tool_result", "document",
            ):
                # Copy before marking: list-content blocks are shared BY
                # REFERENCE with the caller's live `messages` (the converters
                # shallow-copy), so mutating in place would leak a stray
                # cache_control key into later rounds and non-Anthropic
                # fallback providers.
                marked = dict(last_block)
                marked["cache_control"] = cache_ctrl
                chat_messages[-1]["content"] = list(last_content[:-1]) + [marked]
    # Anthropic only accepts temperature in [0.0, 1.0] and 400s on anything above
    # 1.0. Clamp here (in the Anthropic builder only) so presets/sliders that use
    # the wider OpenAI 0.0-2.0 range — e.g. the shipped "Nietzsche" preset at 1.2
    # — don't hard-break every Claude request. OpenAI's own path is left untouched.
    if temperature is not None:
        temperature = max(0.0, min(temperature, 1.0))
    payload = {
        "model": model,
        "messages": chat_messages,
        "max_tokens": max_tokens if max_tokens and max_tokens > 0 else 4096,
    }
    # Opus 4.7+ removed the sampling parameters — sending `temperature` (even 0.0)
    # returns HTTP 400. Omit it for those models; older Claude models still take it.
    if not _anthropic_rejects_temperature(model):
        payload["temperature"] = temperature
    # Reasoning-effort: only present when explicitly configured. Omitting is the
    # safe default — some models reject an explicit thinking/effort config.
    _eff = _resolve_anthropic_effort(effort)
    if _eff is not None:
        payload["output_config"] = {"effort": _eff}
    if system_parts:
        system_text = "\n\n".join(system_parts)
        # Send `system` as a structured text block so we can attach a prompt-cache
        # breakpoint. The agent loop re-sends this same large prefix every round;
        # caching it makes Anthropic re-read it from cache (~90% cheaper, lower TTFB)
        # instead of re-billing it. Skip caching tiny one-off prompts, where the
        # cache-WRITE premium wouldn't pay back (no reuse). Presence of `tools`
        # means an agentic/multi-round call, where the prefix is always reused.
        # Split at the "## Available tools" boundary the prompt assembler
        # guarantees: everything BEFORE it (preamble + rules) is byte-stable
        # across turns, everything after varies with per-turn tool selection.
        # Caching the stable block separately means the frozen core keeps
        # hitting even on turns where the tool tail changed; when the tail is
        # unchanged too, the messages breakpoint still covers the whole prefix.
        _SPLIT = "\n\n## Available tools"
        _idx = system_text.find(_SPLIT)
        if _idx > 0 and (tools or len(system_text) > 4000):
            stable = {"type": "text", "text": system_text[:_idx],
                      "cache_control": cache_ctrl}
            volatile = {"type": "text", "text": system_text[_idx:].lstrip("\n")}
            payload["system"] = [stable, volatile]
        else:
            system_block = {"type": "text", "text": system_text}
            if tools or len(system_text) > 4000:
                system_block["cache_control"] = cache_ctrl
            payload["system"] = [system_block]
    if stream:
        payload["stream"] = True
    # Convert OpenAI-format tools to Anthropic format
    if tools:
        anthropic_tools = []
        for t in tools:
            if t.get("type") == "function":
                fn = t["function"]
                anthropic_tools.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })
        if anthropic_tools:
            # Deterministic order: tools render at position 0 of the prompt, so a
            # mere reorder (RAG selection returns sets in varying order) would
            # invalidate the entire cache even when the SET is unchanged.
            anthropic_tools.sort(key=lambda t: t["name"])
            # Cache the tool schemas too — they're stable for the whole agent run.
            # The breakpoint caches all tool defs preceding it in the request.
            anthropic_tools[-1]["cache_control"] = cache_ctrl
            payload["tools"] = anthropic_tools
    # Byte-for-byte cache-stability diagnostic. Prompt caching only hits if the
    # cached prefix is byte-identical turn-to-turn. Log a SHA of exactly what
    # carries the cache_control breakpoints — the stable system block (bytes
    # before "## Available tools") and the sorted tool defs — so two consecutive
    # turns can be diffed: same hash + cache_read=None ⇒ the 5-min TTL expired
    # (normal); hash CHANGES between same-session turns ⇒ real instability (bug).
    # Pairs with the [anthropic-cache] read/write log downstream.
    try:
        import hashlib as _hl
        _sys_stable = ""
        if isinstance(payload.get("system"), list) and payload["system"]:
            _sys_stable = payload["system"][0].get("text", "")
        _tools_for_hash = [
            {k: v for k, v in t.items() if k != "cache_control"}
            for t in payload.get("tools", [])
        ]
        _sys_h = _hl.sha256(_sys_stable.encode("utf-8")).hexdigest()[:16]
        _tools_h = _hl.sha256(
            json.dumps(_tools_for_hash, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]
        logger.info("[anthropic-cache-hash] stable_sys=%s tools=%s sys_len=%d ntools=%d",
                    _sys_h, _tools_h, len(_sys_stable), len(_tools_for_hash))
    except Exception:
        pass
    return payload

def _build_anthropic_headers(headers):
    """Convert Bearer auth to x-api-key for Anthropic."""
    h = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    if headers:
        for k, v in headers.items():
            if k.lower() == "authorization" and isinstance(v, str) and v.startswith("Bearer "):
                h["x-api-key"] = v[7:]
            else:
                h[k] = v
    return h

def _parse_anthropic_response(data: dict) -> str:
    """Extract text from an Anthropic response.

    The Messages API `content` is an array that can hold more than one text
    block (e.g. text split around a tool_use block, or citation-segmented
    text). Concatenate them all instead of returning only the first, which
    silently dropped the rest of the reply.
    """
    return "".join(
        block.get("text", "")
        for block in data.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _as_content_blocks(content) -> List[Dict]:
    """Coerce a message `content` into a list of content blocks.

    A list (multimodal: text + image parts) passes through; a non-empty string
    becomes a single text block; None/empty yields no blocks. Used when merging
    consecutive user messages so multimodal content isn't str()-ed away.
    """
    if isinstance(content, list):
        return content
    if content:
        return [{"type": "text", "text": str(content)}]
    return []


def _is_untrusted_context_content(content) -> bool:
    if isinstance(content, str):
        return (
            content.startswith("UNTRUSTED SOURCE DATA\n")
            or "<<<UNTRUSTED_SOURCE_DATA>>>" in content
        )
    if isinstance(content, list):
        return any(
            isinstance(block, dict)
            and block.get("type") == "text"
            and _is_untrusted_context_content(block.get("text") or "")
            for block in content
        )
    return False


_REFERENCE_CONTEXT_BOUNDARY = "Reference context received."


def _sanitize_llm_messages(messages: List[Dict]) -> List[Dict]:
    """Strip Odysseus-only metadata before sending messages to providers.

    Per the OpenAI chat format: user/system messages must have content; a tool
    message needs content + tool_call_id; an assistant message may carry content,
    tool_calls, or both. The old guard required content on every message, which
    dropped a valid assistant message that has only tool_calls — e.g. the
    follow-up message _append_tool_results builds for a no-prose native tool call
    (content=None, since Gemini/Ollama reject tool_calls alongside ""). Dropping
    it leaves the tool result dangling and breaks the next round.
    """
    allowed = {"role", "content", "name", "tool_call_id", "tool_calls", "function_call", "reasoning_content"}
    cleaned = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        item = {k: v for k, v in msg.items() if k in allowed and v is not None}
        role = item.get("role")
        if not role:
            continue
        if role == "assistant":
            # Re-add an explicit content=None when the message is tool-calls-only
            # (the None was stripped above) so the provider gets the spec-correct
            # `content: null`, not an omitted key.
            if "content" not in item and item.get("tool_calls"):
                item["content"] = None
            if "content" in item or item.get("tool_calls"):
                cleaned.append(item)
        elif role == "tool":
            if "content" in item and "tool_call_id" in item:
                cleaned.append(item)
        elif "content" in item:
            cleaned.append(item)

    # Repair tool-call adjacency before sending to any OpenAI-compatible
    # provider. Trimming/compaction/retries can leave `role:"tool"` messages
    # without their immediately-preceding assistant `tool_calls` parent, which
    # DeepSeek rejects with:
    # "Messages with role 'tool' must be a response to a preceding message with
    # 'tool_calls'". Also strip unanswered assistant tool_calls; some providers
    # reject those as incomplete conversations.
    repaired: List[Dict] = []
    i = 0
    while i < len(cleaned):
        msg = cleaned[i]
        role = msg.get("role")

        if role == "tool":
            # Orphan tool result. There is no valid assistant tool_calls parent
            # immediately before this batch, so it cannot be sent.
            logger.debug("Dropping orphan tool message before provider request")
            i += 1
            continue

        tool_calls = msg.get("tool_calls") if role == "assistant" else None
        if not tool_calls:
            repaired.append(msg)
            i += 1
            continue

        call_ids = [
            str(tc.get("id"))
            for tc in tool_calls
            if isinstance(tc, dict) and tc.get("id")
        ]
        expected = set(call_ids)
        answered_ids = []
        tool_batch = []
        j = i + 1
        while j < len(cleaned) and cleaned[j].get("role") == "tool":
            tid = str(cleaned[j].get("tool_call_id") or "")
            if tid in expected and tid not in answered_ids:
                answered_ids.append(tid)
                tool_batch.append(cleaned[j])
            else:
                logger.debug("Dropping unmatched/duplicate tool message before provider request")
            j += 1

        if not tool_batch:
            plain = {k: v for k, v in msg.items() if k != "tool_calls"}
            if (plain.get("content") or "").strip():
                repaired.append(plain)
            else:
                logger.debug("Dropping unanswered assistant tool_calls before provider request")
            i = j
            continue

        answered = set(answered_ids)
        pruned_calls = [
            tc for tc in tool_calls
            if isinstance(tc, dict) and str(tc.get("id")) in answered
        ]
        fixed = dict(msg)
        fixed["tool_calls"] = pruned_calls
        if "content" not in fixed:
            fixed["content"] = None
        repaired.append(fixed)
        repaired.extend(tool_batch)
        if len(pruned_calls) != len(tool_calls):
            logger.debug("Pruned unanswered assistant tool_calls before provider request")
        i = j

    # Merge consecutive user messages to satisfy strict role alternation
    # requirements after invalid tool-call fragments have been removed.
    merged: List[Dict] = []
    for item in repaired:
        if not merged:
            merged.append(item)
            continue

        last = merged[-1]
        if last.get("role") == "user" and item.get("role") == "user":
            if _is_untrusted_context_content(last.get("content")):
                merged.append({"role": "assistant", "content": _REFERENCE_CONTEXT_BOUNDARY})
                merged.append(item)
                continue
            last_copy = dict(last)
            lc = last_copy.get("content")
            ic = item.get("content")
            if isinstance(lc, list) or isinstance(ic, list):
                # Preserve multimodal content blocks (e.g. an image part) by
                # concatenating the block lists. str()-ing a list turned an
                # image message into its Python repr and dropped the image.
                merged_blocks = _as_content_blocks(lc) + _as_content_blocks(ic)
                if merged_blocks:
                    last_copy["content"] = merged_blocks
                else:
                    last_copy.pop("content", None)
            else:
                last_str = str(lc) if lc is not None else ""
                item_str = str(ic) if ic is not None else ""
                new_content = "\n\n".join(part for part in (last_str, item_str) if part)
                if new_content:
                    last_copy["content"] = new_content
                else:
                    last_copy.pop("content", None)
            merged[-1] = last_copy
        else:
            merged.append(item)

    return merged


def _normalize_anthropic_url(url: str) -> str:
    """Ensure Anthropic URL points to /v1/messages."""
    url = url.rstrip("/")
    if url.endswith("/v1/messages"):
        return url
    if url.endswith("/v1"):
        return url + "/messages"
    return url + "/v1/messages"


def _model_list_base(url: str) -> str:
    """Normalize model/chat URLs to the configured endpoint base."""
    base = (url or "").strip().rstrip("/")
    for suffix in ("/models", "/chat/completions", "/completions", "/v1/messages", "/responses"):
        if base.endswith(suffix):
            base = base[: -len(suffix)].rstrip("/")
    for suffix in ("/chat", "/tags", "/generate"):
        if base.endswith("/api" + suffix):
            base = base[: -len(suffix)].rstrip("/")
    return base


def _parse_model_cache(raw) -> List[str]:
    if not raw:
        return []
    try:
        models = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return []
    if not isinstance(models, list):
        return []
    out = []
    seen = set()
    for item in models:
        mid = str(item or "").strip()
        if not mid or mid in seen:
            continue
        out.append(mid)
        seen.add(mid)
    return out


def _configured_cached_model_ids(
    endpoint_url: str,
    *,
    owner: Optional[str] = None,
    endpoint_id: Optional[str] = None,
) -> List[str]:
    """Return cached models for a configured endpoint matching endpoint_url."""
    target = _model_list_base(endpoint_url)
    if not target:
        return []
    try:
        from src.database import SessionLocal, ModelEndpoint
    except Exception:
        return []
    db = SessionLocal()
    try:
        q = db.query(ModelEndpoint).filter(ModelEndpoint.is_enabled == True)
        if endpoint_id:
            q = q.filter(ModelEndpoint.id == endpoint_id)
        if owner:
            from src.auth_helpers import owner_filter
            q = owner_filter(q, ModelEndpoint, owner)
        rows = q.all()
        for ep in rows:
            if _model_list_base(getattr(ep, "base_url", "")) != target:
                continue
            models = _parse_model_cache(getattr(ep, "cached_models", None) or getattr(ep, "models", None))
            if not models:
                continue
            hidden = set(_parse_model_cache(getattr(ep, "hidden_models", None)))
            return [m for m in models if m not in hidden]
    except Exception:
        return []
    finally:
        try:
            db.close()
        except Exception:
            pass
    return []


def list_model_ids(
    base_chat_url: str,
    timeout: int = LLMConfig.DEFAULT_TIMEOUT,
    headers: Optional[Dict] = None,
    *,
    owner: Optional[str] = None,
    endpoint_id: Optional[str] = None,
) -> List[str]:
    """List available model IDs from an endpoint."""
    cached = _configured_cached_model_ids(base_chat_url, owner=owner, endpoint_id=endpoint_id)
    if cached:
        return cached
    provider = _detect_provider(base_chat_url)
    if provider == "anthropic":
        return list(ANTHROPIC_MODELS)
    try:
        h = {}
        if headers:
            h.update(headers)
        if provider == "ollama":
            models_url = _ollama_api_root(base_chat_url) + "/tags"
        else:
            from src.endpoint_resolver import build_models_url

            models_url = build_models_url(base_chat_url)
        r = httpx_get_kimi_aware(models_url, h, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        # Some OpenAI-compatible APIs (e.g. Together) return a bare list here.
        items = data if isinstance(data, list) else (data.get("data") or [])
        model_ids = [m.get("id") for m in items if isinstance(m, dict) and m.get("id")]
        if not model_ids and isinstance(data, dict):
            model_ids = [
                m.get("name") or m.get("model")
                for m in (data.get("models") or [])
                if m.get("name") or m.get("model")
            ]
        return model_ids
    except Exception:
        try:
            if ":11434" in base_chat_url or "ollama" in base_chat_url.lower():
                root = base_chat_url.replace("/v1/chat/completions", "").replace("/chat/completions", "").rstrip("/")
                r = httpx.get(root + "/api/tags", timeout=timeout)
                r.raise_for_status()
                return [m.get("name") or m.get("model") for m in (r.json().get("models") or []) if m.get("name") or m.get("model")]
        except Exception as e:
            logger.warning("Failed to fetch model list from configured endpoint", exc_info=e)
        return []

def normalize_model_id(
    endpoint_url: str,
    requested: str,
    timeout: int = LLMConfig.DEFAULT_TIMEOUT,
    *,
    owner: Optional[str] = None,
    endpoint_id: Optional[str] = None,
) -> Optional[str]:
    """Normalize a model ID to match available models."""
    avail = list_model_ids(endpoint_url, timeout, owner=owner, endpoint_id=endpoint_id)
    if not avail:
        return None
    if requested in avail:
        return requested
    import os as _os
    req_base = _os.path.basename(requested.rstrip("/"))
    for a in avail:
        if _os.path.basename(a.rstrip("/")) == req_base:
            return a
    return None

def llm_call(url: str, model: str, messages: List[Dict], temperature: float = LLMConfig.DEFAULT_TEMPERATURE,
             max_tokens: int = LLMConfig.DEFAULT_MAX_TOKENS, headers: Optional[Dict] = None, 
             timeout: int = LLMConfig.DEFAULT_TIMEOUT, prompt_type: Optional[str] = None) -> str:
    """Synchronous LLM call with optional prompt type enhancement."""
    h = _provider_headers(_detect_provider(url))
    # Tolerate headers that arrive as a JSON string (some sessions stored them
    # double-encoded) — otherwise h.update() throws "dictionary update sequence
    # element #0 has length 1; 2 is required".
    if isinstance(headers, str):
        try:
            headers = json.loads(headers)
        except Exception:
            headers = None
    if isinstance(headers, dict):
        h.update(headers)

    messages_copy = _sanitize_llm_messages(messages)

    # Consolidate multiple system messages into one at the start.
    sys_parts = []
    non_sys = []
    for m in messages_copy:
        if m.get("role") == "system":
            sys_parts.append(m.get('content') or '')
        else:
            non_sys.append(m)
    if sys_parts:
        messages_copy = [{"role": "system", "content": "\n\n".join(sys_parts)}] + non_sys
    else:
        messages_copy = non_sys

    provider = _detect_provider(url)
    cache_key = _get_cache_key(url, model, messages_copy, temperature, max_tokens)
    cached_response = _get_cached_response(cache_key)
    if cached_response:
        logger.debug(f"Returning cached response for key: {cache_key}")
        return cached_response

    if provider == "anthropic":
        target_url = _normalize_anthropic_url(url)
        h = _build_anthropic_headers(headers)
        payload = _build_anthropic_payload(model, messages_copy, temperature, max_tokens)
    elif provider == "ollama":
        target_url = _normalize_ollama_url(url)
        payload = _build_ollama_payload(
            model, messages_copy, temperature, max_tokens,
            stream=False, num_ctx=get_context_length(url, model),
        )
    else:
        target_url = _normalize_openai_chat_url(url)
        if provider == "copilot":
            from src.copilot import apply_request_headers
            apply_request_headers(h, messages_copy)
        payload = {
            "model": model,
            "messages": messages_copy,
            "temperature": temperature,
        }
        if _omit_temperature(provider, model):
            payload.pop("temperature", None)
        if max_tokens and max_tokens > 0:
            tok_key = "max_completion_tokens" if _uses_max_completion_tokens(model) else "max_tokens"
            payload[tok_key] = max_tokens
        if provider == "mistral" and _supports_thinking(model):
            payload["reasoning_effort"] = _MISTRAL_REASONING_EFFORT
    try:
        note_model_activity(target_url, model)
        r = httpx_post_kimi_aware(target_url, h, json=payload, timeout=timeout)
    except Exception as e:
        raise HTTPException(502, f"POST {target_url} failed: {e}")
    if not r.is_success:
        raise HTTPException(502, f"Upstream {target_url} -> {r.status_code}: {r.text}")
    data = r.json()
    try:
        if provider == "anthropic":
            response = _parse_anthropic_response(data)
        elif provider == "ollama":
            response = _parse_ollama_response(data)
        else:
            msg = data["choices"][0]["message"]
            content = msg.get("content")
            if isinstance(content, list):
                # Mistral structured content — extract thinking + text
                text_part, thinking_part = _normalize_mistral_content(content)
                if thinking_part:
                    response = thinking_part + "\n\n" + (text_part or "")
                else:
                    response = text_part or msg.get("reasoning_content") or ""
            else:
                response = content or msg.get("reasoning_content") or ""
        _set_cached_response(cache_key, response)
        return response
    except Exception:
        raise HTTPException(502, f"Unexpected schema from {target_url}: {str(data)[:400]}")


def _dedupe_candidates(candidates):
    """Filter malformed entries and drop a later repeat of an already-seen
    ``(url, model)`` route, preserving order (first occurrence wins).

    The chain is the primary target followed by the configured fallbacks, so a
    fallback that repeats the session's current model — a common misconfiguration,
    since callers prepend the live ``(url, model)`` to ``default_model_fallbacks``
    — would otherwise make the chain re-attempt the very route that just failed:
    a wasted round-trip plus a spurious ``fallback`` notice for a switch that did
    not happen. Headers are not part of the key; the first tuple (with its
    headers) is the one kept.
    """
    seen = set()
    out = []
    for c in candidates or []:
        if not c or not c[0] or not c[1]:
            continue
        key = (c[0], c[1])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def llm_call_with_fallback(candidates, messages, **kwargs) -> str:
    """Sync `llm_call` with an ordered fallback chain.

    `candidates` is a list of (url, model, headers). The first one that returns
    without an exception wins. Connection / 5xx-style failures fall through to
    the next candidate. The dead-host cooldown inside `llm_call` makes repeat
    attempts at an offline primary effectively free.
    """
    cands = _dedupe_candidates(candidates)
    if not cands:
        raise HTTPException(503, "No model endpoint configured")
    last_err = None
    for i, (url, model, headers) in enumerate(cands):
        try:
            return llm_call(url, model, messages, headers=headers, **kwargs)
        except Exception as e:
            last_err = e
            tag = "primary" if i == 0 else "candidate"
            logger.warning(f"[fallback] {tag} {model} failed ({type(e).__name__}); trying next")
            continue
    raise last_err if last_err else HTTPException(503, "All fallback candidates failed")


async def llm_call_async_with_fallback(candidates, messages, **kwargs) -> str:
    """Async variant of `llm_call_with_fallback` — same semantics."""
    cands = _dedupe_candidates(candidates)
    if not cands:
        raise HTTPException(503, "No model endpoint configured")
    last_err = None
    for i, (url, model, headers) in enumerate(cands):
        try:
            return await llm_call_async(url, model, messages, headers=headers, **kwargs)
        except Exception as e:
            last_err = e
            tag = "primary" if i == 0 else "candidate"
            logger.warning(f"[fallback] {tag} {model} failed ({type(e).__name__}); trying next")
            continue
    raise last_err if last_err else HTTPException(503, "All fallback candidates failed")


async def llm_call_async(
    url: str,
    model: str,
    messages: List[Dict],
    temperature: float = LLMConfig.DEFAULT_TEMPERATURE,
    max_tokens: int = LLMConfig.DEFAULT_MAX_TOKENS,
    headers: Optional[Dict] = None,
    timeout: int = LLMConfig.STREAM_TIMEOUT,
    max_retries: int = LLMConfig.MAX_RETRIES,
    prompt_type: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Asynchronous LLM call using httpx with connection pooling, timeout, retry logic, and performance logging."""
    provider = _detect_provider(url)
    messages_copy = _sanitize_llm_messages(messages)

    # Consolidate multiple system messages into one at the start.
    sys_parts = []
    non_sys = []
    for m in messages_copy:
        if m.get("role") == "system":
            sys_parts.append(m.get('content') or '')
        else:
            non_sys.append(m)
    if sys_parts:
        messages_copy = [{"role": "system", "content": "\n\n".join(sys_parts)}] + non_sys
    else:
        messages_copy = non_sys

    cache_key = _get_cache_key(url, model, messages_copy, temperature, max_tokens)
    cached_response = _get_cached_response(cache_key)
    if cached_response:
        logger.debug(f"Returning cached response for key: {cache_key}")
        return cached_response

    if provider == "chatgpt-subscription":
        # ChatGPT/Codex requires streamed Responses requests even for callers
        # that want a plain string (auto-title, memory extraction, etc.).
        # Reuse stream_llm's validated Codex SSE path and collect deltas.
        parts: List[str] = []
        async for chunk in stream_llm(
            url,
            model,
            messages_copy,
            temperature=temperature,
            max_tokens=max_tokens,
            headers=headers,
            timeout=timeout,
        ):
            event_is_error = False
            for line in str(chunk).splitlines():
                if line.startswith("event:"):
                    event_is_error = line[6:].strip() == "error"
                    continue
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw:
                    continue
                if raw == "[DONE]":
                    response = "".join(parts)
                    _set_cached_response(cache_key, response)
                    return response
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if event_is_error or data.get("error") or (data.get("status") and data.get("text")):
                    status = int(data.get("status") or 502)
                    text = data.get("text") or data.get("error") or "ChatGPT Subscription request failed"
                    raise HTTPException(status, text)
                delta = data.get("delta")
                if isinstance(delta, str):
                    parts.append(delta)
        response = "".join(parts)
        _set_cached_response(cache_key, response)
        return response

    if provider == "anthropic":
        target_url = _normalize_anthropic_url(url)
        h = _build_anthropic_headers(headers)
        payload = _build_anthropic_payload(model, messages_copy, temperature, max_tokens)
    elif provider == "ollama":
        target_url = _normalize_ollama_url(url)
        h = {"Content-Type": "application/json"}
        if headers:
            h.update(headers)
        payload = _build_ollama_payload(
            model, messages_copy, temperature, max_tokens,
            stream=False, num_ctx=get_context_length(url, model),
        )
    else:
        target_url = _normalize_openai_chat_url(url)
        h = _provider_headers(provider, headers)
        if provider == "copilot":
            from src.copilot import apply_request_headers
            apply_request_headers(h, messages_copy)
        payload = {
            "model": model,
            "messages": messages_copy,
            "temperature": temperature,
        }
        if _omit_temperature(provider, model):
            payload.pop("temperature", None)
        if max_tokens and max_tokens > 0:
            tok_key = "max_completion_tokens" if _uses_max_completion_tokens(model) else "max_tokens"
            payload[tok_key] = max_tokens
        # Suppress thinking for qwen3/gemma4 on Ollama /v1 — same as stream_llm.
        if _is_ollama_openai_compat_url(url) and _supports_thinking(model):
            payload["think"] = False
        if provider == "mistral" and _supports_thinking(model):
            payload["reasoning_effort"] = _MISTRAL_REASONING_EFFORT
        _apply_local_cache_affinity(payload, url, session_id)

    if _is_host_dead(target_url):
        raise HTTPException(503, f"Upstream {_host_key(target_url)} marked unreachable (cooldown active)")

    call_timeout = _call_timeout(timeout)
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        start = time.time()
        try:
            note_model_activity(target_url, model)
            client = _get_http_client()
            r = await httpx_post_kimi_aware_async(client, target_url, h, json=payload, timeout=call_timeout)
            duration = time.time() - start
            if not r.is_success:
                friendly = _format_upstream_error(r.status_code, r.text, target_url)
                logger.warning(
                    f"LLM async call to {target_url} failed in {duration:.2f}s "
                    f"(attempt {attempt}): HTTP {r.status_code} {friendly}"
                )
                if r.status_code in (429, 502, 503, 504) and attempt < max_retries:
                    await asyncio.sleep(LLMConfig.RETRY_DELAY)
                    continue
                raise HTTPException(r.status_code, friendly)
            logger.info(f"LLM async call to {target_url} succeeded in {duration:.2f}s (attempt {attempt})")
            _clear_host_dead(target_url)
            data = r.json()
            try:
                if provider == "anthropic":
                    response = _parse_anthropic_response(data)
                elif provider == "ollama":
                    response = _parse_ollama_response(data)
                else:
                    msg = data["choices"][0]["message"]
                    response = msg.get("content") or msg.get("reasoning_content") or ""
                _set_cached_response(cache_key, response)
                return response
            except Exception:
                raise HTTPException(502, f"Unexpected schema from {target_url}: {str(data)[:400]}")
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            _cooled = _mark_host_dead(target_url)
            duration = time.time() - start
            _tail = f" — host cooled for {DEAD_HOST_COOLDOWN:.0f}s" if _cooled else " — transient, will retry"
            logger.warning(f"LLM async connect to {target_url} failed after {duration:.2f}s: {e}{_tail}")
            if _cooled or attempt >= max_retries:
                raise HTTPException(503, f"Cannot reach {_host_key(target_url)}: {e}")
            await asyncio.sleep(LLMConfig.RETRY_DELAY)
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            duration = time.time() - start
            logger.warning(f"LLM async call attempt {attempt} failed after {duration:.2f}s: {e}")
            if attempt >= max_retries:
                raise HTTPException(502, f"POST {target_url} failed after {max_retries} attempts: {e}")
            await asyncio.sleep(LLMConfig.RETRY_DELAY)

async def stream_llm(url: str, model: str, messages: List[Dict], temperature: float = LLMConfig.DEFAULT_TEMPERATURE,
                     max_tokens: int = LLMConfig.DEFAULT_MAX_TOKENS, headers: Optional[Dict] = None,
                     timeout: int = LLMConfig.STREAM_TIMEOUT, prompt_type: Optional[str] = None,
                     tools: Optional[List[Dict]] = None, session_id: Optional[str] = None,
                     tool_choice_none: bool = False, effort: Optional[str] = None,
                     reasoning_effort: Optional[str] = None):
    """Stream LLM responses with improved error handling.

    Yields SSE chunks:
      - data: {"delta": "text"}           — text content
      - data: {"type": "tool_calls", ...}  — accumulated native tool calls (before DONE)
      - event: error                       — errors
      - data: [DONE]                       — end of stream

    `effort` and `reasoning_effort` are two independent knobs, deliberately not
    merged:
      - `effort` is the pre-existing Anthropic-only `output_config={"effort":
        ...}` value (see `_resolve_anthropic_effort`), sourced from the
        `anthropic_effort` setting or an explicit per-call override.
      - `reasoning_effort` is the newer UI reasoning-effort selector
        ("off"/"low"/"medium"/"high"/"default"), applied via
        `apply_reasoning_effort` for every non-Anthropic provider. For
        Anthropic it is translated into the SAME `effort` kwarg above
        (conservatively — see `apply_reasoning_effort`'s docstring for why a
        second, conflicting Anthropic mechanism is not introduced) rather than
        stacked alongside it; an explicit `effort=` always wins if both are
        somehow passed, since that is the pre-existing, narrower-scoped knob.
    """
    provider = _detect_provider(url)
    messages_copy = _sanitize_llm_messages(messages)

    # Consolidate multiple system messages into one at the start.
    # Some models (e.g. Qwen3.5) reject system messages that aren't first.
    sys_parts = []
    non_sys = []
    for m in messages_copy:
        if m.get("role") == "system":
            sys_parts.append(m.get('content') or '')
        else:
            non_sys.append(m)
    if sys_parts:
        messages_copy = [{"role": "system", "content": "\n\n".join(sys_parts)}] + non_sys
    else:
        messages_copy = non_sys

    # qwen3 "/no_think" soft switch (reasoning_effort="off") mutates the
    # message list itself, not a payload field — apply once here, before any
    # provider branch builds its payload from messages_copy, so it's never
    # duplicated between the native-Ollama and OpenAI-compat branches below.
    # Independent of `effort` (the separate, Anthropic-only knob) — a qwen3
    # model reached through a local Ollama endpoint is never also an
    # Anthropic call, so there is no real scenario where both are meaningful
    # at once, but the check below intentionally does not couple the two.
    if reasoning_effort:
        apply_reasoning_effort({}, model, url, reasoning_effort, messages=messages_copy)

    if provider == "anthropic":
        target_url = _normalize_anthropic_url(url)
        h = _build_anthropic_headers(headers)
        _anthropic_effort = effort
        if _anthropic_effort is None and reasoning_effort and reasoning_effort not in ("default", "off"):
            # Reuse the existing output_config={"effort":...} mechanism rather
            # than introducing a second, conflicting Anthropic reasoning path —
            # see apply_reasoning_effort's docstring. "off"/"default" both mean
            # "don't pass anything" here since there is no established
            # "explicitly disabled" value for this existing mechanism.
            _anthropic_effort = reasoning_effort
        payload = _build_anthropic_payload(model, messages_copy, temperature, max_tokens, stream=True, tools=tools, effort=_anthropic_effort)
    elif provider == "ollama":
        target_url = _normalize_ollama_url(url)
        h = {"Content-Type": "application/json"}
        if headers:
            h.update(headers)
        payload = _build_ollama_payload(
            model, messages_copy, temperature, max_tokens,
            stream=True, tools=tools, num_ctx=get_context_length(url, model),
        )
        # UI reasoning-effort selector — native Ollama /api/chat accepts a
        # top-level "think" param (bool; gpt-oss additionally accepts
        # "low"/"medium"/"high" strings). This branch previously never applied
        # the selector at all, so it was a silent no-op on every local model
        # (the user-visible "effort isn't working" bug). Only touched for
        # thinking-capable models so non-thinking models never see an
        # unexpected param.
        if reasoning_effort in ("off", "low", "medium", "high") and _supports_thinking(model):
            if "gpt-oss" in (model or "").lower() and reasoning_effort != "off":
                payload["think"] = reasoning_effort
            else:
                payload["think"] = reasoning_effort != "off"
    elif provider == "chatgpt-subscription":
        target_url = _normalize_chatgpt_subscription_url(url)
        h = _provider_headers(provider, headers)
        payload = _build_chatgpt_responses_payload(model, messages_copy, temperature, max_tokens, stream=True)
    else:
        target_url = _normalize_openai_chat_url(url)
        payload = {
            "model": model,
            "messages": messages_copy,
            "temperature": temperature,
            "stream": True,
        }
        if _omit_temperature(provider, model):
            payload.pop("temperature", None)
        if provider not in {"openrouter", "groq"}:
            payload["stream_options"] = {"include_usage": True}
        if max_tokens and max_tokens > 0:
            tok_key = "max_completion_tokens" if _uses_max_completion_tokens(model) else "max_tokens"
            payload[tok_key] = max_tokens
        if tools:
            payload["tools"] = tools
        elif tool_choice_none:
            payload["tool_choice"] = "none"
        # Mistral thinking-capable models — send reasoning_effort so Mistral
        # activates thinking mode and returns structured reasoning_content.
        # Effort level is configurable via ODYSSEUS_MISTRAL_REASONING_EFFORT
        # (high / medium / low / none); default "high".
        if provider == "mistral" and _supports_thinking(model):
            payload["reasoning_effort"] = _MISTRAL_REASONING_EFFORT
        # For Ollama's OpenAI-compat /v1 endpoint with thinking models (qwen3,
        # gemma4, etc.), suppress thinking BY DEFAULT so tool calls aren't
        # swallowed inside <think> blocks — but an EXPLICIT low/medium/high on
        # the UI effort selector overrides the suppression (the user asked for
        # thinking; the fence parser strips <think> blocks before tool parsing,
        # so the swallow risk is accepted as their call). Ollama /v1 accepts
        # "think" as a top-level param.
        if _is_ollama_openai_compat_url(url) and _supports_thinking(model):
            payload["think"] = reasoning_effort in ("low", "medium", "high")
        # UI reasoning-effort selector — OpenAI/Gemini/Z.AI get a payload field;
        # DeepSeek/unrecognized hosts are always a no-op (see docstring). The
        # qwen3 "/no_think" message mutation for this same knob already ran
        # above, before messages_copy was baked into this payload.
        if reasoning_effort:
            apply_reasoning_effort(payload, model, url, reasoning_effort)
        _apply_local_cache_affinity(payload, url, session_id)
        h = _provider_headers(provider, headers)
        if provider == "copilot":
            from src.copilot import apply_request_headers
            apply_request_headers(h, messages_copy)

    # Connect budget from LLMConfig.CONNECT_TIMEOUT (env LLM_CONNECT_TIMEOUT).
    # The dead-host cooldown still bounds a genuinely unreachable upstream, so a
    # wider connect budget only affects first contact and stops a brief cold
    # connect blip (offshore/public endpoints) surfacing as a 503 on this stream
    # path, which -- unlike llm_call -- does not retry the connect.
    stream_timeout = _stream_timeout(timeout)

    if _is_host_dead(target_url):
        yield f'event: error\ndata: {json.dumps({"error": f"Upstream {_host_key(target_url)} unreachable (cooldown active)", "status": 503})}\n\n'
        return
    note_model_activity(target_url, model)

    # ── ChatGPT Subscription / Codex Responses streaming ──
    if provider == "chatgpt-subscription":
        event_name = ""
        input_tokens = 0
        output_tokens = 0
        try:
            client = _get_http_client()
            async with client.stream('POST', target_url, json=payload, headers=h, timeout=stream_timeout) as r:
                _clear_host_dead(target_url)
                if r.status_code != 200:
                    raw = (await r.aread()).decode(errors="replace")
                    friendly = _format_chatgpt_subscription_error(r.status_code, raw)
                    yield f'event: error\ndata: {json.dumps({"status": r.status_code, "text": friendly, "raw": raw[:500]})}\n\n'
                    return
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    evt = data.get("type") or event_name
                    if evt == "response.output_text.delta":
                        delta = data.get("delta") or ""
                        if delta:
                            yield f'data: {json.dumps({"delta": delta})}\n\n'
                    elif evt == "response.completed":
                        usage = (data.get("response") or {}).get("usage") or data.get("usage") or {}
                        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or input_tokens
                        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or output_tokens
                        if input_tokens or output_tokens:
                            yield f'data: {json.dumps({"type": "usage", "data": {"input_tokens": input_tokens, "output_tokens": output_tokens}})}\n\n'
                        yield "data: [DONE]\n\n"
                        return
                    elif evt in ("response.failed", "error"):
                        err = data.get("error") or (data.get("response") or {}).get("error") or {}
                        text = err.get("message") if isinstance(err, dict) else str(err or "ChatGPT Subscription request failed")
                        yield f'event: error\ndata: {json.dumps({"status": 502, "text": text})}\n\n'
                        return
                yield "data: [DONE]\n\n"
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            _cooled = _mark_host_dead(target_url)
            _tail = f" — host cooled for {DEAD_HOST_COOLDOWN:.0f}s" if _cooled else " — transient, will retry"
            logger.warning(f"ChatGPT Subscription stream connect to {target_url} failed: {e}{_tail}")
            yield f'event: error\ndata: {json.dumps({"error": f"Cannot reach {_host_key(target_url)}", "status": 503})}\n\n'
        except httpx.ReadTimeout:
            yield f'event: error\ndata: {json.dumps({"error": "Read timeout", "status": 504})}\n\n'
        except httpx.NetworkError:
            yield f'event: error\ndata: {json.dumps({"error": "Network error", "status": 502})}\n\n'
        except Exception as e:
            logger.error(f"ChatGPT Subscription stream error: {e}")
            yield f'event: error\ndata: {json.dumps({"error": str(e), "status": 502})}\n\n'
        return

    # ── Native Ollama streaming ──
    if provider == "ollama":
        _ollama_tool_calls: List[Dict] = []
        _harmony_router = _HarmonyStreamRouter()
        try:
            client = _get_http_client()
            async with client.stream('POST', target_url, json=payload, headers=h, timeout=stream_timeout) as r:
                _clear_host_dead(target_url)
                if r.status_code != 200:
                    raw = (await r.aread()).decode(errors="replace")
                    yield _build_upstream_error_chunk(r.status_code, raw, target_url, r.headers)
                    return
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        j = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    message = j.get("message") or {}
                    thinking = message.get("thinking") or ""
                    if thinking:
                        yield _stream_delta_event(thinking, thinking=True)
                    content = message.get("content") or ""
                    if content:
                        for part, is_thinking in _harmony_router.feed(content):
                            yield _stream_delta_event(part, thinking=is_thinking)
                    for tc in message.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            _ollama_tool_calls.append({
                                "id": tc.get("id") or f"call_{len(_ollama_tool_calls)}",
                                "name": fn.get("name") or "",
                                "arguments": json.dumps(fn.get("arguments") or {}),
                            })
                    if j.get("done"):
                        for part, is_thinking in _harmony_router.flush():
                            yield _stream_delta_event(part, thinking=is_thinking)
                        if _ollama_tool_calls:
                            yield f'data: {json.dumps({"type": "tool_calls", "calls": _ollama_tool_calls})}\n\n'
                        if j.get("prompt_eval_count") is not None or j.get("eval_count") is not None:
                            yield f'data: {json.dumps({"type": "usage", "data": {"input_tokens": j.get("prompt_eval_count", 0), "output_tokens": j.get("eval_count", 0)}})}\n\n'
                        yield "data: [DONE]\n\n"
                        return
                for part, is_thinking in _harmony_router.flush():
                    yield _stream_delta_event(part, thinking=is_thinking)
                yield "data: [DONE]\n\n"
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            _cooled = _mark_host_dead(target_url)
            _tail = f" — host cooled for {DEAD_HOST_COOLDOWN:.0f}s" if _cooled else " — transient, will retry"
            logger.warning(f"Ollama stream connect to {target_url} failed: {e}{_tail}")
            yield f'event: error\ndata: {json.dumps({"error": f"Cannot reach {_host_key(target_url)}", "status": 503})}\n\n'
        except httpx.ReadTimeout:
            yield f'event: error\ndata: {json.dumps({"error": "Read timeout", "status": 504})}\n\n'
        except httpx.NetworkError:
            yield f'event: error\ndata: {json.dumps({"error": "Network error", "status": 502})}\n\n'
        except Exception as e:
            logger.error(f"Ollama stream error: {e}")
            yield f'event: error\ndata: {json.dumps({"error": str(e), "status": 502})}\n\n'
        return

    # ── Anthropic streaming ──
    if provider == "anthropic":
        _anth_input_tokens = 0
        _anth_output_tokens = 0
        _anth_cache_read = 0
        _anth_cache_write = 0
        _anth_actual_model = ""
        # Track tool_use blocks: {index: {id, name, arguments_json}}
        _anth_tool_blocks: Dict[int, Dict] = {}
        _anth_block_idx = -1
        _anth_block_type = ""
        try:
            client = _get_http_client()
            async with client.stream('POST', target_url, json=payload, headers=h, timeout=stream_timeout) as r:
                _clear_host_dead(target_url)
                if r.status_code != 200:
                    raw = (await r.aread()).decode(errors="replace")
                    yield _build_upstream_error_chunk(r.status_code, raw, target_url, r.headers)
                    return
                async for line in r.aiter_lines():
                    # SSE allows "data:value" with no space after the colon
                    # (the space is optional per the spec). Some gateways and
                    # local servers omit it; gating on "data: " dropped their
                    # entire stream.
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or not data.startswith("{"):
                        continue
                    try:
                        j = json.loads(data)
                        evt = j.get("type", "")
                        if evt == "content_block_start":
                            _anth_block_idx = j.get("index", _anth_block_idx + 1)
                            cb = j.get("content_block") or {}
                            _anth_block_type = cb.get("type", "text")
                            if _anth_block_type == "tool_use":
                                _anth_tool_blocks[_anth_block_idx] = {
                                    "id": cb.get("id") or f"call_{_anth_block_idx}",
                                    "name": cb.get("name") or "",
                                    "arguments": "",
                                }
                        elif evt == "content_block_delta":
                            delta = j.get("delta") or {}
                            delta_type = delta.get("type", "")
                            if delta_type == "text_delta":
                                text = delta.get("text") or ""
                                if text:
                                    yield f'data: {json.dumps({"delta": text})}\n\n'
                            elif delta_type == "input_json_delta":
                                # Accumulate tool arguments JSON
                                idx = j.get("index", _anth_block_idx)
                                if idx in _anth_tool_blocks:
                                    partial = delta.get("partial_json") or ""
                                    _anth_tool_blocks[idx]["arguments"] += partial
                                    # Stream tool arg deltas for doc tools
                                    if partial and _anth_tool_blocks[idx].get("name") in ("create_document", "update_document", "edit_document"):
                                        yield f'data: {json.dumps({"type": "tool_call_delta", "index": idx, "name": _anth_tool_blocks[idx]["name"], "arg_delta": partial})}\n\n'
                        elif evt == "message_start":
                            _msg = j.get("message", {}) or {}
                            _u = _msg.get("usage", {})
                            _anth_input_tokens = _u.get("input_tokens", 0)
                            # The model that actually served this response. On a
                            # fallback/alias the served model can differ from the
                            # requested `model`; persist it downstream so metadata
                            # records what really ran, not just what was asked.
                            _served = _msg.get("model")
                            if isinstance(_served, str) and _served.strip():
                                _anth_actual_model = _served.strip()
                            # Surface prompt-cache effectiveness: cache_read > 0 means the
                            # stable system+tools prefix was served from cache this round.
                            # Retain the counts (not just log them) so the usage event can
                            # forward them — a cache_read token is billed ~0.1x, so cost is
                            # unauditable without them.
                            _anth_cache_read = _u.get("cache_read_input_tokens", 0) or 0
                            _anth_cache_write = _u.get("cache_creation_input_tokens", 0) or 0
                            if _anth_cache_read or _anth_cache_write:
                                logger.info(
                                    "[anthropic-cache] read=%s write=%s fresh_input=%s",
                                    _anth_cache_read, _anth_cache_write, _anth_input_tokens,
                                )
                        elif evt == "message_delta":
                            _anth_output_tokens = j.get("usage", {}).get("output_tokens", 0)
                        elif evt == "message_stop":
                            # Emit accumulated tool calls in OpenAI-compatible format
                            if _anth_tool_blocks:
                                calls = []
                                for idx in sorted(_anth_tool_blocks):
                                    tb = _anth_tool_blocks[idx]
                                    calls.append({
                                        "id": tb["id"],
                                        "name": tb["name"],
                                        "arguments": tb["arguments"],
                                    })
                                yield f'data: {json.dumps({"type": "tool_calls", "calls": calls})}\n\n'
                            if _anth_input_tokens or _anth_output_tokens or _anth_cache_read or _anth_cache_write:
                                _usage_data = {
                                    "input_tokens": _anth_input_tokens,
                                    "output_tokens": _anth_output_tokens,
                                }
                                # Only attach cache fields when non-zero so non-cached
                                # responses stay lean and downstream code can treat a
                                # missing key as zero.
                                if _anth_cache_read:
                                    _usage_data["cache_read_tokens"] = _anth_cache_read
                                if _anth_cache_write:
                                    _usage_data["cache_creation_tokens"] = _anth_cache_write
                                # Report the served model (may differ from requested on
                                # an alias/fallback) so metadata records what ran.
                                if _anth_actual_model:
                                    _usage_data["model"] = _anth_actual_model
                                    if not _same_model_identity(_anth_actual_model, model):
                                        _usage_data["requested_model"] = model
                                yield f'data: {json.dumps({"type": "usage", "data": _usage_data})}\n\n'
                            yield "data: [DONE]\n\n"
                            return
                        elif evt == "error":
                            err_msg = j.get("error", {}).get("message", "Unknown error")
                            yield f'event: error\ndata: {json.dumps({"error": err_msg, "status": 400})}\n\n'
                            return
                    except json.JSONDecodeError:
                        continue
                yield "data: [DONE]\n\n"
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            _cooled = _mark_host_dead(target_url)
            _tail = f" — host cooled for {DEAD_HOST_COOLDOWN:.0f}s" if _cooled else " — transient, will retry"
            logger.warning(f"Anthropic stream connect to {target_url} failed: {e}{_tail}")
            yield f'event: error\ndata: {json.dumps({"error": f"Cannot reach {_host_key(target_url)}", "status": 503})}\n\n'
        except httpx.ReadTimeout:
            yield f'event: error\ndata: {json.dumps({"error": "Read timeout", "status": 504})}\n\n'
        except httpx.NetworkError:
            yield f'event: error\ndata: {json.dumps({"error": "Network error", "status": 502})}\n\n'
        except Exception as e:
            logger.error(f"Anthropic stream error: {e}")
            yield f'event: error\ndata: {json.dumps({"error": str(e), "status": 502})}\n\n'
        return

    # ── OpenAI-compatible streaming ──
    # Accumulate native tool_calls across streaming chunks
    _tc_acc: Dict[int, Dict] = {}  # index -> {id, name, arguments}
    _tc_last_idx = [-1]  # most-recently-touched slot, for providers that omit `index`
    # For thinking models: prepend <think> to first content delta so frontend
    # can detect thinking-in-progress (some models output </think> but no <think>)
    _thinking_model = _supports_thinking(model)
    _first_content_sent = False
    _in_think_tag = False        # True while consuming <think>…</think> content
    _think_open_stripped = False  # opening <think> tag already removed
    # Hold-back buffer for the plain-<think> auto-detect below. A provider can
    # split "<think>" across multiple SSE deltas (e.g. first chunk is just
    # "<th"), and stripped.lower().startswith("<think") fails on that partial
    # chunk alone — which permanently sets _first_content_sent=True and
    # disables auto-detect for the rest of the round (observed live: a
    # continuation round's <think> streamed as plain text). Buffer content
    # until it's unambiguously not (or unambiguously is) a "<think" prefix,
    # mirroring the harmony router's existing suffix-hold approach.
    _think_detect_buf = ""
    _harmony_router = _HarmonyStreamRouter()
    _harmony_active = False       # sticky: gpt-oss harmony <|channel|> stream detected
    _actual_model = ""
    _actual_model_announced = False

    def _emit_tool_calls():
        """Build the tool_calls event string if any were accumulated."""
        if not _tc_acc:
            return None
        calls = [_tc_acc[i] for i in sorted(_tc_acc)]
        return f'data: {json.dumps({"type": "tool_calls", "calls": calls})}\n\n'

    def _format_routed_content(parts: List[Tuple[str, bool]]) -> List[str]:
        nonlocal _first_content_sent
        events = []
        for part, is_thinking in parts:
            if is_thinking:
                events.append(_stream_delta_event(part, thinking=True))
                continue
            # Some thinking backends start normal content with a stray closing
            # tag. Repair only that shape; do not wrap every first token for
            # model families like MiniMax, which often stream ordinary answers.
            if _thinking_model and not _first_content_sent and part.lstrip().lower().startswith("</think"):
                part = "<think>" + part
            _first_content_sent = True
            events.append(_stream_delta_event(part))
        return events

    h = apply_kimi_code_headers(h, target_url)
    try:
        client = _get_http_client()
        async with client.stream('POST', target_url, json=payload, headers=h, timeout=stream_timeout) as r:
            _clear_host_dead(target_url)
            if r.status_code != 200:
                raw = (await r.aread()).decode(errors="replace")
                yield _build_upstream_error_chunk(r.status_code, raw, target_url, r.headers)
                return

            async for line in r.aiter_lines():
                if not line:
                    continue

                # SSE allows "data:value" with no space after the colon; gating
                # on "data: " silently dropped content + usage from providers
                # that omit it.
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data == "[DONE]":
                        for event in _format_routed_content(_harmony_router.flush()):
                            yield event
                        tc_event = _emit_tool_calls()
                        if tc_event:
                            yield tc_event
                        yield "data: [DONE]\n\n"
                        return

                    try:
                        if data.strip():
                            if data.startswith("{"):
                                j = json.loads(data)
                                chunk_model = j.get("model")
                                if isinstance(chunk_model, str) and chunk_model.strip():
                                    _actual_model = chunk_model.strip()
                                    if (
                                        not _actual_model_announced
                                        and not _same_model_identity(_actual_model, model)
                                    ):
                                        _actual_model_announced = True
                                        yield f'data: {json.dumps({"type": "model_actual", "requested_model": model, "model": _actual_model})}\n\n'
                                # Usage chunk (from stream_options)
                                _choices = j.get("choices") or []
                                _delta0 = _choices[0].get("delta") if (_choices and _choices[0] is not None) else None
                                # Capture usage whenever the chunk carries it and
                                # the delta has no actual output. Some gateways /
                                # local servers attach usage to the FINAL delta,
                                # which also carries role/finish_reason (so it is
                                # not exactly None/{}/{"content": None}); gating on
                                # those exact shapes discarded their token counts.
                                _delta_has_output = isinstance(_delta0, dict) and (
                                    _delta0.get("content")
                                    or _delta0.get("reasoning_content")
                                    or _delta0.get("reasoning")
                                    or _delta0.get("thinking")
                                    or _delta0.get("tool_calls")
                                )
                                if "usage" in j and not _delta_has_output:
                                    u = j["usage"] or {}
                                    _usage_data = {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)}
                                    # llama.cpp puts a `timings` block alongside `usage` with the
                                    # TRUE generation speed (predicted_per_second) — pure decode,
                                    # excluding prefill/network. Pass it through so the UI shows the
                                    # real gen t/s instead of recomputing tokens/wall-clock (which
                                    # includes prefill and reads ~20-40% low). Prefill speed too.
                                    _tm = j.get("timings")
                                    if isinstance(_tm, dict):
                                        if _tm.get("predicted_per_second"):
                                            _usage_data["gen_tps"] = round(_tm["predicted_per_second"], 2)
                                        if _tm.get("prompt_per_second"):
                                            _usage_data["prefill_tps"] = round(_tm["prompt_per_second"], 2)
                                    if _actual_model:
                                        _usage_data["model"] = _actual_model
                                        if not _same_model_identity(_actual_model, model):
                                            _usage_data["requested_model"] = model
                                    yield f'data: {json.dumps({"type": "usage", "data": _usage_data})}\n\n'
                                elif "choices" in j:
                                    _c0 = (j["choices"] or [None])[0]
                                    if _c0 is None:
                                        continue
                                    delta = _c0.get("delta") or {}
                                    if isinstance(delta, dict):
                                        # Text content
                                        # Reasoning tokens (VLLM --reasoning-parser, e.g. Qwen3/DeepSeek-R1, Nemotron). vLLM 0.20.2 / NIM emit the field as `reasoning`; older builds use `reasoning_content`. Some OpenAI-compatible Ollama builds use `thinking`.
                                        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or delta.get("thinking") or ""
                                        content = delta.get("content") or ""
                                        # Mistral structured content: content is a list of typed blocks
                                        # ({"type": "thinking", ...}, {"type": "text", ...}). Split into
                                        # reasoning + text so thinking streams into the thinking panel.
                                        if isinstance(content, list):
                                            text_part, thinking_part = _normalize_mistral_content(content)
                                            if thinking_part:
                                                reasoning = (reasoning + thinking_part) if reasoning else thinking_part
                                            content = text_part
                                        if reasoning:
                                            yield _stream_delta_event(reasoning, thinking=True)
                                        if content:
                                            content = _strip_visible_chat_template_artifacts(content)
                                            if not content:
                                                continue
                                            content = re.sub(r"<mm:think(\s+[^>]*)?>", r"<think\1>", content, flags=re.IGNORECASE)
                                            content = re.sub(r"</mm:think>", "</think>", content, flags=re.IGNORECASE)
                                            stripped = content.lstrip()
                                            # gpt-oss harmony format (<|channel|>analysis/final): route via the harmony
                                            # stream router. Sticky once the first marker appears — distinct from the
                                            # <think> path below (handled in the else, preserving #2588 behaviour).
                                            if _harmony_active or "<|" in content:
                                                _harmony_active = True
                                                for event in _format_routed_content(_harmony_router.feed(content)):
                                                    yield event
                                            else:
                                                # Auto-detect <think>…</think> in content stream.
                                                # Covers Qwen3-derived models (Qwopus, QwQ forks) whose
                                                # names don't match _THINKING_MODEL_PATTERNS but still
                                                # emit literal <think> markup via llama.cpp --jinja.
                                                #
                                                # The prefix check below needs the WHOLE "<think" in one
                                                # chunk. A provider that splits it across deltas (first
                                                # chunk just "<th") fails the check, falls to the "else"
                                                # below, and permanently sets _first_content_sent=True —
                                                # disabling auto-detect for the rest of the round. So while
                                                # still undecided, hold back short content that could still
                                                # be a "<think" prefix and resolve once we have enough of it
                                                # (or it's clearly not a match), mirroring the harmony
                                                # router's suffix-hold buffering above.
                                                if not _first_content_sent and not _thinking_model and not _in_think_tag:
                                                    _think_detect_buf += content
                                                    _buf_stripped = _think_detect_buf.lstrip()
                                                    _probe = _buf_stripped[:len("<think")].lower()
                                                    if len(_buf_stripped) < len("<think") and "<think"[:len(_probe)] == _probe:
                                                        # Too short to decide yet AND still a plausible
                                                        # prefix — keep holding, emit nothing this chunk.
                                                        continue
                                                    # Resolved: either long enough to check for real, or it
                                                    # already diverged from "<think" — replay the full held
                                                    # buffer as `content`/`stripped` through the normal path.
                                                    content = _think_detect_buf
                                                    stripped = _buf_stripped
                                                    _think_detect_buf = ""
                                                if not _first_content_sent and not _thinking_model and not _in_think_tag and stripped.lower().startswith("<think"):
                                                    _thinking_model = True
                                                    _in_think_tag = True
                                                if _in_think_tag:
                                                    close_idx = content.lower().find("</think>")
                                                    if close_idx != -1:
                                                        # Split: up-to-</think> → thinking, remainder → content
                                                        think_part = content[:close_idx]
                                                        if not _think_open_stripped:
                                                            # Strip the opening <think[...] > from the first chunk.
                                                            # Use a dedicated flag — _first_content_sent stays False
                                                            # throughout the think block, so it must not be reused.
                                                            tag_end = think_part.lower().find(">")
                                                            if tag_end != -1:
                                                                think_part = think_part[tag_end + 1:]
                                                            _think_open_stripped = True
                                                        regular_part = content[close_idx + len("</think>"):]
                                                        _in_think_tag = False
                                                        if think_part:
                                                            yield f'data: {json.dumps({"delta": think_part, "thinking": True})}\n\n'
                                                        if regular_part:
                                                            _first_content_sent = True
                                                            yield f'data: {json.dumps({"delta": regular_part})}\n\n'
                                                    else:
                                                        # Still inside <think>: route to thinking channel
                                                        if not _think_open_stripped:
                                                            # Strip the opening <think[...] > tag (first chunk only)
                                                            tag_end = stripped.lower().find(">")
                                                            if tag_end != -1:
                                                                content = stripped[tag_end + 1:]
                                                            _think_open_stripped = True
                                                        if content:
                                                            yield f'data: {json.dumps({"delta": content, "thinking": True})}\n\n'
                                                else:
                                                    # Some thinking backends start normal content with a
                                                    # stray closing tag. Repair only that shape; do not
                                                    # wrap every first token for model families like
                                                    # MiniMax, which often stream ordinary answers.
                                                    if _thinking_model and not _first_content_sent and stripped.lower().startswith("</think"):
                                                        content = "<think>" + content
                                                    _first_content_sent = True
                                                    yield f'data: {json.dumps({"delta": content})}\n\n'
                                        # Native tool calls — accumulate across chunks
                                        for tc in delta.get("tool_calls") or []:
                                            if tc is None:
                                                continue
                                            func = tc.get("function") or {}
                                            raw_idx = tc.get("index")
                                            if raw_idx is None:
                                                # Gemini's OpenAI-compat layer omits `index` on
                                                # parallel tool calls (every delta arrives as
                                                # index=None) and sends each call complete in one
                                                # delta. Without this, all parallel calls collide
                                                # into slot 0 — later calls overwrite the first's
                                                # name and CORRUPT its arguments by concatenation,
                                                # so only one malformed call survives and the
                                                # follow-up round 400s. A function name marks the
                                                # start of a new call → allocate a fresh slot;
                                                # an arg-only continuation attaches to the last.
                                                if func.get("name") or _tc_last_idx[0] < 0:
                                                    # Next free slot ABOVE any existing key (not
                                                    # len()), so a provider mixing integer indices
                                                    # with index=None can never collide.
                                                    idx = max(_tc_acc, default=-1) + 1
                                                else:
                                                    idx = _tc_last_idx[0]
                                            else:
                                                idx = raw_idx
                                            _tc_last_idx[0] = idx
                                            if idx not in _tc_acc:
                                                _tc_acc[idx] = {"id": "", "name": "", "arguments": ""}
                                            if tc.get("id"):
                                                _tc_acc[idx]["id"] = tc["id"]
                                            # Gemini 3 returns an opaque thought_signature in
                                            # extra_content on the function-call delta. It MUST be
                                            # echoed back on the assistant tool_call next round or the
                                            # follow-up request 400s ("Function call is missing a
                                            # thought_signature"). Preserve it verbatim; other
                                            # providers never send it, so this is a no-op for them.
                                            if tc.get("extra_content"):
                                                _tc_acc[idx]["extra_content"] = tc["extra_content"]
                                            if func.get("name"):
                                                _tc_acc[idx]["name"] = func["name"]
                                            if "arguments" in func:
                                                # Guard against a null arguments delta: `func` can be
                                                # {"arguments": None} (JSON null), and a raw `+= None`
                                                # raises TypeError that the broad except swallows,
                                                # silently dropping the rest of the chunk. Matches the
                                                # Anthropic accumulator (`partial = ... or ""`) above.
                                                _tc_acc[idx]["arguments"] += func["arguments"] or ""
                                                # Stream tool arg deltas for doc tools
                                                if func["arguments"] and _tc_acc[idx].get("name") in ("create_document", "update_document", "edit_document"):
                                                    yield f'data: {json.dumps({"type": "tool_call_delta", "index": idx, "name": _tc_acc[idx]["name"], "arg_delta": func["arguments"]})}\n\n'
                                elif "text" in j:
                                    if j["text"]:
                                        for event in _format_routed_content(_harmony_router.feed(j["text"])):
                                            yield event
                            else:
                                if data.strip():
                                    for event in _format_routed_content(_harmony_router.feed(data)):
                                        yield event
                    except Exception as e:
                        logger.error(f"Error parsing stream data: {e}")
                        continue

            # End of stream (no explicit [DONE] received)
            for event in _format_routed_content(_harmony_router.flush()):
                yield event
            tc_event = _emit_tool_calls()
            if tc_event:
                yield tc_event
            yield "data: [DONE]\n\n"

    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        _cooled = _mark_host_dead(target_url)
        _tail = f" — host cooled for {DEAD_HOST_COOLDOWN:.0f}s" if _cooled else " — transient, will retry"
        logger.warning(f"Stream connect to {target_url} failed: {e}{_tail}")
        yield f'event: error\ndata: {json.dumps({"error": f"Cannot reach {_host_key(target_url)}", "status": 503})}\n\n'
    except httpx.ReadTimeout:
        yield f'event: error\ndata: {json.dumps({"error": "Read timeout", "status": 504})}\n\n'
    except httpx.NetworkError:
        yield f'event: error\ndata: {json.dumps({"error": "Network error", "status": 502})}\n\n'
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f'event: error\ndata: {json.dumps({"error": str(e), "status": 502})}\n\n'


def _parse_retry_after(headers) -> Optional[float]:
    """Parse a Retry-After header (seconds form only; HTTP-date form is rare on
    LLM APIs and not worth the parse). Returns seconds capped at 15s, or None."""
    if not headers:
        return None
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:
        raw = None
    if not raw:
        return None
    try:
        secs = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if secs < 0:
        return None
    return min(secs, 15.0)


def _build_upstream_error_chunk(status: int, raw: str, target_url: str, resp_headers=None) -> str:
    """Build the `event: error` SSE chunk for a non-200 upstream response.

    Centralizes three things the per-provider branches all need:
      - a friendly message (_format_upstream_error),
      - the raw upstream body (truncated) — logged at WARNING for 4xx so the
        ACTUAL cause survives (the real "credit balance too low" text was
        previously lost; only the generic friendly string was kept),
      - a parsed Retry-After (seconds) so the fallback layer can honor it.
    """
    friendly = _format_upstream_error(status, raw, target_url)
    body = (raw or "")[:500]
    if 400 <= status < 500:
        # 4xx bodies carry the real reason (bad key, exhausted credits, invalid
        # request). Persist it at WARNING; the friendly string alone hid it.
        logger.warning(
            "[upstream-error] %s %s -> %s | body: %s",
            status, _host_key(target_url), friendly, body,
        )
    payload = {"status": status, "text": friendly, "raw": body}
    _ra = _parse_retry_after(resp_headers)
    if _ra is not None:
        payload["retry_after"] = _ra
    return f'event: error\ndata: {json.dumps(payload)}\n\n'


def _summarize_stream_error(err_chunk: Optional[str]) -> str:
    """Pull a short human reason out of an `event: error` SSE chunk for the
    fallback notice. Returns a generic message if it can't be parsed."""
    if not err_chunk:
        return "primary model failed"
    try:
        for line in err_chunk.split("\n"):
            if line.startswith("data: "):
                j = json.loads(line[6:])
                txt = j.get("text") or j.get("error") or ""
                status = j.get("status")
                msg = (f"HTTP {status}: " if status else "") + str(txt)
                return msg[:200].strip() or "primary model failed"
    except Exception:
        pass
    return "primary model failed"


# Pre-content transient statuses worth retrying the SAME candidate on before
# advancing: rate-limit (429) and server-side 5xx. 400/401/403 are NOT here —
# they're deterministic (bad request / auth / permission), so retrying just
# burns another round-trip; they trip the circuit breaker instead.
_RETRYABLE_STREAM_STATUSES = frozenset({429, 500, 502, 503, 504})
_STREAM_RETRY_MAX = 2          # extra attempts at the same candidate
_STREAM_RETRY_BASE_DELAY = 0.5  # exponential backoff base (s): 0.5, 1.0, ...
_STREAM_RETRY_MAX_DELAY = 15.0  # cap on any single backoff / Retry-After wait


def _error_chunk_status(chunk: Optional[str]) -> Optional[int]:
    """Extract the HTTP status from an `event: error` SSE chunk, if present."""
    if not chunk:
        return None
    try:
        for line in chunk.split("\n"):
            if line.startswith("data: "):
                j = json.loads(line[6:])
                s = j.get("status")
                return int(s) if s is not None else None
    except Exception:
        pass
    return None


def _error_chunk_retry_after(chunk: Optional[str]) -> Optional[float]:
    """Extract a parsed Retry-After (seconds) from an `event: error` chunk."""
    if not chunk:
        return None
    try:
        for line in chunk.split("\n"):
            if line.startswith("data: "):
                j = json.loads(line[6:])
                ra = j.get("retry_after")
                return float(ra) if ra is not None else None
    except Exception:
        pass
    return None


async def stream_llm_with_fallback(candidates, messages, **kwargs):
    """Wrap stream_llm with an ordered fallback chain + a per-endpoint circuit
    breaker + same-candidate transient retry.

    `candidates` is a list of (url, model, headers). Each is tried in order,
    but only retried on a *pre-content* failure — i.e. an ``event: error``
    that arrives before any assistant text / tool-call data has been yielded.
    Once a candidate has emitted real output we never switch (that would
    duplicate streamed tokens); a later error from that candidate passes
    through unchanged.

    Three reliability layers, all pre-content only:
      1. Circuit breaker: an endpoint (host, model) that just 401'd / repeatedly
         failed is SKIPPED so a dead primary stops eating a round-trip every
         agent round. If EVERY candidate is broken, we try anyway (better a
         long-shot attempt than an instant hard failure).
      2. Same-candidate retry: on a transient pre-content status (429/5xx) or a
         connect error, retry the SAME candidate up to _STREAM_RETRY_MAX times
         with exponential backoff, honoring Retry-After (capped at 15s), before
         advancing. 400/401/403 are never retried (they trip the breaker).
      3. Ordered fallback: then advance to the next candidate as before.

    Yields the same SSE chunk protocol as stream_llm.
    """
    cands = _dedupe_candidates(candidates)
    if not cands:
        yield f'event: error\ndata: {json.dumps({"error": "No model endpoint configured", "status": 503})}\n\n'
        return

    # Circuit breaker: prefer healthy candidates, but never fail outright just
    # because all are cooling — fall back to the full list so a transient global
    # blip can't wedge chat.
    healthy = [(u, m, h) for (u, m, h) in cands if not _is_endpoint_unhealthy(u, m)]
    if healthy and len(healthy) < len(cands):
        skipped = [m for (u, m, h) in cands if _is_endpoint_unhealthy(u, m)]
        logger.warning(f"[circuit] skipping unhealthy candidate(s) {skipped}; {len(healthy)} healthy left")
        run_cands = healthy
    elif not healthy:
        logger.warning("[circuit] all candidates unhealthy — trying anyway rather than hard-failing")
        run_cands = cands
    else:
        run_cands = cands

    primary_model = run_cands[0][1]
    last_error = None
    for i, (url, model, headers) in enumerate(run_cands):
        is_last = (i == len(run_cands) - 1)
        emitted = False
        advance = False  # move to the next candidate after this one
        attempt = 0
        while True:
            emitted = False
            got_error = None  # the pre-content error chunk this attempt produced
            async for chunk in stream_llm(url, model, messages, headers=headers, **kwargs):
                if chunk.startswith("event: error"):
                    if not emitted:
                        # Pre-content failure. Decide retry-same vs advance vs surface.
                        got_error = chunk
                        break
                    # Post-content error: pass through unchanged (can't switch now).
                    yield chunk
                    continue
                # Any data chunk other than the terminal [DONE] means real output.
                if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                    try:
                        event_data = json.loads(chunk[6:])
                    except Exception:
                        event_data = {}
                    if event_data.get("type") == "model_actual":
                        yield chunk
                        continue
                    if not emitted:
                        # Real output — this endpoint is healthy; clear any breaker
                        # state and (if a non-primary answered) announce the switch.
                        _clear_endpoint_unhealthy(url, model)
                        if i > 0:
                            yield ('data: ' + json.dumps({
                                "type": "fallback",
                                "selected_model": primary_model,
                                "answered_by": model,
                                "reason": _summarize_stream_error(last_error),
                            }) + '\n\n')
                    emitted = True
                yield chunk

            if got_error is None:
                # Candidate finished (success or a post-content terminal error
                # already forwarded). Done with the whole chain.
                return

            # Pre-content failure handling.
            last_error = got_error
            status = _error_chunk_status(got_error)
            # Feed the circuit breaker (connect errors surface as 503 here too;
            # the connect path already marks the host, this also flags the model).
            _cooled = _mark_endpoint_status_failure(url, model, status)
            _retryable = (status is None) or (status in _RETRYABLE_STREAM_STATUSES)
            # Auth/permission/bad-request are deterministic — never retry same.
            _hard = status in _ENDPOINT_HARD_STATUSES or status == 400
            if _retryable and not _hard and attempt < _STREAM_RETRY_MAX:
                attempt += 1
                ra = _error_chunk_retry_after(got_error)
                if ra is not None:
                    delay = min(ra, _STREAM_RETRY_MAX_DELAY)
                else:
                    delay = min(_STREAM_RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                                _STREAM_RETRY_MAX_DELAY)
                logger.warning(
                    f"[retry] {model} pre-content {status or 'connect'} — "
                    f"retry {attempt}/{_STREAM_RETRY_MAX} in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                continue  # retry SAME candidate
            # Out of same-candidate retries (or a hard status): advance.
            _tail = f" (endpoint cooled {_circuit_cooldown_seconds():.0f}s)" if _cooled else ""
            if not is_last:
                if i == 0:
                    logger.warning(f"[fallback] primary {model} failed before output ({status}){_tail}; trying fallback")
                else:
                    logger.warning(f"[fallback] candidate {model} failed ({status}){_tail}; trying next")
                advance = True
            break
        if not advance:
            # Last candidate exhausted its retries — surface the error.
            break
    # Every candidate failed pre-content — surface the last error.
    if last_error:
        yield last_error
