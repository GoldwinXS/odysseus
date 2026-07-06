"""Server-authoritative USD cost accounting for LLM token usage.

The frontend historically priced turns with a static ``MODEL_PRICING`` table in
``static/js/chatRenderer.js`` (``getModelCost(model, input, output)``) plus an
``isLocalEndpoint()`` heuristic that treats self-hosted models as free. That
path had three problems this module fixes:

  1. it priced by the *requested* model even when a fallback served the turn;
  2. it ignored cache economics (cache reads ~0.1x input, cache writes ~1.25x);
  3. it had no server-side source of truth, so a displayed cost could not be
     trusted or persisted.

``price_usage`` here is the single server-side entry point. It prices a usage
dict (as emitted by ``llm_core``'s per-round ``usage`` events and
``agent_loop``'s per-turn ``metrics`` event) with the SERVED model, cache-aware,
and returns 0.0 for local/self-hosted endpoints and for models absent from the
table. Rates are USD per million tokens (per-MTok), mirroring the frontend
table and the Anthropic public pricing.

Local-endpoint detection mirrors the frontend ``isLocalEndpoint()`` logic
(loopback, RFC1918, Tailscale CGNAT 100.64-127.x, ``.local``, single-label
hosts, ``host.docker.internal``) and additionally honours the admin-configured
endpoint kind via ``src.model_context.is_local_endpoint`` when a base URL is
resolvable in the DB. When no endpoint URL is available at the pricing site,
unknown models simply cost 0 (documented below and in the callers).
"""

import ipaddress
import logging
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ── Pricing table (USD per million tokens) ───────────────────────────────── #
#
# input  = fresh (uncached) input tokens
# output = output tokens
# cache_read  = tokens served from an Anthropic prompt cache (~0.1x input)
# cache_write = tokens written to an Anthropic prompt cache (~1.25x input)
#
# Substring/normalized matching (see match_model_key) picks the longest key
# that is a substring of the normalized model id, mirroring the frontend's
# matchModelKey — so "claude-opus-4-8" matches before a shorter "claude-opus".
#
# Non-Anthropic cloud models are ported from the frontend MODEL_PRICING table.
# They lack published prompt-cache rates here, so cache_read defaults to the
# input rate (no cache discount assumed) and cache_write to the input rate (no
# write premium assumed) — see _entry(). If a provider's real cache rates are
# known later, add them explicitly.
MODEL_PRICING: dict[str, dict[str, float]] = {}


def _entry(
    key: str,
    inp: float,
    out: float,
    cache_read: Optional[float] = None,
    cache_write: Optional[float] = None,
) -> None:
    """Register a pricing row. When cache rates are unknown (non-Anthropic
    ports), default cache_read to the input rate and cache_write to the input
    rate (i.e. no discount / no premium) so cache tokens are never mispriced
    downward or upward relative to plain input."""
    MODEL_PRICING[key] = {
        "input": inp,
        "output": out,
        "cache_read": cache_read if cache_read is not None else inp,
        "cache_write": cache_write if cache_write is not None else inp,
    }


# --- Anthropic (cache rates: read ~0.1x input, write ~1.25x input) ---
# Seeded from the claude-api pricing reference (cached 2026-06-24):
#   fable-5      10 / 50   (read 1.00, write 12.50)
#   opus 4-8/4-7/4-6  5 / 25    (read 0.50, write 6.25)
#   sonnet-5     2 / 10 introductory (read 0.20, write 2.50)
#                NOTE: list price is 3 / 15 from 2026-09-01 — update then.
#   sonnet-4-6   3 / 15    (read 0.30, write 3.75)
#   haiku-4-5    1 / 5     (read 0.10, write 1.25)
_entry("claude-fable-5",    10.00, 50.00, 1.00, 12.50)
_entry("claude-mythos-5",   10.00, 50.00, 1.00, 12.50)
_entry("claude-opus-4-8",    5.00, 25.00, 0.50,  6.25)
_entry("claude-opus-4-7",    5.00, 25.00, 0.50,  6.25)
_entry("claude-opus-4-6",    5.00, 25.00, 0.50,  6.25)
# sonnet-5: 2/10 intro (read 0.20, write 2.50); list price 3/15 from 2026-09-01.
_entry("claude-sonnet-5",    2.00, 10.00, 0.20,  2.50)
_entry("claude-sonnet-4-6",  3.00, 15.00, 0.30,  3.75)
_entry("claude-haiku-4-5",   1.00,  5.00, 0.10,  1.25)
# Older Anthropic ids the frontend table also carries (cache = 0.1x/1.25x input).
_entry("claude-opus-4-5",    5.00, 25.00, 0.50,  6.25)
_entry("claude-sonnet-4-5",  3.00, 15.00, 0.30,  3.75)
_entry("claude-sonnet-4",    3.00, 15.00, 0.30,  3.75)
_entry("claude-opus-4",     15.00, 75.00, 1.50, 18.75)
_entry("claude-haiku-4",     0.80,  4.00, 0.08,  1.00)
_entry("claude-haiku-3-5",   0.80,  4.00, 0.08,  1.00)
_entry("claude-3-5-sonnet",  3.00, 15.00, 0.30,  3.75)
_entry("claude-3-5-haiku",   0.80,  4.00, 0.08,  1.00)
_entry("claude-3-opus",     15.00, 75.00, 1.50, 18.75)
_entry("claude-3-sonnet",    3.00, 15.00, 0.30,  3.75)
_entry("claude-3-haiku",     0.25,  1.25, 0.03,  0.31)

# --- OpenAI (ported from frontend MODEL_PRICING; no cache rates) ---
_entry("gpt-5",              2.00,  8.00)
_entry("gpt-4.1",           2.00,  8.00)
_entry("gpt-4.1-mini",      0.40,  1.60)
_entry("gpt-4.1-nano",      0.10,  0.40)
_entry("gpt-4o",            2.50, 10.00)
_entry("gpt-4o-mini",       0.15,  0.60)
_entry("gpt-4-turbo",      10.00, 30.00)
_entry("o1",               15.00, 60.00)
_entry("o1-mini",           3.00, 12.00)
_entry("o1-pro",          150.00,600.00)
_entry("o3",                2.00,  8.00)
_entry("o3-mini",           1.10,  4.40)
_entry("o4-mini",           1.10,  4.40)

# --- DeepSeek ---
_entry("deepseek-chat",     0.27,  1.10)
_entry("deepseek-coder",    0.27,  1.10)
_entry("deepseek-reasoner", 0.55,  2.19)
_entry("deepseek-r1",       0.55,  2.19)
_entry("deepseek-v3",       0.27,  1.10)
_entry("deepseek-v2",       0.14,  0.28)

# --- Google ---
# gemini-3.x rows added 2026-07-06 from ai.google.dev/gemini-api/docs/pricing
# (fetched live; verify against the current page if a turn's cost looks off —
# Gemini 3.x was in preview and rates may still move). Keys use the base
# name (no -preview/-002 suffix) so match_model_key's longest-substring match
# still hits the runtime model id.
_entry("gemini-3.5-flash",     1.50,  9.00)
_entry("gemini-3.1-pro",       2.00, 12.00)   # <=200k token prompts; page also
                                              # lists a >200k tier at 4.00/18.00
_entry("gemini-3.1-flash-lite",0.25,  1.50)
_entry("gemini-3-flash",       0.50,  3.00)
# No published rate found for a general/bare "gemini-3" (pro) tier as of the
# 2026-07-06 fetch — mirroring the closest same-tier row (2.5-pro) per
# instructions until Google publishes one. Update when a real rate appears.
_entry("gemini-3",             1.25, 10.00)
_entry("gemini-2.5-pro",    1.25, 10.00)
_entry("gemini-2.5-flash",  0.15,  0.60)
_entry("gemini-2.0-flash",  0.10,  0.40)
_entry("gemini-1.5-pro",    1.25,  5.00)
_entry("gemini-1.5-flash",  0.075, 0.30)
_entry("gemma-3",           0.10,  0.10)

# --- Mistral ---
_entry("mistral-large",     2.00,  6.00)
_entry("mistral-medium",    2.00,  6.00)
_entry("mistral-small",     0.20,  0.60)
_entry("mistral-nemo",      0.15,  0.15)
_entry("mixtral",           0.24,  0.24)
_entry("codestral",         0.30,  0.90)
_entry("pixtral",           2.00,  6.00)

# --- xAI ---
_entry("grok-4",            3.00, 15.00)
_entry("grok-3",            3.00, 15.00)
_entry("grok-2",            2.00, 10.00)

# --- Meta ---
_entry("llama-4",           0.20,  0.20)
_entry("llama-3.3",         0.20,  0.20)
_entry("llama-3.2",         0.20,  0.20)
_entry("llama-3.1",         0.20,  0.20)
_entry("llama-3",           0.20,  0.20)

# --- Qwen ---
_entry("qwen3",             0.30,  1.20)
_entry("qwen2.5",           0.30,  1.20)
_entry("qwq",               0.30,  1.20)

# --- Cohere ---
_entry("command-a",         2.50, 10.00)
_entry("command-r-plus",    2.50, 10.00)
_entry("command-r",         0.15,  0.60)

# --- Perplexity ---
_entry("sonar-pro",         3.00, 15.00)
_entry("sonar",             1.00,  1.00)

# --- MiniMax / Moonshot / Microsoft / Nvidia / Nous ---
_entry("minimax",           0.70,  0.70)
_entry("moonshot",          1.00,  1.00)
_entry("kimi",              1.00,  1.00)
_entry("phi-4",             0.07,  0.14)
_entry("phi-3",             0.07,  0.14)
_entry("nemotron",          0.30,  1.20)
_entry("hermes",            0.20,  0.20)


# ── Model matching ───────────────────────────────────────────────────────── #

def match_model_key(name: str) -> Optional[str]:
    """Return the most specific (longest) pricing key that is a substring of
    the normalized model name, or None. Mirrors the frontend matchModelKey:
    returning the first match instead would let "gpt-4o-mini" match the shorter
    "gpt-4o" key and bill at ~16x."""
    n = (name or "").lower()
    if not n:
        return None
    # Provider prefixes (openrouter, bedrock's "anthropic.", etc.) and path
    # segments must not defeat substring matching — the keys are already bare
    # ids, and str.__contains__ ignores the surrounding text, so no stripping
    # is needed. We only lowercase.
    best: Optional[str] = None
    for key in MODEL_PRICING:
        if key in n and (best is None or len(key) > len(best)):
            best = key
    return best


# ── Local-endpoint detection (mirror of the frontend isLocalEndpoint) ─────── #

_LOCAL_HOST_LITERALS = {"localhost", "0.0.0.0", "host.docker.internal", "::1"}


def _host_is_local(host: str) -> bool:
    host = (host or "").strip().lower()
    if not host:
        return True  # missing host → bias to not over-bill (matches frontend)
    if host in _LOCAL_HOST_LITERALS or host.endswith(".local"):
        return True
    # A single-label hostname (no dot) is an internal/Docker service name or a
    # LAN shortname — never a public API, which always needs an FQDN.
    if "." not in host:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.is_loopback or ip.is_private or ip.is_link_local:
        return True
    # Tailscale CGNAT 100.64.0.0/10 (100.64.x – 100.127.x).
    return ip in ipaddress.ip_network("100.64.0.0/10")


def is_endpoint_free(endpoint_url: Optional[str]) -> bool:
    """True when the serving endpoint is local/self-hosted → the model is free.

    Combines the admin-configured endpoint kind (via
    ``src.model_context.is_local_endpoint`` — which reads ModelEndpoint rows to
    honour an explicit local/api/proxy classification) with the frontend's
    host-shape heuristic. When ``endpoint_url`` is falsy, returns False so that
    the caller falls back to "unknown model → 0" pricing rather than declaring
    every model free.
    """
    if not endpoint_url:
        return False
    # Prefer the DB-backed classification when it can resolve the endpoint: an
    # admin may have marked a public-looking host as "local", or a private-IP
    # host as "api"/"proxy" (billed). is_local_endpoint honours endpoint_kind.
    try:
        from src.model_context import is_local_endpoint as _cfg_is_local

        if _cfg_is_local(endpoint_url):
            return True
    except Exception:
        pass
    try:
        host = urlparse(endpoint_url).hostname or ""
    except Exception:
        return False
    return _host_is_local(host)


# ── Cost computation ─────────────────────────────────────────────────────── #

def cost_for_tokens(
    model: str,
    fresh_input: int = 0,
    output: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
) -> float:
    """Cache-aware USD cost for a set of token counts priced by ``model``.

    Returns 0.0 when the model is not in the pricing table (documented
    fallback: an unknown or local model costs nothing). ``fresh_input`` is the
    UNCACHED input; cache_read/cache_write are billed at their own rates. Total
    input reported by a provider is fresh_input + cache_read + cache_write, so
    callers that only have the combined ``input_tokens`` should subtract the
    cache counts before passing fresh_input (see ``price_usage``).
    """
    key = match_model_key(model)
    if not key:
        return 0.0
    p = MODEL_PRICING[key]
    return (
        max(fresh_input, 0) * p["input"]
        + max(output, 0) * p["output"]
        + max(cache_read, 0) * p["cache_read"]
        + max(cache_write, 0) * p["cache_write"]
    ) / 1_000_000.0


def _usage_tokens(usage: dict) -> tuple[int, int, int, int]:
    """Extract (input_tokens, output_tokens, cache_read, cache_write) from a
    usage/metrics dict, tolerating both the lean llm_core keys and the
    Anthropic-native ``*_input_tokens`` spellings."""
    def _int(*keys) -> int:
        for k in keys:
            v = usage.get(k)
            if v:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    return 0
        return 0

    input_tokens = _int("input_tokens")
    output_tokens = _int("output_tokens")
    cache_read = _int("cache_read_tokens", "cache_read_input_tokens")
    cache_write = _int("cache_creation_tokens", "cache_creation_input_tokens")
    return input_tokens, output_tokens, cache_read, cache_write


def served_model_of(usage: dict, fallback_model: Optional[str] = None) -> str:
    """The model that ACTUALLY produced this usage — prefer served_model, then
    the reported model, then requested/selected, then the caller fallback.
    Pricing by the served model is the whole point: a fallback that answered a
    turn must be billed at its own rate, not the requested model's."""
    for k in ("served_model", "model", "actual_model", "requested_model", "selected_model"):
        v = usage.get(k)
        if v and str(v).strip():
            return str(v).strip()
    return (fallback_model or "").strip()


def price_usage(usage: dict, endpoint_url: Optional[str] = None,
                fallback_model: Optional[str] = None) -> float:
    """Cache-aware USD cost for one usage/metrics dict, priced by the SERVED
    model, returning 0.0 for local/self-hosted endpoints and unknown models.

    ``endpoint_url`` is the serving endpoint (session.endpoint_url); when it is
    local/self-hosted the cost is 0. When it is None the endpoint is treated as
    non-local and pricing falls through to the model table (unknown → 0), which
    is the documented degradation when no endpoint info is available.

    Anthropic-style usage reports the combined input (``input_tokens``) with
    cache reads/writes reported separately; providers differ on whether cache
    tokens are *included in* ``input_tokens``. llm_core/agent_loop follow the
    Anthropic convention where ``input_tokens`` is the fresh (uncached) count
    and cache tokens are additive, so we bill ``input_tokens`` as fresh_input
    and add the cache tokens. (If a provider folded cache reads into
    ``input_tokens`` we would slightly over-count fresh input — acceptable and
    conservative, and it matches the frontend's cache-aware formula.)
    """
    if is_endpoint_free(endpoint_url):
        return 0.0
    model = served_model_of(usage, fallback_model)
    if not model:
        return 0.0
    fresh_input, output, cache_read, cache_write = _usage_tokens(usage)
    return cost_for_tokens(
        model,
        fresh_input=fresh_input,
        output=output,
        cache_read=cache_read,
        cache_write=cache_write,
    )
