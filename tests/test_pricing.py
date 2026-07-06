"""Unit tests for src/pricing.py — model matching, cache-aware math, local-free,
unknown-model=0, and the price_usage served-model / endpoint behaviour."""

import math

import pytest

from src import pricing


# --- model matching ---------------------------------------------------------

def test_match_model_key_longest_wins():
    # "gpt-4o-mini" must match the longer key, not the shorter "gpt-4o" (would
    # bill ~16x). Mirrors the frontend matchModelKey guarantee.
    assert pricing.match_model_key("gpt-4o-mini") == "gpt-4o-mini"
    assert pricing.match_model_key("gpt-4o") == "gpt-4o"


def test_match_model_key_normalizes_and_handles_prefixes():
    # Case-insensitive + tolerant of provider prefixes / path segments.
    assert pricing.match_model_key("Claude-Opus-4-8") == "claude-opus-4-8"
    assert pricing.match_model_key("anthropic.claude-opus-4-8") == "claude-opus-4-8"
    assert pricing.match_model_key("openrouter/claude-sonnet-5") == "claude-sonnet-5"


def test_match_model_key_unknown_returns_none():
    assert pricing.match_model_key("totally-unknown-model") is None
    assert pricing.match_model_key("") is None
    assert pricing.match_model_key(None) is None


# --- Gemini 3.x rows (added once Gemini 3 models started shipping) --------- #

@pytest.mark.parametrize("model_id,expected_key", [
    ("gemini-3.5-flash", "gemini-3.5-flash"),
    ("gemini-3.5-flash-preview", "gemini-3.5-flash"),
    ("gemini-3.1-pro", "gemini-3.1-pro"),
    ("gemini-3.1-pro-preview", "gemini-3.1-pro"),
    ("gemini-3.1-flash-lite", "gemini-3.1-flash-lite"),
    ("gemini-3.1-flash-lite-preview", "gemini-3.1-flash-lite"),
    ("gemini-3-flash", "gemini-3-flash"),
    ("gemini-3-flash-preview", "gemini-3-flash"),
])
def test_match_model_key_gemini_3x_specific_rows(model_id, expected_key):
    # Each Gemini 3.x row must resolve to its OWN specific key, not fall back
    # to the shorter "gemini-3" general-fallback row (which would misprice
    # every 3.x variant at the mirrored 2.5-pro rate instead of its real one).
    assert pricing.match_model_key(model_id) == expected_key


def test_match_model_key_gemini_3_bare_falls_back_to_general_row():
    # No published rate exists for a bare/general gemini-3 (pro) tier as of
    # this table's last update; it resolves to the "gemini-3" fallback row
    # (mirroring gemini-2.5-pro), not to one of the more specific 3.x rows.
    assert pricing.match_model_key("gemini-3") == "gemini-3"
    assert pricing.match_model_key("gemini-3-pro-preview") == "gemini-3"


# --- cost_for_tokens: base + cache-aware ------------------------------------

def test_cost_for_tokens_input_output():
    # opus-4-8 = 5/25 per MTok.
    cost = pricing.cost_for_tokens("claude-opus-4-8", fresh_input=1_000_000, output=1_000_000)
    assert math.isclose(cost, 5.00 + 25.00, rel_tol=1e-9)


def test_cost_for_tokens_cache_aware_anthropic():
    # opus-4-8 cache_read 0.50, cache_write 6.25 per MTok.
    cost = pricing.cost_for_tokens(
        "claude-opus-4-8",
        fresh_input=1_000_000,   # 5.00
        output=0,
        cache_read=1_000_000,    # 0.50
        cache_write=1_000_000,   # 6.25
    )
    assert math.isclose(cost, 5.00 + 0.50 + 6.25, rel_tol=1e-9)


def test_cost_for_tokens_sonnet5_intro_rates():
    # sonnet-5 introductory 2/10 (read 0.20, write 2.50).
    cost = pricing.cost_for_tokens(
        "claude-sonnet-5",
        fresh_input=1_000_000,
        output=1_000_000,
        cache_read=1_000_000,
        cache_write=1_000_000,
    )
    assert math.isclose(cost, 2.00 + 10.00 + 0.20 + 2.50, rel_tol=1e-9)


def test_cost_for_tokens_non_anthropic_cache_defaults_to_input_rate():
    # gpt-4o has no published cache rates → cache_read/write default to input
    # rate (2.50) so cache tokens are never mispriced.
    cost = pricing.cost_for_tokens(
        "gpt-4o", fresh_input=0, output=0, cache_read=1_000_000, cache_write=1_000_000
    )
    assert math.isclose(cost, 2.50 + 2.50, rel_tol=1e-9)


def test_cost_for_tokens_unknown_model_is_zero():
    assert pricing.cost_for_tokens("some-local-model", fresh_input=1_000_000) == 0.0


# --- local-endpoint detection (mirror of frontend isLocalEndpoint) ----------

@pytest.mark.parametrize("url", [
    "http://localhost:8080/v1/chat/completions",
    "http://127.0.0.1:11434/v1",
    "http://10.0.0.5:8000/v1",
    "http://192.168.1.20:1234/v1",
    "http://172.16.5.5:8000/v1",
    "http://100.64.0.1:8080/v1",       # Tailscale CGNAT
    "http://100.127.255.254:8080/v1",  # CGNAT upper bound
    "http://nemotron:8000/v1",          # single-label docker service name
    "http://myhost.local:8000/v1",      # .local
    "http://host.docker.internal:8000/v1",
    "",                                  # missing → free (bias to not over-bill)
    None,
])
def test_host_shape_local_are_free(url):
    # is_endpoint_free returns False for None/'' (falls through to unknown->0),
    # so test _host_is_local directly for the host-shape cases, and
    # is_endpoint_free for real local URLs.
    if url:
        assert pricing.is_endpoint_free(url) is True
    else:
        # No endpoint → treated as non-local so pricing falls through to the
        # model table (documented degradation), NOT declared free.
        assert pricing.is_endpoint_free(url) is False


@pytest.mark.parametrize("url", [
    "https://api.anthropic.com/v1/messages",
    "https://api.openai.com/v1/chat/completions",
    "https://openrouter.ai/api/v1/chat/completions",
    "http://100.200.0.1:8080/v1",  # 100.x OUTSIDE CGNAT → public
])
def test_public_endpoints_are_billed(url):
    assert pricing.is_endpoint_free(url) is False


# --- served_model_of --------------------------------------------------------

def test_served_model_prefers_served_over_requested():
    usage = {"requested_model": "claude-fable-5", "served_model": "claude-opus-4-8"}
    assert pricing.served_model_of(usage) == "claude-opus-4-8"


def test_served_model_falls_back_to_model_then_requested():
    assert pricing.served_model_of({"model": "gpt-4o"}) == "gpt-4o"
    assert pricing.served_model_of({"requested_model": "gpt-4o"}) == "gpt-4o"
    assert pricing.served_model_of({}, fallback_model="claude-haiku-4-5") == "claude-haiku-4-5"


# --- price_usage: end-to-end ------------------------------------------------

def test_price_usage_prices_served_model_not_requested():
    # A fallback served the turn: requested fable-5 (10/50) but opus-4-8 (5/25)
    # answered → must be billed at opus rates.
    usage = {
        "requested_model": "claude-fable-5",
        "served_model": "claude-opus-4-8",
        "input_tokens": 1_000_000,
        "output_tokens": 0,
    }
    cost = pricing.price_usage(usage, endpoint_url="https://api.anthropic.com/v1/messages")
    assert math.isclose(cost, 5.00, rel_tol=1e-9)  # opus, not fable's 10.00


def test_price_usage_cache_aware():
    usage = {
        "model": "claude-opus-4-8",
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "cache_read_tokens": 1_000_000,
        "cache_creation_tokens": 1_000_000,
    }
    cost = pricing.price_usage(usage, endpoint_url="https://api.anthropic.com/v1/messages")
    assert math.isclose(cost, 5.00 + 25.00 + 0.50 + 6.25, rel_tol=1e-9)


def test_price_usage_local_endpoint_is_free():
    usage = {"model": "claude-opus-4-8", "input_tokens": 1_000_000, "output_tokens": 1_000_000}
    assert pricing.price_usage(usage, endpoint_url="http://localhost:8080/v1") == 0.0
    assert pricing.price_usage(usage, endpoint_url="http://nemotron:8000/v1") == 0.0


def test_price_usage_unknown_model_is_zero():
    usage = {"model": "my-local-llama", "input_tokens": 1_000_000, "output_tokens": 1_000_000}
    # Public endpoint but unknown model → 0 (documented fallback).
    assert pricing.price_usage(usage, endpoint_url="https://example.com/v1") == 0.0


def test_price_usage_no_endpoint_prices_from_table():
    # endpoint_url=None: not declared free; unknown->0, known->priced.
    known = {"model": "claude-opus-4-8", "input_tokens": 1_000_000}
    assert math.isclose(pricing.price_usage(known), 5.00, rel_tol=1e-9)
    unknown = {"model": "local-thing", "input_tokens": 1_000_000}
    assert pricing.price_usage(unknown) == 0.0
