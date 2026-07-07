"""Prompt-cache cost fix: for caching endpoints, the tool set is sent STABLE
(full) instead of per-turn RAG, so the tools+system cache prefix stays
byte-identical turn-to-turn (tools sit at the front of an Anthropic cache
prefix; a changing tool SET busts the whole cache and re-bills it ~10x).

Covers agent_loop._endpoint_caches_prompts (the gate). The stable set itself is
query-INDEPENDENT by construction (all built-in ∪ MCP ∪ always − disabled), so
stability across turns is guaranteed structurally, not by this test.
"""

import pytest

from src.agent_loop import _endpoint_caches_prompts


@pytest.mark.parametrize("url", [
    "https://api.anthropic.com/v1/messages",
    "https://api.anthropic.com",
    "http://api.anthropic.com.",   # trailing dot still matches the host
])
def test_anthropic_endpoints_cache(url):
    assert _endpoint_caches_prompts(url) is True


@pytest.mark.parametrize("url", [
    "",
    None,
    "http://localhost:11434/v1",           # local Ollama — no prompt cache
    "https://api.z.ai/v1/chat",            # z.ai / GLM — no caching (would cost MORE)
    "https://api.novita.ai/v3/openai",     # novita reseller — no reliable cache
    "https://generativelanguage.googleapis.com",  # gemini (implicit) — not enabled yet, verify first
    "https://api.moonshot.cn/v1",          # moonshot — not enabled yet, verify first
])
def test_non_caching_endpoints_keep_rag(url):
    # Default to False so a non-caching provider isn't handed 17k tokens of tools
    # every turn with no cache discount (a pure cost increase).
    assert _endpoint_caches_prompts(url) is False
