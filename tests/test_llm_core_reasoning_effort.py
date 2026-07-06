"""Unit tests for apply_reasoning_effort — the UI reasoning-effort selector
("off"/"low"/"medium"/"high"/"default") mapped to each provider's actual
request shape. Pure-function tests: call the helper directly against a bare
payload dict, no network/streaming involved.

Anthropic is intentionally NOT covered here as a payload mutation — it is a
no-op in apply_reasoning_effort by design (see its docstring); the selector
reaches Anthropic through the pre-existing `effort=`/`output_config` mechanism
in stream_llm instead, which test_llm_core_anthropic_cache.py already covers.
"""
from src import llm_core


def _payload(**overrides):
    p = {"model": "placeholder", "messages": [], "stream": True}
    p.update(overrides)
    return p


# ── "default" / falsy → always a no-op ──

def test_default_is_noop_for_every_provider():
    for url in (
        "https://api.openai.com/v1/chat/completions",
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "https://api.z.ai/api/paas/v4/chat/completions",
        "https://api.deepseek.com/v1/chat/completions",
        "http://host.docker.internal:11434/v1/chat/completions",
    ):
        p = _payload()
        llm_core.apply_reasoning_effort(p, "some-model", url, "default")
        assert p == _payload(), f"default must be a no-op for {url}"


def test_none_effort_is_noop():
    p = _payload()
    llm_core.apply_reasoning_effort(p, "gpt-5", "https://api.openai.com/v1/chat/completions", None)
    assert p == _payload()


def test_unrecognized_effort_string_is_noop():
    p = _payload()
    llm_core.apply_reasoning_effort(p, "gpt-5", "https://api.openai.com/v1/chat/completions", "extreme")
    assert p == _payload()


# ── Anthropic: always a no-op in this helper (handled separately) ──

def test_anthropic_is_noop_in_this_helper():
    for effort in ("off", "low", "medium", "high"):
        p = _payload()
        llm_core.apply_reasoning_effort(p, "claude-opus-4-6", "https://api.anthropic.com/v1/messages", effort)
        assert p == _payload(), f"anthropic must be untouched by apply_reasoning_effort for effort={effort}"
        assert "thinking" not in p


# ── OpenAI o-series / gpt-5 family ──

def test_openai_gpt5_low_medium_high():
    for effort in ("low", "medium", "high"):
        p = _payload()
        llm_core.apply_reasoning_effort(p, "gpt-5", "https://api.openai.com/v1/chat/completions", effort)
        assert p["reasoning_effort"] == effort


def test_openai_gpt5_off_maps_to_minimal():
    p = _payload()
    llm_core.apply_reasoning_effort(p, "gpt-5-mini", "https://api.openai.com/v1/chat/completions", "off")
    assert p["reasoning_effort"] == "minimal"


def test_openai_o_series_low_medium_high():
    for effort in ("low", "medium", "high"):
        p = _payload()
        llm_core.apply_reasoning_effort(p, "o3-mini", "https://api.openai.com/v1/chat/completions", effort)
        assert p["reasoning_effort"] == effort


def test_openai_o_series_off_is_noop_no_minimal_tier():
    p = _payload()
    llm_core.apply_reasoning_effort(p, "o3-mini", "https://api.openai.com/v1/chat/completions", "off")
    assert "reasoning_effort" not in p


def test_openai_non_reasoning_model_is_noop():
    p = _payload()
    llm_core.apply_reasoning_effort(p, "gpt-4o", "https://api.openai.com/v1/chat/completions", "high")
    assert "reasoning_effort" not in p


# ── Gemini OpenAI-compat ──

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


def test_gemini_25_flash_low_medium_high():
    for effort in ("low", "medium", "high"):
        p = _payload()
        llm_core.apply_reasoning_effort(p, "gemini-2.5-flash", _GEMINI_URL, effort)
        assert p["reasoning_effort"] == effort


def test_gemini_flash_off_maps_to_none():
    p = _payload()
    llm_core.apply_reasoning_effort(p, "gemini-2.5-flash", _GEMINI_URL, "off")
    assert p["reasoning_effort"] == "none"


def test_gemini_pro_off_is_noop_thinking_cannot_be_disabled():
    p = _payload()
    llm_core.apply_reasoning_effort(p, "gemini-2.5-pro", _GEMINI_URL, "off")
    assert "reasoning_effort" not in p


def test_gemini_pro_low_medium_high_still_applies():
    p = _payload()
    llm_core.apply_reasoning_effort(p, "gemini-2.5-pro", _GEMINI_URL, "medium")
    assert p["reasoning_effort"] == "medium"


def test_gemini_pre_25_model_is_noop():
    p = _payload()
    llm_core.apply_reasoning_effort(p, "gemini-1.5-pro", _GEMINI_URL, "high")
    assert "reasoning_effort" not in p


# ── Z.AI / GLM ──

def test_zai_glm45_enabled_for_low_medium_high():
    for effort in ("low", "medium", "high"):
        p = _payload()
        llm_core.apply_reasoning_effort(p, "glm-4.5", "https://api.z.ai/api/paas/v4/chat/completions", effort)
        assert p["thinking"] == {"type": "enabled"}


def test_zai_glm45_disabled_for_off():
    p = _payload()
    llm_core.apply_reasoning_effort(p, "glm-4.5", "https://api.z.ai/api/paas/v4/chat/completions", "off")
    assert p["thinking"] == {"type": "disabled"}


def test_zai_bigmodel_cn_host_also_recognized():
    p = _payload()
    llm_core.apply_reasoning_effort(p, "glm-5", "https://open.bigmodel.cn/api/paas/v4/chat/completions", "high")
    assert p["thinking"] == {"type": "enabled"}


def test_zai_pre_45_model_is_noop():
    p = _payload()
    llm_core.apply_reasoning_effort(p, "glm-4", "https://api.z.ai/api/paas/v4/chat/completions", "high")
    assert "thinking" not in p


# ── DeepSeek: always a no-op, including "off" ──

def test_deepseek_always_noop():
    for effort in ("off", "low", "medium", "high"):
        p = _payload()
        llm_core.apply_reasoning_effort(p, "deepseek-reasoner", "https://api.deepseek.com/v1/chat/completions", effort)
        assert p == _payload(), f"deepseek must never get a reasoning-effort field (effort={effort})"


# ── Ollama / local qwen3 "/no_think" message mutation ──

def test_qwen3_off_appends_no_think_to_last_user_message():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    p = _payload()
    llm_core.apply_reasoning_effort(p, "qwen3:14b", "http://host.docker.internal:11434/v1/chat/completions", "off", messages=messages)
    assert messages[-1]["content"] == "second\n/no_think"
    assert messages[1]["content"] == "first", "only the LAST user message is mutated"
    assert "reasoning_effort" not in p and "thinking" not in p, "qwen3 uses a message mutation, not a payload field"


def test_qwen3_low_medium_high_is_noop_no_granularity():
    messages = [{"role": "user", "content": "hi"}]
    for effort in ("low", "medium", "high"):
        msgs = [{"role": "user", "content": "hi"}]
        p = _payload()
        llm_core.apply_reasoning_effort(p, "qwen3:14b", "http://host.docker.internal:11434/v1/chat/completions", effort, messages=msgs)
        assert msgs[0]["content"] == "hi", f"no /no_think mutation expected for effort={effort}"
        assert p == _payload()


def test_qwen3_off_without_messages_arg_is_noop():
    """messages not supplied → nothing to mutate, must not raise."""
    p = _payload()
    llm_core.apply_reasoning_effort(p, "qwen3:14b", "http://host.docker.internal:11434/v1/chat/completions", "off")
    assert p == _payload()


def test_qwen3_off_multimodal_last_message_appends_text_block():
    messages = [{"role": "user", "content": [{"type": "text", "text": "describe this"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}]}]
    p = _payload()
    llm_core.apply_reasoning_effort(p, "qwen3-vl", "http://localhost:11434/v1/chat/completions", "off", messages=messages)
    assert messages[0]["content"][-1] == {"type": "text", "text": "/no_think"}


def test_non_qwen3_local_model_is_noop():
    messages = [{"role": "user", "content": "hi"}]
    p = _payload()
    llm_core.apply_reasoning_effort(p, "llama3.1:8b", "http://localhost:11434/v1/chat/completions", "off", messages=messages)
    assert messages[0]["content"] == "hi"
    assert p == _payload()


# ── Unrecognized host: always a no-op ──

def test_unrecognized_host_is_noop():
    p = _payload()
    llm_core.apply_reasoning_effort(p, "some-model", "https://my-custom-openrouter-like-proxy.example.com/v1/chat/completions", "high")
    assert p == _payload()
