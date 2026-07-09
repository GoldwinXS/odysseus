"""Regression tests for Anthropic prompt-cache breakpoints in _build_anthropic_payload (#791)."""
from src import llm_core


def _payload(system="sys", user="hi", tools=None):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return llm_core._build_anthropic_payload("claude", messages, 0.0, 1000, stream=True, tools=tools)


def _assert_cache_ctrl(cc):
    """Breakpoints are ephemeral with a ttl from the anthropic_cache_ttl
    setting (default '1h'; '5m' is the only other valid value)."""
    assert isinstance(cc, dict)
    assert cc.get("type") == "ephemeral"
    assert cc.get("ttl") in ("5m", "1h")


def test_agentic_caches_system_and_last_tool():
    tools = [
        {"type": "function", "function": {"name": "a", "description": "x", "parameters": {}}},
        {"type": "function", "function": {"name": "b", "description": "y", "parameters": {}}},
    ]
    p = _payload(system="SYS PROMPT " * 50, tools=tools)
    assert isinstance(p["system"], list)
    _assert_cache_ctrl(p["system"][0].get("cache_control"))
    assert "cache_control" not in p["tools"][0], "only the LAST tool is a breakpoint"
    _assert_cache_ctrl(p["tools"][-1].get("cache_control"))
    breakpoints = sum("cache_control" in b for b in p["system"]) + sum("cache_control" in t for t in p["tools"])
    assert breakpoints == 2


def test_tiny_tool_less_prompt_not_cached():
    p = _payload(system="hi", tools=None)
    assert isinstance(p["system"], list)
    assert "cache_control" not in p["system"][0]


def test_large_system_only_is_cached():
    p = _payload(system="z" * 5000, tools=None)
    _assert_cache_ctrl(p["system"][0].get("cache_control"))


# ── effort / output_config ──

def test_effort_omitted_by_default(monkeypatch):
    """No setting, no explicit value → no output_config (some models reject it)."""
    monkeypatch.setattr(llm_core, "_resolve_anthropic_effort", lambda explicit=None: None)
    p = _payload()
    assert "output_config" not in p


def test_explicit_effort_sets_output_config():
    p = llm_core._build_anthropic_payload(
        "claude", [{"role": "user", "content": "hi"}], 0.0, 1000, effort="high"
    )
    assert p["output_config"] == {"effort": "high"}


def test_effort_from_setting(monkeypatch):
    """Settings-driven default: anthropic_effort flows into output_config."""
    import src.settings as settings
    monkeypatch.setattr(settings, "get_setting",
                        lambda key, default=None: "low" if key == "anthropic_effort" else default)
    p = _payload()
    assert p["output_config"] == {"effort": "low"}


def test_no_thinking_param_ever():
    """We deliberately never send a `thinking` param (fable models reject it)."""
    p = llm_core._build_anthropic_payload(
        "claude", [{"role": "user", "content": "hi"}], 0.0, 1000, effort="high"
    )
    assert "thinking" not in p
