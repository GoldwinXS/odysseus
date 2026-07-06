"""Regression: the agent-mode SSE forwarding whitelist in chat_routes.py silently
dropped several event types that agent_loop.py emits and the frontend already
has handlers for.

``stream_with_save``'s agent-mode branch parses each SSE chunk and only
re-yields it to the client when ``data["type"]`` is in a hardcoded tuple. Any
event type NOT in that tuple is parsed and then discarded — it never reaches
the live client, the replay buffer (agent_runs), or a second connected device.

Confirmed missing before this fix, despite the frontend already handling them:
  - ``steering_injected`` (a mid-turn user/steer message injected by
    agent_loop.py) — this was the CONFIRMED root cause of "steering messages
    don't appear on other devices".
  - ``tool_progress`` (live elapsed/tail updates for long-running tools)
  - ``agent_prep`` (per-turn prep-phase timing breakdown)
  - ``budget_exceeded`` (tool-call budget hit mid-turn)

This test pins the actual whitelist tuple in the source (the same
source-wiring-pin approach ``test_partial_save_tool_history.py`` uses for a
comparable route-internals contract) rather than standing up the full
streaming/session/DB machinery ``stream_with_save`` needs to execute — that
machinery is disproportionate to what this fix touches (one tuple literal).
"""
import re
from pathlib import Path

_ROUTES_SRC = (Path(__file__).resolve().parents[1] / "routes" / "chat_routes.py").read_text(encoding="utf-8")


def _agent_mode_whitelist_tuple_source() -> str:
    """Extract the source text of the `data.get("type") in (...)` tuple in the
    agent-mode branch of stream_with_save, so the assertions below check the
    ACTUAL tuple contents rather than merely searching the whole file (which
    could false-pass on an unrelated string elsewhere)."""
    marker = 'elif data.get("type") in ('
    start = _ROUTES_SRC.index(marker)
    # Find the matching close paren for the tuple opened right after the marker.
    depth = 0
    i = start + len(marker) - 1  # index of the opening "("
    assert _ROUTES_SRC[i] == "("
    for j in range(i, len(_ROUTES_SRC)):
        if _ROUTES_SRC[j] == "(":
            depth += 1
        elif _ROUTES_SRC[j] == ")":
            depth -= 1
            if depth == 0:
                return _ROUTES_SRC[i:j + 1]
    raise AssertionError("could not find the end of the agent-mode whitelist tuple")


def _quoted_string_literals(tuple_src: str) -> set:
    return set(re.findall(r'"([^"\\]*)"', tuple_src))


def test_steering_injected_is_forwarded():
    """CONFIRMED root cause of steering messages not appearing on other
    devices: steering_injected must be forwarded, not silently dropped."""
    literals = _quoted_string_literals(_agent_mode_whitelist_tuple_source())
    assert "steering_injected" in literals


def test_tool_progress_agent_prep_budget_exceeded_are_forwarded():
    """These three have frontend handlers already (chat.js tool_progress at
    ~2004/2834, agent_prep at ~2009) but were absent from the whitelist —
    parsed then silently dropped."""
    literals = _quoted_string_literals(_agent_mode_whitelist_tuple_source())
    assert "tool_progress" in literals
    assert "agent_prep" in literals
    assert "budget_exceeded" in literals


def test_previously_whitelisted_types_are_not_regressed():
    """Guard against the fix accidentally narrowing the tuple instead of only
    extending it — every event type forwarded before this fix must still be
    forwarded after it."""
    literals = _quoted_string_literals(_agent_mode_whitelist_tuple_source())
    pre_existing = {
        "tool_start", "tool_output", "agent_step",
        "doc_stream_open", "doc_stream_delta",
        "doc_update", "doc_suggestions", "ui_control",
        "rounds_exhausted", "ask_user", "plan_update",
    }
    assert pre_existing.issubset(literals)
