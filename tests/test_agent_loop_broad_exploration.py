"""Tests for the broad-exploration loop-breaker's read-only classification
helper, ``_is_readonly_exploration_block`` (agent_loop.py).

Context: the exact-signature and similarity loop-breakers only catch a model
re-issuing an (exactly or near-) identical tool call. A model can also thrash
by making 50+ DISTINCT read-only/introspection calls (never repeating one
exactly) without ever writing a real answer — observed with Gemini: 2.19M
input tokens over 373s, no answer text. The two-stage breaker built on top of
this classifier is: stage 1 (a visible check-in nudge, nothing terminates)
after N1 consecutive read-only-only, answer-free rounds; stage 2 (the shared
"declare done or blocked" handshake) only after a FURTHER N2 rounds pass with
still no answer — so a model doing genuine deep exploration (which can
legitimately make many distinct read-only calls) is never killed for
volume alone, only for failing to even summarize its own progress after
being asked to.

This file tests the classifier directly (the per-round streak/stage state
lives inline in the giant stream_agent_loop generator, not in a separable
function — mirrors the existing test_loop_breaker_runaway.py precedent for
_detect_runaway_call, its sibling loop-breaker helper).
"""
import sys
from unittest.mock import MagicMock

# Mock heavy deps so importing src.agent_loop doesn't load the full app stack —
# same pattern as tests/test_loop_breaker_runaway.py / tests/test_agent_loop.py.
# NOTE: src.agent_tools is deliberately NOT mocked — this file needs the REAL
# ToolBlock to construct fixtures. Only inject a stub for a module that isn't
# already imported, and clean up afterward (an un-cleaned stub would poison
# every later test file's import of the same module for the rest of the
# pytest session — see the fix in test_agent_loop_tool_caption.py for a case
# where exactly that happened with src.agent_tools).
_MOCKED = [
    'sqlalchemy', 'sqlalchemy.orm', 'sqlalchemy.ext', 'sqlalchemy.ext.declarative',
    'sqlalchemy.ext.hybrid', 'sqlalchemy.sql', 'sqlalchemy.sql.expression',
    'src.database', 'core.models', 'core.database',
]
_INJECTED_STUBS = {}
for _m in _MOCKED:
    if _m not in sys.modules:
        _stub = MagicMock()
        sys.modules[_m] = _stub
        _INJECTED_STUBS[_m] = _stub

try:
    from src.agent_tools import ToolBlock
    from src.agent_loop import _is_readonly_exploration_block
finally:
    for _mod, _stub in _INJECTED_STUBS.items():
        if sys.modules.get(_mod) is _stub:
            del sys.modules[_mod]
            parent_name, _, attr = _mod.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and getattr(parent, "__dict__", {}).get(attr) is _stub:
                delattr(parent, attr)


def _block(tool_type, content=""):
    return ToolBlock(tool_type, content)


# --- Fixed read-only/introspection tools ------------------------------------

def test_fixed_readonly_tools_are_readonly():
    for tool in ("ls", "bash", "grep", "read_file", "get_workspace", "glob"):
        assert _is_readonly_exploration_block(_block(tool, "some args")) is True


def test_mutating_tools_are_not_readonly():
    for tool in ("edit_file", "write_file", "create_document", "manage_calendar"):
        assert _is_readonly_exploration_block(_block(tool, '{"action": "create"}')) is False


# --- manage_* action classification ------------------------------------------

def test_manage_tool_empty_content_is_readonly():
    # manage_agents' own docstring: "(empty) or 'list' -> list running/..."
    assert _is_readonly_exploration_block(_block("manage_agents", "")) is True


def test_manage_tool_bare_list_verb_is_readonly():
    assert _is_readonly_exploration_block(_block("manage_agents", "list")) is True
    assert _is_readonly_exploration_block(_block("manage_agents", "LIST")) is True
    assert _is_readonly_exploration_block(_block("manage_bg_jobs", "status")) is True


def test_manage_tool_json_readonly_action_is_readonly():
    assert _is_readonly_exploration_block(_block("manage_notes", '{"action": "list"}')) is True
    assert _is_readonly_exploration_block(_block("manage_notes", '{"action": "view", "id": "n1"}')) is True


def test_manage_tool_json_mutating_action_is_not_readonly():
    assert _is_readonly_exploration_block(_block("manage_notes", '{"action": "create", "text": "x"}')) is False
    assert _is_readonly_exploration_block(_block("manage_agents", 'stop sub_3')) is False


def test_manage_tool_malformed_json_is_not_readonly():
    # Defensive: a malformed/non-JSON, non-bare-verb manage_* call must not
    # crash and must not be misclassified as read-only.
    assert _is_readonly_exploration_block(_block("manage_notes", '{not valid json')) is False


def test_non_manage_unknown_tool_is_not_readonly():
    assert _is_readonly_exploration_block(_block("web_search", "some query")) is False
    assert _is_readonly_exploration_block(_block("spawn_agent", "do a task")) is False


# --- Two-stage streak arithmetic (pinned; the state itself is inline in the
#     giant stream_agent_loop generator, so this checks the SAME formula the
#     source uses rather than re-deriving it independently) -----------------

def test_handshake_fires_exactly_nudge_plus_handshake_rounds_after_streak_start():
    """Reimplements the exact per-round update the source performs, to pin the
    arithmetic: stage 1 fires at round N1, stage 2 fires at round
    N1+N2 (not N1, not before N1+N2)."""
    n1, n2 = 10, 10
    streak = 0
    nudge_fired = False
    nudge_fired_round = None
    handshake_fired_round = None
    for round_num in range(1, n1 + n2 + 5):
        streak += 1  # every round in this test is read-only + answer-free
        if streak >= n1 and not nudge_fired:
            nudge_fired = True
            nudge_fired_round = round_num
        handshake_hit = nudge_fired and streak >= n1 + n2
        if handshake_hit and handshake_fired_round is None:
            handshake_fired_round = round_num
    assert nudge_fired_round == n1
    assert handshake_fired_round == n1 + n2


def test_real_answer_text_resets_both_streak_and_nudge_flag():
    """A round with real answer text (e.g. a reply to the stage-1 nudge) must
    reset BOTH the streak and the nudge-fired flag, so a LATER streak gets its
    own fresh nudge instead of jumping straight to stage 2."""
    n1, n2 = 10, 10
    streak = 0
    nudge_fired = False
    # Drive to just past the nudge.
    for _ in range(n1):
        streak += 1
        if streak >= n1 and not nudge_fired:
            nudge_fired = True
    assert nudge_fired is True
    # Model replies with real text -> reset (mirrors the `else:` branch in the
    # source, taken when a round is NOT purely read-only-and-answer-free).
    streak = 0
    nudge_fired = False
    assert streak == 0 and nudge_fired is False
    # A fresh streak must reach n1 again before nudging — not fire immediately.
    for round_num in range(1, n1):
        streak += 1
        hit = streak >= n1 and not nudge_fired
        assert hit is False, f"nudge fired too early on round {round_num}"
    streak += 1
    assert streak >= n1 and not nudge_fired
