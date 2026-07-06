"""Regression: tool-call captions were never persisted, so the frontend
re-derived a caption from the SAME 120-char heuristic in two separate places
(live render vs. history reload), applied slightly differently in each,
producing a caption live that could differ from — or vanish on — reload.

Fix: agent_loop.py's round loop now computes an authoritative ``caption``
once per round (the round's plain text with <think> blocks stripped, kept
only when <=120 chars and single-paragraph) and stores it on every tool_event
dict for that round, plus forwards it on the live tool_start SSE event — so
both live and reload render from the same persisted value.

This exercises the REAL ``_strip_think_blocks`` helper (not a reimplementation)
and pins the actual eligibility predicate's source text, mirroring the
source-wiring-pin approach ``test_partial_save_tool_history.py`` uses for
logic embedded in the same giant streaming generator (stream_agent_loop isn't
practical to drive end-to-end for a single per-round computation like this).
"""
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock heavy deps so importing src.agent_loop doesn't load the full app stack —
# same pattern as tests/test_agent_loop.py / tests/test_loop_breaker_runaway.py.
# IMPORTANT: setdefault-injected stubs must be cleaned up afterward (only the
# ones THIS file actually created — a module already present, e.g. because an
# earlier-collected test imported the real thing first, must be left alone).
# Leaving a stub in sys.modules['src.agent_tools'] poisons every later test
# file's plain `from src.agent_tools import X` for the rest of the pytest
# session (a real regression this exact omission caused: see the fix in the
# same commit that added this comment).
_MOCKED = [
    'sqlalchemy', 'sqlalchemy.orm', 'sqlalchemy.ext', 'sqlalchemy.ext.declarative',
    'sqlalchemy.ext.hybrid', 'sqlalchemy.sql', 'sqlalchemy.sql.expression',
    'src.database', 'src.agent_tools', 'core.models', 'core.database',
]
_INJECTED_STUBS = {}
for _m in _MOCKED:
    if _m not in sys.modules:
        _stub = MagicMock()
        sys.modules[_m] = _stub
        _INJECTED_STUBS[_m] = _stub

try:
    from src.agent_loop import _strip_think_blocks
finally:
    for _mod, _stub in _INJECTED_STUBS.items():
        if sys.modules.get(_mod) is _stub:
            del sys.modules[_mod]
            parent_name, _, attr = _mod.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and getattr(parent, "__dict__", {}).get(attr) is _stub:
                delattr(parent, attr)

_LOOP_SRC = (Path(__file__).resolve().parents[1] / "src" / "agent_loop.py").read_text(encoding="utf-8")


def _caption_or_none(cleaned_round: str):
    """Re-apply the EXACT eligibility predicate agent_loop.py uses (pinned
    below via source-text assertions), driven through the real
    _strip_think_blocks helper — not a reimplementation of that helper."""
    text = _strip_think_blocks(cleaned_round).strip()
    if text and len(text) <= 120 and not re.search(r"\n\s*\n", text):
        return text
    return None


def test_short_single_paragraph_lead_in_is_a_caption():
    assert _caption_or_none("Let me check that file.") == "Let me check that file."


def test_think_block_is_stripped_before_evaluating_and_is_never_a_caption():
    """Thinking is not a caption, even if short — the <think> content must be
    stripped before the length/paragraph check, and pure thinking-with-no-
    visible-text yields no caption at all."""
    only_thinking = "<think>should I read the file first?</think>"
    assert _caption_or_none(only_thinking) is None

    think_then_short_reply = "<think>long internal reasoning here, doesn't count</think>Checking the config."
    assert _caption_or_none(think_then_short_reply) == "Checking the config."


def test_over_120_chars_is_not_a_caption():
    long_text = "x" * 121
    assert _caption_or_none(long_text) is None
    assert _caption_or_none("x" * 120) == "x" * 120


def test_blank_line_paragraph_break_is_not_a_caption():
    """A multi-paragraph explanation must not be treated as a short caption,
    even if the total length is under 120 chars."""
    multi_paragraph = "First I'll look.\n\nThen I'll decide what to do next."
    assert len(multi_paragraph) <= 120
    assert _caption_or_none(multi_paragraph) is None


def test_empty_text_is_not_a_caption():
    assert _caption_or_none("") is None
    assert _caption_or_none("   ") is None


def test_single_newline_without_blank_line_is_still_a_caption():
    """A single line break (not a blank-line paragraph break) must not
    disqualify an otherwise-short lead-in."""
    text = "Reading the config file\nnow."
    assert len(text) <= 120
    assert _caption_or_none(text) == text


# --------------------------------------------------------------------------- #
# Source-wiring pins: the real embedded computation in stream_agent_loop must
# match the predicate exercised above, and must actually reach both the
# persisted tool_event and the live tool_start SSE event.
# --------------------------------------------------------------------------- #

def test_source_caption_predicate_matches_the_pinned_logic():
    assert '_round_caption_text = _strip_think_blocks(cleaned_round).strip()' in _LOOP_SRC
    assert 'if _round_caption_text and len(_round_caption_text) <= 120 and not re.search(r"\\n\\s*\\n", _round_caption_text):' in _LOOP_SRC


def test_source_caption_is_attached_to_persisted_tool_event():
    assert 'if _round_caption:\n                tool_event["caption"] = _round_caption' in _LOOP_SRC


def test_source_caption_is_forwarded_on_live_tool_start_event():
    assert '_tool_start_evt["caption"] = _round_caption' in _LOOP_SRC
