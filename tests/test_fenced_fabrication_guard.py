"""Fenced-mode fabrication guard (FIX A/B/C).

Regression coverage for the ornith:9b failure: a small local model in FENCED
tool mode emitted ```bash fences full of interleaved prose / invalid syntax so
`parse_tool_blocks` returned ZERO runnable blocks, then FABRICATED the tool
output as prose ("Status check returned 167 entities"). With 0 parsed blocks
the agent loop accepted that as a final answer and the repeat-based
loop-breakers never fired.

These tests exercise the detection primitive (`_has_unparsed_tool_fence`) and
the cap constant. The full loop is a large async generator, so the counter /
reset SEMANTICS are asserted in isolation (a small model of the loop's
book-keeping) rather than by driving the whole generator — see
test_counter_cap_and_reset_semantics for the note on that tradeoff.
"""

import pytest

from src.agent_tools import parse_tool_blocks
from src.agent_loop import (
    _has_unparsed_tool_fence,
    _MAX_MALFORMED_FENCE_ROUNDS,
    _AGENT_RULES,
)


# ── _has_unparsed_tool_fence: POSITIVE cases (tool intent, 0 parsed blocks) ──

@pytest.mark.parametrize("text", [
    # The observed shape: ```bash with inline garbled command after the tag
    # (invalid `grep -iEk`), which the code-fence classifier refuses to run.
    "Checking status.\n\n```bash grep -iEk entities status\nmore prose here\n```\n\nStatus check returned 167 entities.",
    # Empty ```bash fence — nothing to run, 0 blocks, but a clear tool attempt.
    "Let me look.\n\n```bash\n```\n\nThe kitchen light is now ON.",
    # A `json` tool-channel fence whose body isn't valid JSON (json is a
    # recognized tool-intent alias, not a TOOL_TAG, so covered via extras).
    "Calling the tool.\n\n```json\nnot valid json, just narrative text\n```",
    # A `sh` alias fence with inline command (alias for a shell tool intent).
    "Running.\n\n```sh ls -la /etc && cat foo\n```",
])
def test_has_unparsed_tool_fence_true_for_garbled_tool_intent(text):
    # Precondition for the guard: parse really did drop it to zero blocks.
    assert parse_tool_blocks(text, skip_fenced=False) == []
    assert _has_unparsed_tool_fence(text) is True


# ── _has_unparsed_tool_fence: NEGATIVE cases (no false positives) ──

def test_clean_parseable_fence_is_not_flagged():
    text = "Running it now.\n\n```bash\nls -la /tmp\n```"
    # This DOES parse into a runnable block, so it is not a dropped attempt.
    assert len(parse_tool_blocks(text, skip_fenced=False)) == 1
    assert _has_unparsed_tool_fence(text) is False


def test_plain_prose_answer_is_not_flagged():
    text = "The answer is 42. Here is my reasoning, in prose, with no code at all."
    assert _has_unparsed_tool_fence(text) is False


def test_no_fence_at_all_is_not_flagged():
    assert _has_unparsed_tool_fence("Done. Everything looks good.") is False
    assert _has_unparsed_tool_fence("") is False


def test_illustrative_non_tool_language_fence_is_not_flagged():
    # A ```js / ```yaml example for the USER is a legitimate final answer; its
    # tag is not a tool intent, so it must not be read as a dropped call.
    js = "Here's how you'd do it:\n\n```js\nconsole.log('hi')\n```"
    assert parse_tool_blocks(js, skip_fenced=False) == []
    assert _has_unparsed_tool_fence(js) is False

    yaml = "Example config:\n\n```yaml\nkey: value\n```"
    assert _has_unparsed_tool_fence(yaml) is False


def test_tool_intent_fence_inside_think_is_ignored():
    # Reasoning scratch is not an executed tool attempt.
    text = "<think>\n```bash foo bar baz\n```\n</think>\nFinal answer for the user."
    assert _has_unparsed_tool_fence(text) is False


# ── Cap constant (FIX B) ──

def test_cap_constant_is_small_and_positive():
    # The corrector must fire at least once but be bounded so it can't loop.
    assert isinstance(_MAX_MALFORMED_FENCE_ROUNDS, int)
    assert 1 <= _MAX_MALFORMED_FENCE_ROUNDS <= 3


# ── Anti-fabrication rule (FIX C) ──

def test_agent_rules_contain_anti_fabrication_line():
    assert "NEVER write tool output yourself" in _AGENT_RULES


# ── Counter / reset semantics (FIX B), modeled in isolation ──

def test_counter_cap_and_reset_semantics():
    """The loop is a ~1000-line async generator; wiring a full end-to-end run
    would need heavy mocking of the LLM stream, tool executor, and SSE plumbing.
    Instead we model the exact book-keeping the loop performs around
    `_has_unparsed_tool_fence` and assert the invariant the fix relies on:

      - each consecutive malformed-fence round increments the counter;
      - reaching _MAX_MALFORMED_FENCE_ROUNDS flips force-answer;
      - a real parsed call OR a clean fence-free answer resets the counter to 0.
    """
    garbled = "```bash grep -iEk foo\n```"        # 0 blocks, tool intent
    clean_answer = "All done, here is the summary."  # 0 blocks, no tool fence
    real_call = "```bash\nls -la\n```"              # 1 real parsed block

    def step(counter, round_response):
        """Return (new_counter, forced) mirroring the loop's handling."""
        blocks = parse_tool_blocks(round_response, skip_fenced=False)
        if blocks:
            # A real call resets the streak (loop resets before executing).
            return 0, False
        # 0 blocks branch (the "no tools — done" path in the loop).
        if _has_unparsed_tool_fence(round_response):
            counter += 1
            forced = counter >= _MAX_MALFORMED_FENCE_ROUNDS
            return counter, forced
        # Clean fence-free final answer resets the streak.
        return 0, False

    # Two consecutive garbled rounds -> force-answer on the 2nd.
    c, forced = step(0, garbled)
    assert c == 1 and forced is False
    c, forced = step(c, garbled)
    assert c == _MAX_MALFORMED_FENCE_ROUNDS and forced is True

    # A real parsed call resets the streak.
    c, forced = step(1, real_call)
    assert c == 0 and forced is False

    # A clean fence-free answer resets the streak.
    c, forced = step(1, clean_answer)
    assert c == 0 and forced is False
