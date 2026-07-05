"""FIX B — revive dead plan/TODO tracking.

Server-side pieces:
  1. The base rule sets (_AGENT_RULES / _API_AGENT_RULES, the '## Base rules'
     variants) instruct the model to call `update_plan` for multi-step work.
  2. `build_active_plan_note` pins an approved plan into the system context; it
     fires (returns a non-empty note) when a plan is passed and stays empty
     otherwise. This is what the re-injected `approved_plan` (sent by the
     client each turn) feeds.

The client store/render/re-inject lives in static/js/chat.js; `_setStoredPlan`
is now defined (previously an undefined ReferenceError) and `approved_plan` is
appended to the chat request body. Those are verified by `node --check` on
chat.js plus a code-trace (see the module-level docstring in the report), since
the plan helpers are private to the chat module and not cleanly importable via
the node ESM harness without its full import graph.
"""
from src import agent_loop


# ── Piece 3: the base-rule prompt line ─────────────────────────────────────
def test_base_rules_mention_update_plan():
    # Both active base rule sets ('## Base rules' variants) must tell the model
    # to use update_plan for multi-step work.
    assert "## Base rules" in agent_loop._AGENT_RULES
    assert "## Base rules" in agent_loop._API_AGENT_RULES
    for rules in (agent_loop._AGENT_RULES, agent_loop._API_AGENT_RULES):
        assert "update_plan" in rules
        assert "multi-step" in rules


# ── Piece 2: build_active_plan_note pins an approved plan ───────────────────
def test_build_active_plan_note_fires_when_plan_passed():
    plan = "- [x] step one\n- [ ] step two\n- [ ] step three"
    note = agent_loop.build_active_plan_note(plan)
    assert note  # non-empty
    assert "ACTIVE PLAN" in note
    assert "step two" in note  # the actual plan text is embedded
    assert plan.strip() in note


def test_build_active_plan_note_empty_when_no_plan():
    assert agent_loop.build_active_plan_note("") == ""
    assert agent_loop.build_active_plan_note("   ") == ""
    assert agent_loop.build_active_plan_note(None) == ""
