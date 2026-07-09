"""Round-cap server-side auto-continue: caps, reset, and scheduling.

The 2026-07-09 audit found rounds_exhausted only ever rendered a manual
Continue button — with no browser watching, the turn just died and users
typed "please continue" by hand. These tests pin the server-side machinery:
a capped per-session budget that resets on genuine user activity, and a
scheduler that consumes budget and fires start_server_resume_turn with the
auto-continue prompt.
"""
import asyncio

import pytest

from src import subagent_runs
from src import chat_flows


@pytest.fixture(autouse=True)
def _clean_counters():
    subagent_runs._auto_continue_count.clear()
    subagent_runs._resume_running.clear()
    yield
    subagent_runs._auto_continue_count.clear()
    subagent_runs._resume_running.clear()


def _effective_cap() -> int:
    """The live cap: settings-driven (agent_auto_continue_max), constant fallback."""
    return subagent_runs._setting_int(
        "agent_auto_continue_max", subagent_runs._MAX_AUTO_CONTINUES, 0, 50
    )


def test_auto_continue_cap_and_reset():
    sid = "sess-cap"
    assert subagent_runs.can_auto_continue(sid)
    for _ in range(_effective_cap()):
        assert subagent_runs.can_auto_continue(sid)
        subagent_runs.note_auto_continue(sid)
    # Cap consumed.
    assert not subagent_runs.can_auto_continue(sid)
    # Genuine user activity resets the budget.
    subagent_runs.note_user_activity(sid)
    assert subagent_runs.can_auto_continue(sid)


def test_auto_continue_blocked_while_resume_running():
    sid = "sess-inflight"
    subagent_runs.set_resume_running(sid, True)
    assert not subagent_runs.can_auto_continue(sid)
    subagent_runs.set_resume_running(sid, False)
    assert subagent_runs.can_auto_continue(sid)


def test_schedule_consumes_budget_and_fires_resume(monkeypatch):
    sid = "sess-fire"
    calls = []

    async def _fake_resume(session_id, framed_result=None, owner=None,
                           prompt=None, prompt_meta=None):
        calls.append({"session_id": session_id, "prompt": prompt,
                      "meta": prompt_meta, "owner": owner})

    monkeypatch.setattr(chat_flows, "start_server_resume_turn", _fake_resume)

    async def _run():
        chat_flows.maybe_schedule_auto_continue(sid, owner="alice")
        # Scheduler sleeps 1.5s before starting the turn — wait it out.
        await asyncio.sleep(1.8)

    asyncio.run(_run())

    assert len(calls) == 1
    assert calls[0]["session_id"] == sid
    assert calls[0]["owner"] == "alice"
    assert calls[0]["prompt"] == chat_flows.AUTO_CONTINUE_PROMPT
    assert calls[0]["meta"].get("hidden") is True
    # One budget slot consumed.
    assert subagent_runs._auto_continue_count.get(sid) == 1


def test_schedule_noop_when_cap_reached(monkeypatch):
    sid = "sess-capped"
    for _ in range(_effective_cap()):
        subagent_runs.note_auto_continue(sid)

    fired = []

    async def _fake_resume(*a, **k):
        fired.append(1)

    monkeypatch.setattr(chat_flows, "start_server_resume_turn", _fake_resume)

    async def _run():
        chat_flows.maybe_schedule_auto_continue(sid)
        await asyncio.sleep(1.8)

    asyncio.run(_run())
    assert not fired
    assert subagent_runs._auto_continue_count.get(sid) == _effective_cap()
