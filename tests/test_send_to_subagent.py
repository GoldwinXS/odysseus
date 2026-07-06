"""Mid-flight sub-agent steering: send_to_subagent enqueues guidance onto a
RUNNING sub-agent's ephemeral steer queue, and the graceful wind-down steer the
watchdog sends before a hard-kill.

A background sub-agent runs stream_agent_loop with an ephemeral queue session
(``_subagent_<id>``); subagent_runs.start() stands up a bare steer mailbox in
agent_runs._RUNS[queue_session] so enqueue_steer/drain_steering work for it.
send_to_subagent validates the target (running, in this chat) and enqueues a
framed message that the sub-agent's own loop drains at a round boundary.

These bypass the real agent loop (stream_agent_loop is monkeypatched) and assert
on the queue directly — the drain half is covered by test_chat_steering.py.
"""
import asyncio
import json

import pytest

import src.agent_loop as agent_loop
import src.agent_runs as agent_runs
import src.ai_interaction as ai_interaction
import src.subagent_runs as subagent_runs
from src.agent_tools import model_interaction_tools as mit
from src.agent_tools.interaction_tools import SendToSubagentTool


class _FakeSession:
    def __init__(self):
        self.endpoint_url = "http://x/v1"
        self.model = "test-model"
        self.headers = {}
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)


class _FakeSM:
    def __init__(self, session):
        self._session = session

    def get_session(self, sid):
        return self._session


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


@pytest.fixture
def fake_env(monkeypatch):
    sess = _FakeSession()
    monkeypatch.setattr(ai_interaction, "get_session_manager", lambda: _FakeSM(sess))
    return sess


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    yield
    agent_runs._RUNS.clear()
    for d in (subagent_runs._resume_count, subagent_runs._resume_running,
              subagent_runs._failure_resume_count):
        d.clear()
    subagent_runs._UPDATES.clear()


async def _spawn_hanging(session_id, hold, monkeypatch):
    async def hang_loop(*args, **kwargs):
        yield _sse({"delta": "working"})
        await hold.wait()
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_agent_loop", hang_loop)
    ret = await mit.spawn_agent("do a thing", session_id=session_id, owner="u")
    await asyncio.sleep(0.05)   # let it register + start
    return ret


# ── send_to_subagent: enqueue reaches the sub-agent's queue ──────────────

async def test_send_reaches_running_subagent_queue(fake_env, monkeypatch):
    hold = asyncio.Event()
    try:
        ret = await _spawn_hanging("st-1", hold, monkeypatch)
        sub_id = ret["subagent_id"]

        # The sub-agent got an ephemeral steer-queue session AND a live mailbox.
        rec = subagent_runs.find_running("st-1", sub_id)
        assert rec is not None
        queue_session = rec["queue_session"]
        assert queue_session == f"{subagent_runs._QUEUE_SESSION_PREFIX}{sub_id}"
        assert agent_runs.is_active(queue_session) is True   # mailbox is live

        desc, result = await SendToSubagentTool().execute(
            json.dumps({"subagent_id": sub_id, "message": "focus on the parser"}),
            {"session_id": "st-1", "owner": "u"},
        )
        assert result["exit_code"] == 0 and result["subagent_id"] == sub_id

        # The framed guidance landed on the sub-agent's own queue.
        drained = agent_runs.drain_steering(queue_session)
        assert len(drained) == 1
        assert drained[0]["kind"] == "user"
        assert "focus on the parser" in drained[0]["text"]
        assert "dispatched you" in drained[0]["text"].lower()   # framed as guidance
    finally:
        hold.set()
        await asyncio.sleep(0.05)


async def test_send_unknown_id_fails_cleanly(fake_env, monkeypatch):
    hold = asyncio.Event()
    try:
        await _spawn_hanging("st-2", hold, monkeypatch)
        desc, result = await SendToSubagentTool().execute(
            json.dumps({"subagent_id": "sub_999", "message": "hi"}),
            {"session_id": "st-2", "owner": "u"},
        )
        assert result["exit_code"] == 1
        assert "no running sub-agent" in result["error"].lower()
        assert "manage_agents" in result["error"]
    finally:
        hold.set()
        await asyncio.sleep(0.05)


async def test_send_to_finished_subagent_fails(fake_env, monkeypatch):
    # A sub-agent that has already finished is not steerable — clean error, and
    # its mailbox has been torn down.
    async def quick_loop(*args, **kwargs):
        yield _sse({"delta": "done fast"})
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_agent_loop", quick_loop)
    ret = await mit.spawn_agent("do a thing", session_id="st-3", owner="u")
    sub_id = ret["subagent_id"]
    # Wait for it to finish.
    for _ in range(100):
        if subagent_runs.find_running("st-3", sub_id) is None:
            break
        await asyncio.sleep(0.02)

    desc, result = await SendToSubagentTool().execute(
        json.dumps({"subagent_id": sub_id, "message": "too late"}),
        {"session_id": "st-3", "owner": "u"},
    )
    assert result["exit_code"] == 1
    # Mailbox torn down on finish.
    assert agent_runs.is_active(f"{subagent_runs._QUEUE_SESSION_PREFIX}{sub_id}") is False


async def test_send_validates_missing_fields(fake_env, monkeypatch):
    hold = asyncio.Event()
    try:
        ret = await _spawn_hanging("st-4", hold, monkeypatch)
        sub_id = ret["subagent_id"]
        tool = SendToSubagentTool()
        # Missing id.
        _, r1 = await tool.execute(json.dumps({"message": "x"}), {"session_id": "st-4"})
        assert r1["exit_code"] == 1 and "subagent_id" in r1["error"]
        # Missing message.
        _, r2 = await tool.execute(json.dumps({"subagent_id": sub_id}), {"session_id": "st-4"})
        assert r2["exit_code"] == 1 and "message" in r2["error"]
        # No session at all.
        _, r3 = await tool.execute(json.dumps({"subagent_id": sub_id, "message": "x"}), {})
        assert r3["exit_code"] == 1 and "chat session" in r3["error"]
    finally:
        hold.set()
        await asyncio.sleep(0.05)


# ── registration: reachable through the native-function converter ────────

def test_send_to_subagent_registered():
    from src.agent_tools import TOOL_HANDLERS, TOOL_TAGS
    assert "send_to_subagent" in TOOL_HANDLERS
    assert "send_to_subagent" in TOOL_TAGS


def test_native_convert_normalises_args():
    from src.tool_schemas import function_call_to_tool_block
    block = function_call_to_tool_block(
        "send_to_subagent", '{"subagent_id": "sub_3", "message": "narrow it"}'
    )
    assert block is not None and block.tool_type == "send_to_subagent"
    payload = json.loads(block.content)
    assert payload == {"subagent_id": "sub_3", "message": "narrow it"}
    # Alias keys (id/text) are accepted too.
    block2 = function_call_to_tool_block(
        "send_to_subagent", '{"id": "sub_4", "text": "also check X"}'
    )
    payload2 = json.loads(block2.content)
    assert payload2 == {"subagent_id": "sub_4", "message": "also check X"}


def test_send_to_subagent_always_available_and_indexed():
    from src.tool_index import ALWAYS_AVAILABLE, BUILTIN_TOOL_DESCRIPTIONS
    assert "send_to_subagent" in ALWAYS_AVAILABLE
    assert "send_to_subagent" in BUILTIN_TOOL_DESCRIPTIONS
    assert len(BUILTIN_TOOL_DESCRIPTIONS["send_to_subagent"]) > 50


# ── graceful wrap-up before kill (GOAL 2) ────────────────────────────────

def test_wrapup_grace_windows():
    # Stall kill uses the shorter window; wall/loop use the full base window.
    assert mit._subagent_wrapup_grace("wall") == mit._SUBAGENT_WRAPUP_GRACE_S
    assert mit._subagent_wrapup_grace("loop") == mit._SUBAGENT_WRAPUP_GRACE_S
    assert mit._subagent_wrapup_grace("stall") == min(
        mit._SUBAGENT_WRAPUP_GRACE_S, mit._SUBAGENT_WRAPUP_STALL_GRACE_S
    )


def test_send_wrapup_steer_enqueues_into_live_mailbox():
    # The wind-down steer lands on a live mailbox; it fails (False) when there
    # is no live mailbox. For "wall"/"stall" (the two reasons the watchdog's
    # grace-window kill can actually be CANCELLED by resumed activity) the
    # message is a check-in — not an unconditional stop order — since the
    # agent may legitimately still be working: it must invite the worker to
    # continue if it's genuinely still making progress, while still asking
    # for a final summary if it's actually done/stuck.
    q = "_subagent_wrap1"
    agent_runs.register_steer_mailbox(q)
    try:
        ok = mit._send_wrapup_steer(q, "wall", 180, 3600)
        assert ok is True
        drained = agent_runs.drain_steering(q)
        assert len(drained) == 1
        txt = drained[0]["text"].lower()
        assert "check-in" in txt
        assert "continue normally" in txt  # invites continuing, not a stop order
        assert "summary" in txt
    finally:
        agent_runs.close_steer_mailbox(q)
    # No mailbox → cannot steer.
    assert mit._send_wrapup_steer("_subagent_none", "stall", 180, 3600) is False


def test_send_wrapup_steer_stall_reason_is_also_a_check_in():
    q = "_subagent_wrap2"
    agent_runs.register_steer_mailbox(q)
    try:
        ok = mit._send_wrapup_steer(q, "stall", 180, 3600)
        assert ok is True
        drained = agent_runs.drain_steering(q)
        txt = drained[0]["text"].lower()
        assert "check-in" in txt
        assert "continue normally" in txt
        assert "summary" in txt
    finally:
        agent_runs.close_steer_mailbox(q)


def test_send_wrapup_steer_loop_reason_stays_an_unconditional_stop_order():
    # A LOOP trigger fires from inside the drain on a CONFIRMED stuck pattern
    # (the same exact tool call repeated) — unlike stall/wall it is not
    # cancellable by new activity, so its message stays an unconditional stop.
    q = "_subagent_wrap3"
    agent_runs.register_steer_mailbox(q)
    try:
        ok = mit._send_wrapup_steer(q, "loop", 180, 3600)
        assert ok is True
        drained = agent_runs.drain_steering(q)
        txt = drained[0]["text"].lower()
        assert "immediately stop" in txt
        assert "summary" in txt
    finally:
        agent_runs.close_steer_mailbox(q)


async def test_stall_kill_attempts_wrapup_then_kills(fake_env, monkeypatch):
    # A stalling sub-agent (streams once, then goes silent) is first STEERED to
    # wrap up, then hard-killed after the short grace when it stays silent. The
    # delivered failure notes the wind-down was attempted.
    # Trip the stall guard fast and give a 1s grace, robustly (bypass settings).
    monkeypatch.setattr(mit, "_subagent_stall_timeout", lambda: 1)
    monkeypatch.setattr(mit, "_subagent_wrapup_grace", lambda reason: 1)

    async def stall_loop(*args, **kwargs):
        yield _sse({"delta": "starting"})
        await asyncio.sleep(30)   # go silent → stall watchdog fires
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_agent_loop", stall_loop)

    await mit.spawn_agent("do a thing", session_id="wrap-stall", owner="u")
    # Wait for the run to finish (stall + grace + kill).
    for _ in range(300):
        upd = subagent_runs.get_updates("wrap-stall")
        if upd["updates"] and all(u["status"] != "running" for u in upd["updates"]):
            break
        await asyncio.sleep(0.05)
    upd = subagent_runs.get_updates("wrap-stall")
    assert upd["updates"][0]["status"] == "error"
    content = fake_env.messages[0].content.lower()
    assert "stalled" in content
    # The wind-down was attempted (worker stayed silent → noted as such).
    assert "wind-down" in content or "did not respond" in content
