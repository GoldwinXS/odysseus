"""Phase-1 async sub-agent flow: spawn_agent returns immediately (non-blocking)
and delivers the sub-agent's result into the parent session when it finishes."""
import asyncio
import json

import pytest

import src.agent_loop as agent_loop
import src.ai_interaction as ai_interaction
import src.subagent_runs as subagent_runs
from src.agent_tools import model_interaction_tools as mit


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


async def _wait_done(session_id, timeout=5.0):
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        upd = subagent_runs.get_updates(session_id)
        if upd["updates"] and all(u["status"] != "running" for u in upd["updates"]):
            return upd
        await asyncio.sleep(0.02)
    raise AssertionError("sub-agent did not finish in time")


@pytest.fixture
def fake_env(monkeypatch):
    sess = _FakeSession()
    monkeypatch.setattr(ai_interaction, "get_session_manager", lambda: _FakeSM(sess))
    return sess


async def test_spawn_agent_returns_immediately_and_delivers(fake_env, monkeypatch):
    async def fake_loop(*args, **kwargs):
        yield _sse({"delta": "Hello "})
        yield _sse({"delta": "world"})
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_agent_loop", fake_loop)

    ret = await mit.spawn_agent("do a thing", session_id="sess-1", owner="u")
    # Immediate, non-blocking ack — NOT the final result.
    assert ret.get("background") is True
    assert ret.get("subagent_id")
    assert "world" not in ret.get("result", "")  # result not inlined into the ack

    upd = await _wait_done("sess-1")
    assert upd["updates"][0]["status"] == "done"

    # Result delivered into the parent session as an assistant message.
    assert len(fake_env.messages) == 1
    msg = fake_env.messages[0]
    assert msg.role == "assistant"
    assert "Hello world" in msg.content
    assert (msg.metadata or {}).get("subagent") is True


async def test_spawn_agent_delivers_error_on_failure(fake_env, monkeypatch):
    async def boom_loop(*args, **kwargs):
        raise RuntimeError("kaboom")
        yield  # pragma: no cover — make it an async generator

    monkeypatch.setattr(agent_loop, "stream_agent_loop", boom_loop)

    ret = await mit.spawn_agent("do a thing", session_id="sess-2", owner="u")
    assert ret.get("background") is True

    upd = await _wait_done("sess-2")
    assert upd["updates"][0]["status"] == "error"
    assert len(fake_env.messages) == 1
    assert "Sub-agent failed" in fake_env.messages[0].content


async def test_total_cap_rejects_when_full(fake_env, monkeypatch):
    # Fill the tracker up to the cap with never-finishing tasks.
    hold = asyncio.Event()

    async def hang_loop(*args, **kwargs):
        await hold.wait()
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_agent_loop", hang_loop)
    try:
        started = []
        for i in range(subagent_runs._MAX_TOTAL):
            r = await mit.spawn_agent("x", session_id=f"cap-{i}", owner="u")
            started.append(r)
            assert r.get("background") is True
        # One more must be rejected.
        rejected = await mit.spawn_agent("x", session_id="cap-extra", owner="u")
        assert "error" in rejected
        assert rejected.get("background") is not True
    finally:
        hold.set()
        # Let the hung tasks drain so they don't leak into other tests.
        await asyncio.sleep(0.05)


async def test_stop_cancels_running_subagent_and_delivers_notice(fake_env, monkeypatch):
    hold = asyncio.Event()

    async def hang_loop(*args, **kwargs):
        yield _sse({"delta": "partial "})
        await hold.wait()   # never completes until cancelled
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_agent_loop", hang_loop)
    try:
        ret = await mit.spawn_agent("do a thing", session_id="stop-1", owner="u")
        sub_id = ret["subagent_id"]
        await asyncio.sleep(0.05)   # let it start and stream the partial
        assert subagent_runs.stop("stop-1", sub_id) is True

        upd = await _wait_done("stop-1")
        assert upd["updates"][0]["status"] == "error"
        # Cancellation delivered a notice (with the partial) into the session.
        assert len(fake_env.messages) == 1
        assert "cancelled" in fake_env.messages[0].content.lower()
    finally:
        hold.set()

    # Stopping an unknown id is a no-op, not an error.
    assert subagent_runs.stop("stop-1", "nope") is False


async def test_empty_result_delivers_informative_fallback(fake_env, monkeypatch):
    # A sub-agent that only makes tool calls and hits the round cap (no final text)
    # must deliver an INFORMATIVE note, not the bare "(no text output)".
    async def toolonly_loop(*args, **kwargs):
        yield _sse({"type": "tool_start", "tool": "read_file"})
        yield _sse({"type": "tool_output", "tool": "read_file"})
        yield _sse({"type": "tool_start", "tool": "grep"})
        yield _sse({"type": "rounds_exhausted"})
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_agent_loop", toolonly_loop)
    await mit.spawn_agent("do a thing", session_id="empty-1", owner="u")
    upd = await _wait_done("empty-1")
    assert upd["updates"][0]["status"] == "done"
    assert len(fake_env.messages) == 1
    content = fake_env.messages[0].content.lower()
    assert "no text output" not in content
    assert "round" in content and "tool call" in content  # names tools + round limit


async def test_updates_payload_has_server_now(fake_env, monkeypatch):
    upd = subagent_runs.get_updates("whatever")
    assert isinstance(upd.get("now"), float)


async def test_no_session_runs_synchronously(monkeypatch):
    async def fake_loop(*args, **kwargs):
        yield _sse({"delta": "inline"})
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_agent_loop", fake_loop)
    # No parent session and an explicit model override so resolution succeeds
    # without a session to inherit from.
    monkeypatch.setattr(
        ai_interaction, "_resolve_model",
        lambda spec, owner=None: ("http://x/v1", "m", {}),
    )
    ret = await mit.spawn_agent("model: m\ndo a thing", session_id=None, owner="u")
    # Synchronous fallback returns the result directly, no background ack.
    assert ret.get("background") is not True
    assert "inline" in ret.get("result", "")
