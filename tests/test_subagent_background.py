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
        # Spread across owners so the per-owner fairness cap doesn't trip first —
        # this test exercises the GLOBAL _MAX_TOTAL backstop specifically.
        for i in range(subagent_runs._MAX_TOTAL):
            r = await mit.spawn_agent("x", session_id=f"cap-{i}", owner=f"u{i}")
            started.append(r)
            assert r.get("background") is True
        # One more must be rejected (global cap), even for a brand-new owner.
        rejected = await mit.spawn_agent("x", session_id="cap-extra", owner="u-extra")
        assert "error" in rejected
        assert rejected.get("background") is not True
    finally:
        hold.set()
        # Let the hung tasks drain so they don't leak into other tests.
        await asyncio.sleep(0.05)


async def test_per_owner_cap_rejects_before_global(fake_env, monkeypatch):
    # One owner may hold at most _MAX_PER_OWNER outstanding sub-agents, even
    # though the global cap is higher — so a busy user can't starve others.
    hold = asyncio.Event()

    async def hang_loop(*args, **kwargs):
        await hold.wait()
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_agent_loop", hang_loop)
    try:
        for i in range(subagent_runs._MAX_PER_OWNER):
            r = await mit.spawn_agent("x", session_id=f"owner-{i}", owner="busy")
            assert r.get("background") is True
        # The next one for the SAME owner is rejected on the per-owner cap...
        rejected = await mit.spawn_agent("x", session_id="owner-x", owner="busy")
        assert "error" in rejected and rejected.get("background") is not True
        # ...but a DIFFERENT owner can still start (global pool not yet full).
        other = await mit.spawn_agent("x", session_id="other-1", owner="fresh")
        assert other.get("background") is True
    finally:
        hold.set()
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


async def test_upstream_error_surfaced_not_empty(fake_env, monkeypatch):
    # A 429 / spend-cap failure must be reported as the REAL reason, not the
    # generic "empty response" placeholder the loop emits after an error.
    async def err_loop(*args, **kwargs):
        yield "event: error\ndata: " + json.dumps({"status": 429, "text": "Google rate-limited the request (429)."}) + "\n\n"
        yield _sse({"delta": "The model returned an empty response. Please try again or switch to a different model."})
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_agent_loop", err_loop)
    await mit.spawn_agent("do a thing", session_id="err-1", owner="u")
    upd = await _wait_done("err-1")
    assert upd["updates"][0]["status"] == "error"
    content = fake_env.messages[0].content.lower()
    assert "rate-limited" in content or "429" in content
    assert "empty response" not in content
    assert "failed" in content


async def test_updates_payload_has_server_now(fake_env, monkeypatch):
    upd = subagent_runs.get_updates("whatever")
    assert isinstance(upd.get("now"), float)


async def test_manage_agents_list_and_stop(fake_env, monkeypatch):
    hold = asyncio.Event()

    async def hang_loop(*a, **k):
        yield _sse({"delta": "working"})
        await hold.wait()
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_agent_loop", hang_loop)
    try:
        ret = await mit.spawn_agent("analyze the thing", session_id="mg-1", owner="u")
        sub_id = ret["subagent_id"]
        await asyncio.sleep(0.05)
        listed = await mit.manage_agents("list", session_id="mg-1", owner="u")
        assert sub_id in listed["results"] and "running" in listed["results"]
        stopped = await mit.manage_agents(f"stop {sub_id}", session_id="mg-1", owner="u")
        assert "ancel" in stopped["results"]           # "Cancelling ..."
        gone = await mit.manage_agents("stop nope", session_id="mg-1", owner="u")
        assert "No running sub-agent" in gone["results"]
    finally:
        hold.set()
        await asyncio.sleep(0.05)


async def test_context_note_none_when_no_subagents(fake_env, monkeypatch):
    # Ordinary turns (no sub-agents ever dispatched) get no injected note.
    assert subagent_runs.context_note("pristine-session") is None


async def test_context_note_gives_ground_truth_and_survives_denial(fake_env, monkeypatch):
    # A still-running sub-agent must produce a note that (a) asserts one WAS
    # dispatched, (b) names the running id + task, and (c) points at the
    # manage_agents tool — this is what stops the model denying it dispatched.
    hold = asyncio.Event()

    async def hang_loop(*a, **k):
        yield _sse({"delta": "working"})
        await hold.wait()
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_agent_loop", hang_loop)
    try:
        ret = await mit.spawn_agent(
            "analyze js/main.js for tech debt", session_id="note-1", owner="u"
        )
        sub_id = ret["subagent_id"]
        await asyncio.sleep(0.05)
        note = subagent_runs.context_note("note-1")
        assert note is not None
        assert sub_id in note and "RUNNING" in note
        assert "js/main.js" in note
        assert "manage_agents" in note
        assert "never dispatched" in note.lower() or "do not claim" in note.lower()
    finally:
        hold.set()
        await asyncio.sleep(0.05)

    # After it finishes, the note flips to reporting it as delivered — not running.
    await _wait_done("note-1")
    done_note = subagent_runs.context_note("note-1")
    assert done_note is not None and "RUNNING" not in done_note
    assert "delivered into this chat" in done_note


async def test_subagent_gets_coding_tool_baseline(fake_env, monkeypatch):
    # A task that never lexically mentions files must STILL be dispatched with the
    # file/shell/search tools — otherwise the worker reports itself "blocked, no
    # filesystem tools" (the reported bug). Also: recursion tools stay stripped.
    captured = {}

    async def capture_loop(*args, **kwargs):
        captured.update(kwargs)
        yield _sse({"delta": "done"})
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_agent_loop", capture_loop)
    await mit.spawn_agent(
        "improve the look of the buildings so they feel more polished",
        session_id="tools-1", owner="u",
    )
    await _wait_done("tools-1")
    rt = captured.get("relevant_tools") or set()
    for t in ("read_file", "write_file", "edit_file", "bash", "ls", "grep", "get_workspace"):
        assert t in rt, f"{t} missing from sub-agent tool set: {sorted(rt)}"
    # Leaf worker: recursion/orchestration tools must never be present.
    assert "spawn_agent" not in rt and "manage_agents" not in rt


async def test_parent_disabled_tools_inherited_by_child(fake_env, monkeypatch):
    # A child must NOT regain a tool the parent turn had disabled: if the parent
    # turn ran with bash disabled, the spawned sub-agent must be dispatched with
    # bash both disabled AND stripped from its selectable tool set.
    captured = {}

    async def capture_loop(*args, **kwargs):
        captured.update(kwargs)
        yield _sse({"delta": "done"})
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_agent_loop", capture_loop)
    await mit.spawn_agent(
        "read a file and run a shell command",
        session_id="inherit-1", owner="u",
        parent_disabled={"bash"},
    )
    await _wait_done("inherit-1")
    # bash flows into the child's disabled set...
    assert "bash" in (captured.get("disabled_tools") or set())
    # ...and is stripped from its selectable tools even though it's in the baseline.
    assert "bash" not in (captured.get("relevant_tools") or set())
    # Other baseline tools the parent did NOT disable are still available.
    assert "read_file" in (captured.get("relevant_tools") or set())


async def test_spawn_agent_ctx_threads_disabled_tools(fake_env, monkeypatch):
    # The registry wrapper must forward the parent turn's disabled_tools from ctx.
    captured = {}

    async def capture_loop(*args, **kwargs):
        captured.update(kwargs)
        yield _sse({"delta": "done"})
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_agent_loop", capture_loop)
    tool = mit.SpawnAgentTool()
    await tool.execute(
        "do a thing",
        {"session_id": "ctx-1", "owner": "u", "disabled_tools": {"write_file"}},
    )
    await _wait_done("ctx-1")
    assert "write_file" in (captured.get("disabled_tools") or set())
    assert "write_file" not in (captured.get("relevant_tools") or set())


async def test_ask_user_excluded_from_subagent(fake_env, monkeypatch):
    # A background sub-agent has no interactive channel, so ask_user/update_plan
    # must be disabled and never dispatched.
    captured = {}

    async def capture_loop(*args, **kwargs):
        captured.update(kwargs)
        yield _sse({"delta": "done"})
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_agent_loop", capture_loop)
    await mit.spawn_agent(
        "ask the user what they want and update the plan",
        session_id="asku-1", owner="u",
    )
    await _wait_done("asku-1")
    assert "ask_user" in mit._SUBAGENT_DISABLED
    assert "update_plan" in mit._SUBAGENT_DISABLED
    assert "ask_user" in (captured.get("disabled_tools") or set())
    rt = captured.get("relevant_tools") or set()
    assert "ask_user" not in rt and "update_plan" not in rt


def test_ack_first_caller_wins():
    # ack() transitions a finished run once: the first caller gets the id, a
    # second call for the same id gets nothing (so a reload won't double-fire).
    sid = "ack-unit-1"

    def _rec(rid, status):
        return {"id": rid, "status": status, "summary": "", "model": "", "owner": None,
                "started_at": 0.0, "finished_at": 0.0, "error": None, "acked": False}

    subagent_runs._UPDATES[sid] = [_rec("sub_a", "done"), _rec("sub_b", "running")]
    try:
        first = subagent_runs.ack(sid, ["sub_a", "sub_b"])
        assert first == ["sub_a"]                 # only the finished one
        second = subagent_runs.ack(sid, ["sub_a"])
        assert second == []                        # already acked — first caller won
        # Exposed in the poll payload.
        upd = subagent_runs.get_updates(sid)
        rec = next(u for u in upd["updates"] if u["id"] == "sub_a")
        assert rec["acked"] is True
    finally:
        subagent_runs._UPDATES.pop(sid, None)


def test_ack_endpoint_owner_verified_and_first_caller_wins(monkeypatch):
    # End-to-end through the HTTP route: owner mismatch is rejected; the first
    # ack of an id wins and already-acked ids are omitted.
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient
    from routes import chat_routes

    # Owner-gate: allow only session "mine".
    def _verify(request, session):
        if session != "mine":
            raise HTTPException(403, "forbidden")

    monkeypatch.setattr(chat_routes, "_verify_session_owner", _verify)

    subagent_runs._UPDATES["mine"] = [
        {"id": "sub_1", "status": "done", "acked": False},
    ]

    # Route bodies reference their deps only at call time; the ack route uses
    # none of them, so building the router with None deps is enough to register
    # and exercise /api/subagent/ack in isolation.
    router = chat_routes.setup_chat_routes(
        session_manager=None,
        chat_handler=None,
        chat_processor=None,
        memory_manager=None,
        research_handler=None,
        upload_handler=None,
    )
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    try:
        # Wrong owner → 403.
        r = client.post("/api/subagent/ack", json={"session_id": "theirs", "run_ids": ["sub_1"]})
        assert r.status_code == 403
        # Right owner, first call wins.
        r = client.post("/api/subagent/ack", json={"session_id": "mine", "run_ids": ["sub_1"]})
        assert r.status_code == 200 and r.json() == {"acked": ["sub_1"]}
        # Second call: already acked → omitted.
        r = client.post("/api/subagent/ack", json={"session_id": "mine", "run_ids": ["sub_1"]})
        assert r.status_code == 200 and r.json() == {"acked": []}
        # Bad body → 400.
        r = client.post("/api/subagent/ack", json={"session_id": "mine"})
        assert r.status_code == 400
    finally:
        subagent_runs._UPDATES.pop("mine", None)


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
