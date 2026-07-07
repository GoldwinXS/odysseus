"""Steering queue + endpoint, and the loop's round-boundary / final-drain
injection. A message sent while a turn streams is injected into the RUNNING
agent loop at the next round boundary (Claude-Code steering), or 409s when the
session is idle so the client falls back to a normal send — never lost."""
import asyncio
import json

import pytest

import src.agent_runs as agent_runs
import src.subagent_runs as subagent_runs


# ── agent_runs steering queue primitives ─────────────────────────────────

def _make_live_run(session_id: str) -> agent_runs._Run:
    """Register a bare running _Run (no drain task) for queue-level tests."""
    run = agent_runs._Run()
    run.status = "running"
    agent_runs._RUNS[session_id] = run
    return run


@pytest.fixture(autouse=True)
def _clean_runs():
    yield
    agent_runs._RUNS.clear()
    subagent_runs._resume_count.clear()
    subagent_runs._resume_running.clear()


def test_enqueue_steer_requires_live_run():
    # No run at all → refused (endpoint 409s → client sends normally).
    assert agent_runs.enqueue_steer("idle-sess", "hello") is False
    # A terminal run also refuses — the turn is over, nothing to steer.
    run = _make_live_run("done-sess")
    run.status = "done"
    assert agent_runs.enqueue_steer("done-sess", "hello") is False


def test_enqueue_then_drain_fifo():
    _make_live_run("live-1")
    assert agent_runs.enqueue_steer("live-1", "first") is True
    assert agent_runs.enqueue_steer("live-1", "second", kind="user") is True
    assert agent_runs.has_steering("live-1") is True
    drained = agent_runs.drain_steering("live-1")
    assert [d["text"] for d in drained] == ["first", "second"]
    # Drained queue is now empty; a second drain returns nothing.
    assert agent_runs.drain_steering("live-1") == []
    assert agent_runs.has_steering("live-1") is False


# ── /api/chat/steer endpoint ─────────────────────────────────────────────

def _build_client(monkeypatch, verify_ok=True):
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient
    from routes import chat_routes

    def _verify(request, session):
        if not verify_ok:
            raise HTTPException(403, "forbidden")

    monkeypatch.setattr(chat_routes, "_verify_session_owner", _verify)
    router = chat_routes.setup_chat_routes(
        session_manager=None, chat_handler=None, chat_processor=None,
        memory_manager=None, research_handler=None, upload_handler=None,
    )
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_steer_endpoint_queues_when_live(monkeypatch, _clean_runs):
    _make_live_run("mine")
    client = _build_client(monkeypatch)
    r = client.post("/api/chat/steer", json={"session_id": "mine", "text": "do X now"})
    assert r.status_code == 200 and r.json() == {"queued": True}
    # The text is actually on the queue.
    assert [d["text"] for d in agent_runs.drain_steering("mine")] == ["do X now"]


def test_steer_endpoint_409_when_idle(monkeypatch, _clean_runs):
    client = _build_client(monkeypatch)
    r = client.post("/api/chat/steer", json={"session_id": "mine", "text": "hi"})
    assert r.status_code == 409   # client falls back to a normal send


def test_steer_endpoint_owner_verified(monkeypatch, _clean_runs):
    _make_live_run("mine")
    client = _build_client(monkeypatch, verify_ok=False)
    r = client.post("/api/chat/steer", json={"session_id": "mine", "text": "hi"})
    assert r.status_code == 403


def test_steer_endpoint_validates_body(monkeypatch, _clean_runs):
    client = _build_client(monkeypatch)
    assert client.post("/api/chat/steer", json={"session_id": "mine"}).status_code == 400
    assert client.post("/api/chat/steer", json={"text": "hi"}).status_code == 400
    assert client.post("/api/chat/steer", json={"session_id": "mine", "text": "  "}).status_code == 400


# ── loop-level injection (round boundary + final drain) ──────────────────
# These exercise _inject_steering_messages directly: persist as user msg,
# append to the model's message list, reset the resume cap.

class _FakeSession:
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)


class _FakeSM:
    def __init__(self, sess):
        self._sess = sess

    def get_session(self, sid):
        return self._sess


def test_inject_steering_persists_and_reaches_messages(monkeypatch, _clean_runs):
    import src.agent_loop as agent_loop
    import core.models as core_models

    sess = _FakeSession()
    monkeypatch.setattr(core_models, "get_session_manager", lambda: _FakeSM(sess))

    _make_live_run("inj-1")
    agent_runs.enqueue_steer("inj-1", "steer me", kind="user")
    # Pre-load some resume budget so we can prove a user steer resets it.
    subagent_runs.note_server_resume("inj-1")
    assert subagent_runs._resume_count.get("inj-1") == 1

    model_messages = [{"role": "user", "content": "original"}]
    drained = agent_loop._inject_steering_messages("inj-1", model_messages)

    assert [d["text"] for d in drained] == ["steer me"]
    # (b) appended to the model's message list
    assert model_messages[-1] == {"role": "user", "content": "steer me"}
    # (a) persisted as a user-role ChatMessage
    assert len(sess.messages) == 1
    assert sess.messages[0].role == "user"
    assert sess.messages[0].content == "steer me"
    assert (sess.messages[0].metadata or {}).get("steered") is True
    # a genuine user steer is the human talking → stays visible in the transcript
    assert not (sess.messages[0].metadata or {}).get("hidden")
    # user steer counts as user activity → resume cap reset
    assert "inj-1" not in subagent_runs._resume_count


def test_subagent_steer_does_not_reset_resume_cap(monkeypatch, _clean_runs):
    import src.agent_loop as agent_loop
    import core.models as core_models

    sess = _FakeSession()
    monkeypatch.setattr(core_models, "get_session_manager", lambda: _FakeSM(sess))

    _make_live_run("inj-2")
    agent_runs.enqueue_steer("inj-2", "sub result", kind="subagent")
    subagent_runs.note_server_resume("inj-2")

    agent_loop._inject_steering_messages("inj-2", [])
    # A sub-agent-result steer is NOT genuine user activity — cap stays.
    assert subagent_runs._resume_count.get("inj-2") == 1
    # It is framed untrusted context, not user prose, so it must persist
    # hidden — otherwise the "UNTRUSTED SOURCE DATA" wrapper renders as a
    # visible "You" bubble in the transcript.
    assert len(sess.messages) == 1
    assert sess.messages[0].role == "user"
    assert (sess.messages[0].metadata or {}).get("hidden") is True
    assert (sess.messages[0].metadata or {}).get("steer_kind") == "subagent"


def test_start_notifies_prev_run_subscribers_superseded(_clean_runs):
    """A new send for a session must tell the OLD run's subscribers they were
    superseded, so another device converges onto the new run instead of ending
    silently on a stale partial (cross-device sync)."""
    async def _run():
        prev = _make_live_run("sup-sess")
        q = asyncio.Queue()
        prev.subscribers.add(q)                 # a connected client watching the old run

        async def _gen():
            yield "data: hi\n\n"

        agent_runs.start("sup-sess", _gen())    # a newer send replaces the run
        got = []
        while not q.empty():
            got.append(q.get_nowait())
        await asyncio.sleep(0.05)               # let the new run's drain finish (no leak)
        return got

    got = asyncio.run(_run())
    assert any("event: superseded" in (ev or "") for (_seq, ev) in got)


def test_inject_no_live_run_returns_empty(monkeypatch, _clean_runs):
    import src.agent_loop as agent_loop
    assert agent_loop._inject_steering_messages("nope", []) == []
    assert agent_loop._inject_steering_messages(None, []) == []


# ── no-lost-message: final drain continues the loop ──────────────────────
# A steer enqueued during the LAST active round must still be processed: the
# loop's final drain injects it and continues with another round rather than
# ending. We drive stream_agent_loop with a mock LLM and a live run whose queue
# is fed a steer just before the model would otherwise finish.

def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


async def test_final_drain_continues_loop(monkeypatch, _clean_runs):
    import src.agent_loop as agent_loop
    import core.models as core_models

    sess = _FakeSession()
    monkeypatch.setattr(core_models, "get_session_manager", lambda: _FakeSM(sess))
    _make_live_run("drain-1")

    # The model writes plain text (no tool call) every round → the loop would
    # end after round 1. We enqueue a steer during round 1 so the FINAL drain
    # picks it up and forces round 2.
    calls = {"n": 0}

    async def fake_stream_llm_with_fallback(candidates, messages, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # Enqueue a steer mid-round-1 (after the round-boundary drain ran).
            agent_runs.enqueue_steer("drain-1", "keep going", kind="user")
            yield _sse({"delta": "round-one answer"})
        else:
            yield _sse({"delta": "reacted to steer"})
        yield "data: [DONE]\n\n"

    # Neutralize heavy prep so the loop reaches the round body deterministically.
    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream_llm_with_fallback)
    monkeypatch.setattr(agent_loop, "get_mcp_manager", lambda: None)
    monkeypatch.setattr(agent_loop, "_session_used_tools", lambda sid: set())

    saw_steer_injected = False
    saw_reacted = False
    async for chunk in agent_loop.stream_agent_loop(
        "http://x/v1", "test-model",
        [{"role": "user", "content": "hello"}],
        session_id="drain-1",
        max_rounds=4,
        relevant_tools={"read_file"},   # skip RAG retrieval path
    ):
        if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
            try:
                d = json.loads(chunk[6:])
            except Exception:
                continue
            if d.get("type") == "steering_injected" and d.get("text") == "keep going":
                saw_steer_injected = True
            if d.get("delta") == "reacted to steer":
                saw_reacted = True

    # The mid-round steer was injected (final drain) AND the loop ran a 2nd round
    # to react to it — no lost message.
    assert calls["n"] >= 2, "loop did not continue after the final drain"
    assert saw_steer_injected, "steering_injected SSE not emitted"
    assert saw_reacted, "model never got a round to react to the steer"
    # The steered message was persisted as a user message.
    assert any(m.role == "user" and m.content == "keep going" for m in sess.messages)
