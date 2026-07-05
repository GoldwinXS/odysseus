"""Reliability: silent failures must reach the model (as a recoverable tool
result) or the user (as a visible event/message) — never vanish mid-turn.

Covers the four fixes in the "make silent failures visible" pass:
  FIX 1  a raising tool handler → execute_tool_block returns a tool error dict
         (the model sees it) instead of re-raising out of the SSE stream; a
         CancelledError still propagates (teardown is not swallowed).
  FIX 2  an unconvertible / unknown native tool call → the loop feeds back a
         "Couldn't run tool call X" error with a difflib close-match, instead
         of dropping the call and leaving the model waiting forever.
  FIX 3  a bare KeyError (empty str(e)) → the surfaced error names the exception
         TYPE, so the model/user don't get a blank "tool: " with no cause.
  FIX 4  timeout vs genuinely-empty vs provider-error produce DISTINCT terminal
         messages — a timeout never tells the user to switch models.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _clean_runs():
    import src.agent_runs as agent_runs
    yield
    agent_runs._RUNS.clear()


def _make_block(tool_type, content=""):
    return SimpleNamespace(tool_type=tool_type, content=content)


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


# ── shared loop-driving fakes (mirror tests/test_chat_steering.py) ──────────

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


# ─────────────────────────────────────────────────────────────────────────
# FIX 1 — a raising tool handler becomes a tool error the model sees
# ─────────────────────────────────────────────────────────────────────────

async def test_raising_handler_returns_tool_error_not_exception(monkeypatch):
    """execute_tool_block's OUTER wrapper turns a handler that raises on an
    UNWRAPPED dispatch branch (here create_document via _document_tool_dispatch)
    into a tool_result-style error dict, so the model can recover instead of the
    exception re-raising out of the stream and killing the turn with no frame."""
    import src.tool_execution as te

    async def _boom(content, ctx):
        raise RuntimeError("handler exploded")

    monkeypatch.setattr(te, "get_mcp_manager", lambda: None)
    # create_document routes through _document_tool_dispatch, which calls the
    # handler UNWRAPPED — so only the execute_tool_block wrapper (FIX 1a) can
    # catch this.
    monkeypatch.setitem(
        __import__("src.agent_tools", fromlist=["TOOL_HANDLERS"]).TOOL_HANDLERS,
        "create_document", _boom,
    )

    desc, result = await te.execute_tool_block(_make_block("create_document", "Title\ntext\nbody"))
    # No exception propagated; the model receives a recoverable error.
    assert result.get("exit_code") == 1
    assert "RuntimeError" in result["error"]
    assert "handler exploded" in result["error"]


async def test_execute_tool_block_reraises_cancelled_error(monkeypatch):
    """A CancelledError must propagate for teardown — it is NOT turned into a
    tool result (that would leak a runaway tool the cancel meant to stop)."""
    import src.tool_execution as te

    async def _cancel(content, ctx):
        raise asyncio.CancelledError()

    monkeypatch.setattr(te, "get_mcp_manager", lambda: None)
    monkeypatch.setitem(
        __import__("src.agent_tools", fromlist=["TOOL_HANDLERS"]).TOOL_HANDLERS,
        "create_document", _cancel,
    )

    with pytest.raises(asyncio.CancelledError):
        await te.execute_tool_block(_make_block("create_document", "Title\ntext\nbody"))


# ─────────────────────────────────────────────────────────────────────────
# FIX 3 — empty str(e) (bare KeyError) still names the exception type
# ─────────────────────────────────────────────────────────────────────────

async def test_bare_keyerror_surfaces_exception_type(monkeypatch):
    """A KeyError with no message has an empty str(e); the surfaced error must
    still identify it by TYPE so the cause isn't a blank string."""
    import src.tool_execution as te

    async def _bare(content, ctx):
        raise KeyError()  # str(KeyError()) == "" — the flattened-error trap

    monkeypatch.setattr(te, "get_mcp_manager", lambda: None)
    monkeypatch.setitem(
        __import__("src.agent_tools", fromlist=["TOOL_HANDLERS"]).TOOL_HANDLERS,
        "bare_tool", _bare,
    )

    desc, result = await te.execute_tool_block(_make_block("bare_tool", "{}"))
    assert "KeyError" in result["error"], result["error"]
    # Never a blank "tool: " with no cause.
    assert result["error"].strip() not in ("bare_tool:", "bare_tool: ")


async def test_direct_fallback_error_includes_type(monkeypatch):
    """_direct_fallback's own except also names the type + logs the traceback."""
    import src.tool_execution as te

    async def _bare(content, ctx):
        raise AttributeError()

    monkeypatch.setitem(
        __import__("src.agent_tools", fromlist=["TOOL_HANDLERS"]).TOOL_HANDLERS,
        "attr_tool", _bare,
    )
    res = await te._direct_fallback("attr_tool", "{}")
    assert "AttributeError" in res["error"]


# ─────────────────────────────────────────────────────────────────────────
# FIX 2 — unconvertible / unknown tool call is surfaced, not dropped
# ─────────────────────────────────────────────────────────────────────────

def test_unknown_tool_call_error_suggests_close_match():
    import src.agent_loop as agent_loop
    # A near-miss of a real tool name yields a "did you mean" suggestion and
    # points at search_tools.
    msg = agent_loop._unknown_tool_call_error("read_fil")
    assert "Couldn't run tool call 'read_fil'" in msg
    assert "search_tools" in msg
    assert "read_file" in msg  # difflib close match


def test_resolve_tool_blocks_reports_failed_calls(monkeypatch):
    """A native call whose name doesn't convert is returned in failed_call_names
    rather than being silently dropped from the round."""
    import src.agent_loop as agent_loop

    monkeypatch.setattr(
        agent_loop, "function_call_to_tool_block",
        lambda name, args: None if name == "not_a_tool" else SimpleNamespace(tool_type=name, content=args),
    )
    tool_blocks, used_native, converted, failed = agent_loop._resolve_tool_blocks(
        "", [{"name": "not_a_tool", "arguments": "{}"}], round_num=1,
    )
    assert tool_blocks == []
    assert failed == ["not_a_tool"]


async def test_loop_feeds_back_unconvertible_call(monkeypatch, _clean_runs):
    """End to end: the model emits a native call that can't be converted. The
    loop must NOT end the turn silently — it feeds an error back and loops so
    the model self-corrects."""
    import src.agent_loop as agent_loop
    import src.agent_runs as agent_runs
    import core.models as core_models

    sess = _FakeSession()
    monkeypatch.setattr(core_models, "get_session_manager", lambda: _FakeSM(sess))

    # Round 1: a native tool call for a bogus name (won't convert).
    # Round 2: plain text so the loop ends normally. Capture the messages the
    # loop feeds the model on round 2 — the recoverable error must be in there.
    calls = {"n": 0}
    round2_messages = {}

    async def fake_stream(candidates, messages, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            yield _sse({"type": "tool_calls", "calls": [{"name": "reed_file", "arguments": "{}"}]})
        else:
            round2_messages["msgs"] = list(messages)
            yield _sse({"delta": "recovered"})
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "function_call_to_tool_block", lambda name, args: None)
    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream)
    monkeypatch.setattr(agent_loop, "get_mcp_manager", lambda: None)
    monkeypatch.setattr(agent_loop, "_session_used_tools", lambda sid: set())

    _run = agent_runs._Run()
    _run.status = "running"
    agent_runs._RUNS["fix2-loop"] = _run

    async for _ in agent_loop.stream_agent_loop(
        "http://x/v1", "test-model",
        [{"role": "user", "content": "hi"}],
        session_id="fix2-loop",
        max_rounds=4,
        relevant_tools={"read_file"},
    ):
        pass

    # It ran a second round instead of dying after the dropped call.
    assert calls["n"] >= 2, "loop ended silently on an unconvertible call"
    # The recoverable error reached the model's message history on round 2.
    joined = " ".join(str(m.get("content", "")) for m in round2_messages.get("msgs", []))
    assert "Couldn't run tool call 'reed_file'" in joined, joined
    assert "search_tools" in joined


# ─────────────────────────────────────────────────────────────────────────
# FIX 4 — timeout vs empty vs provider produce distinct messages
# ─────────────────────────────────────────────────────────────────────────

def test_empty_response_fallback_distinguishes_causes():
    import src.agent_loop as agent_loop

    # Genuinely empty reply → generic "try again / switch model".
    _resp, chunk = agent_loop._empty_response_fallback("", "", [])
    assert "empty response" in _resp
    assert "switch" in _resp.lower()

    # Timeout → says it TIMED OUT with the duration; does NOT suggest switching.
    _resp_t, chunk_t = agent_loop._empty_response_fallback(
        "", "", [], no_output_reason="timeout", timeout_seconds=1200,
    )
    assert "timed out" in _resp_t.lower()
    assert "1200s" in _resp_t
    assert "switch" not in _resp_t.lower()
    assert _resp_t != _resp

    # Provider error → surfaces the real upstream reason.
    _resp_p, chunk_p = agent_loop._empty_response_fallback(
        "", "", [], no_output_reason="provider", reason_detail="502 upstream boom",
    )
    assert "provider" in _resp_p.lower()
    assert "502 upstream boom" in _resp_p
    assert _resp_p != _resp and _resp_p != _resp_t


def test_empty_response_fallback_keeps_real_output():
    import src.agent_loop as agent_loop
    # A real reply is passed through untouched (no fallback chunk).
    resp, chunk = agent_loop._empty_response_fallback("real answer", "", [])
    assert resp == "real answer"
    assert chunk is None


async def test_loop_provider_error_surfaces_real_cause(monkeypatch, _clean_runs):
    """End to end: an upstream provider error with no content must yield a
    terminal message naming the PROVIDER cause — not the generic empty-response
    "switch models" text (FIX 4)."""
    import src.agent_loop as agent_loop
    import src.agent_runs as agent_runs
    import core.models as core_models

    sess = _FakeSession()
    monkeypatch.setattr(core_models, "get_session_manager", lambda: _FakeSM(sess))

    async def fake_stream(candidates, messages, **kwargs):
        # In-stream provider error, no usable content this round.
        yield _sse({"error": "502 Bad Gateway from upstream"})
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream)
    monkeypatch.setattr(agent_loop, "get_mcp_manager", lambda: None)
    monkeypatch.setattr(agent_loop, "_session_used_tools", lambda sid: set())

    _run = agent_runs._Run()
    _run.status = "running"
    agent_runs._RUNS["fix4-loop"] = _run

    deltas = []
    async for chunk in agent_loop.stream_agent_loop(
        "http://x/v1", "test-model",
        [{"role": "user", "content": "hi"}],
        session_id="fix4-loop",
        max_rounds=1,
        relevant_tools={"read_file"},
    ):
        if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
            try:
                d = json.loads(chunk[6:])
            except Exception:
                continue
            if "delta" in d:
                deltas.append(d["delta"])

    joined = " ".join(deltas)
    # The real upstream reason is surfaced, and it is NOT the "switch model" text.
    assert "provider" in joined.lower()
    assert "502 Bad Gateway from upstream" in joined
    assert "switch to a different model" not in joined
