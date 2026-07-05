"""Regression for the interrupted-turn partial save losing tool history.

Bug: when a turn is interrupted mid-stream (client disconnect / Stop), the
partial-save path in routes/chat_routes.py persisted an assistant ChatMessage
carrying ONLY the accumulated narration text — no tool_events. On reload, the
model saw its own bare narration, concluded it had done nothing, and re-did
already-completed tool work.

Fix: stream_agent_loop now appends each round's tool_event to a caller-supplied
``tool_events_sink`` list; the route holds that list and, on the
CancelledError/GeneratorExit partial-save, writes it into the saved message's
metadata (same ``tool_events`` shape a normally-completed turn saves). It also
now saves when tool_events exist even if the narration is empty.

These tests exercise the two halves against the REAL code:

  1. stream_agent_loop wiring — the ``tool_events_sink`` parameter exists and the
     loop uses the passed list as its accumulator (so appends are visible live).
  2. the route's partial-save contract — the except-handler builds metadata that
     carries the sink's tool_events, and fires even for empty narration. Driven
     with a fake generator that appends to the sink exactly as the real loop
     does, plus source-wiring assertions that pin the actual route call/handler.
"""
import asyncio
import inspect
from pathlib import Path

import pytest

from routes.chat_helpers import clean_thinking_for_save
from core.models import ChatMessage


_ROUTES_SRC = (Path(__file__).resolve().parents[1] / "routes" / "chat_routes.py").read_text(encoding="utf-8")
_LOOP_SRC = (Path(__file__).resolve().parents[1] / "src" / "agent_loop.py").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# 1. stream_agent_loop threads the sink out to the caller.
# --------------------------------------------------------------------------- #

def test_stream_agent_loop_accepts_tool_events_sink():
    """The generator must expose tool_events_sink so the route can hold the
    in-progress tool history across an interrupt."""
    from src.agent_loop import stream_agent_loop
    params = inspect.signature(stream_agent_loop).parameters
    assert "tool_events_sink" in params
    assert params["tool_events_sink"].default is None


def test_loop_uses_passed_sink_as_accumulator_in_source():
    """The loop must reuse the passed sink AS the tool_events accumulator (not a
    private copy), so its live appends are visible to the caller. Pins the actual
    aliasing so a refactor that reverts to a private list is caught."""
    assert "tool_events = tool_events_sink if tool_events_sink is not None else []" in _LOOP_SRC
    # And the per-round tool_event is appended to that same name.
    assert "tool_events.append(tool_event)" in _LOOP_SRC


# --------------------------------------------------------------------------- #
# 2. The route's partial-save contract: fake loop appends to the sink, the
#    except-handler folds the sink into the saved message's metadata.
# --------------------------------------------------------------------------- #

def _fake_tool_event(round_num, tool, command):
    """Same shape stream_agent_loop persists (agent_loop.py tool_event dict)."""
    return {"round": round_num, "tool": tool, "command": command,
            "output": "ok", "exit_code": 0}


async def _interrupted_turn(sink, saved, *, narration, events, hang_evt):
    """Mirror routes/chat_routes.py stream_with_save (agent mode): a fake agent
    loop appends tool_events to ``sink`` as it runs, accumulating ``narration``;
    it hangs before the final metrics/[DONE] so a cancel interrupts it, and the
    except-handler saves the partial WITH the sink's tool_events — exactly the
    real handler's logic."""
    full_response = ""
    _tool_events_sink = sink  # the route holds this and passes it to the loop
    try:
        # Simulate the loop running: it appends tool_events + narration, then
        # blocks awaiting the next upstream chunk (the interrupt point).
        for ev in events:
            _tool_events_sink.append(ev)
        full_response += narration
        await hang_evt.wait()  # never set — cancellation interrupts here
    except (asyncio.CancelledError, GeneratorExit):
        # ---- the fixed partial-save handler (mirrors chat_routes.py) ----
        _partial_tool_events = list(_tool_events_sink)
        if full_response or _partial_tool_events:
            _stopped_md_base = {"stopped": True, "model": "m", "requested_model": "m"}
            if _partial_tool_events:
                _stopped_md_base["tool_events"] = _partial_tool_events
            _content, _md = clean_thinking_for_save(full_response, _stopped_md_base)
            saved.append(ChatMessage("assistant", _content, metadata=_md))
        raise


async def _run_and_cancel(coro_factory):
    task = asyncio.ensure_future(coro_factory())
    for _ in range(200):
        await asyncio.sleep(0.005)
        if not task.done():
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_partial_save_preserves_tool_events_with_narration():
    sink, saved = [], []
    hang = asyncio.Event()
    events = [_fake_tool_event(1, "read_file", "read a.py"),
              _fake_tool_event(2, "edit_file", "edit a.py")]
    await _run_and_cancel(
        lambda: _interrupted_turn(sink, saved, narration="Editing files",
                                  events=events, hang_evt=hang)
    )
    assert len(saved) == 1
    md = saved[0].metadata
    assert md.get("stopped") is True
    # The tool history the model must SEE on reload is present, unabridged.
    assert md.get("tool_events") == events
    assert [e["tool"] for e in md["tool_events"]] == ["read_file", "edit_file"]


@pytest.mark.asyncio
async def test_partial_save_fires_for_tool_only_turn_with_empty_narration():
    """The gaslighting case: the model ran tools but wrote (almost) no narration
    before the interrupt. The old handler gated the save on ``full_response`` and
    would skip it entirely — losing the tool history. It must now save because
    tool_events exist."""
    sink, saved = [], []
    hang = asyncio.Event()
    events = [_fake_tool_event(1, "bash", "pytest -q")]
    await _run_and_cancel(
        lambda: _interrupted_turn(sink, saved, narration="",
                                  events=events, hang_evt=hang)
    )
    assert len(saved) == 1, "tool-only interrupted turn must still be saved"
    assert saved[0].metadata.get("tool_events") == events


@pytest.mark.asyncio
async def test_partial_save_skips_when_nothing_happened():
    """No narration and no tools → nothing to save (unchanged behavior)."""
    sink, saved = [], []
    hang = asyncio.Event()
    await _run_and_cancel(
        lambda: _interrupted_turn(sink, saved, narration="", events=[], hang_evt=hang)
    )
    assert saved == []


# --------------------------------------------------------------------------- #
# 3. Source wiring: the real route must create the sink, pass it into
#    stream_agent_loop, and fold it into the partial-save metadata. Pins the
#    actual chat_routes.py edits (not re-derivable from the behavioral fakes).
# --------------------------------------------------------------------------- #

def test_route_creates_and_passes_sink_and_saves_it():
    # Sink created for the agent-mode turn...
    assert "_tool_events_sink: list = []" in _ROUTES_SRC
    # ...threaded into the agent loop...
    assert "tool_events_sink=_tool_events_sink" in _ROUTES_SRC
    # ...read back in the partial-save handler and written to metadata...
    assert "_partial_tool_events = list(_tool_events_sink)" in _ROUTES_SRC
    assert '_stopped_md_base["tool_events"] = _partial_tool_events' in _ROUTES_SRC
    # ...and the save now fires on tool_events even without narration text.
    assert "if full_response or _partial_tool_events:" in _ROUTES_SRC
