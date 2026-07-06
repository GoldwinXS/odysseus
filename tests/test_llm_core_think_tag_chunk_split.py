"""Regression: a "<think>" opener split across SSE delta chunks defeated the
plain-<think> auto-detect in stream_llm's OpenAI-compatible branch.

stream_llm auto-detects literal "<think>...</think>" markup in the content
stream for models that emit it via llama.cpp --jinja but whose name doesn't
match _THINKING_MODEL_PATTERNS (e.g. Qwen3-derived forks). The detector
required the WHOLE "<think" prefix to appear in a SINGLE delta chunk
(``stripped.lower().startswith("<think")``). When a provider splits the tag
across chunks — first chunk just "<th", second chunk "ink>reasoning..." — the
check fails on the first (partial) chunk, falls through to the plain-content
branch, and permanently sets _first_content_sent=True. That flag gates the
detector (``not _first_content_sent``), so it never fires for the REST of
that stream_llm call — the opener and everything after it streams as plain
(non-thinking) content instead of being routed to the thinking channel.

Fix: hold back content that could still be an in-progress "<think" prefix
until it resolves (either the buffer no longer matches any prefix, or it's
long enough to check for real) — mirroring the harmony router's existing
suffix-hold buffering for its own multi-char markers.
"""
import asyncio
import json

from src import llm_core


class _FakeResp:
    def __init__(self, lines):
        self._lines = lines
        self.status_code = 200

    async def aiter_lines(self):
        for ln in self._lines:
            yield ln

    async def aread(self):
        return b""


class _FakeStreamCtx:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return _FakeResp(self._lines)

    async def __aexit__(self, *a):
        return False


class _FakeClient:
    def __init__(self, lines):
        self._lines = lines

    def stream(self, method, url, **kw):
        return _FakeStreamCtx(self._lines)


def _drive(monkeypatch, lines, model="my-custom-local-model"):
    """Drive stream_llm over a generic (non-Anthropic/Ollama-native) URL so it
    takes the plain OpenAI-compatible streaming branch. The model name
    deliberately does NOT match _THINKING_MODEL_PATTERNS, so _thinking_model
    starts False and the plain-<think> AUTO-DETECT path (not the
    name-based/always-on path) is what's under test."""
    monkeypatch.setattr(llm_core, "_get_http_client", lambda: _FakeClient(lines))
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda u: False)
    monkeypatch.setattr(llm_core, "note_model_activity", lambda *a, **k: None)
    monkeypatch.setattr(llm_core, "_clear_host_dead", lambda *a, **k: None)
    monkeypatch.setattr(llm_core, "_mark_host_dead", lambda *a, **k: False, raising=False)
    assert not llm_core._supports_thinking(model), "test model must NOT match _THINKING_MODEL_PATTERNS"

    async def run():
        out = []
        async for chunk in llm_core.stream_llm(
            "https://my-local-endpoint.example.com/v1/chat/completions",
            model, [{"role": "user", "content": "hi"}],
            headers={"Authorization": "Bearer k"},
        ):
            out.append(chunk)
        return "".join(out)

    return asyncio.run(run())


def _events(blob):
    events = []
    for ln in blob.split("\n"):
        ln = ln.strip()
        if ln.startswith("data: ") and ln[6:] != "[DONE]":
            try:
                j = json.loads(ln[6:])
            except ValueError:
                continue
            if "delta" in j:
                events.append((j["delta"], bool(j.get("thinking"))))
    return events


def _content_chunk(text):
    return "data:" + json.dumps({"choices": [{"delta": {"content": text}}]})


def test_think_tag_split_across_two_chunks_is_still_detected(monkeypatch):
    """The opener "<think>" split as "<th" + "ink>reasoning here</think>answer"
    must still be recognized and routed to the thinking channel — not streamed
    as plain content."""
    lines = [
        _content_chunk("<th"),
        _content_chunk("ink>reasoning here</think>answer"),
        "data:[DONE]",
    ]
    blob = _drive(monkeypatch, lines)
    events = _events(blob)

    thinking_text = "".join(text for text, is_thinking in events if is_thinking)
    plain_text = "".join(text for text, is_thinking in events if not is_thinking)

    assert "reasoning here" in thinking_text, f"events={events!r}"
    assert plain_text == "answer", f"events={events!r}"
    # The bug's exact symptom: the opener leaking into the plain channel.
    assert "<th" not in plain_text and "<think" not in plain_text


def test_think_tag_split_across_many_tiny_chunks_is_still_detected(monkeypatch):
    """More aggressive split (one character at a time for the opener) — the
    hold-back buffer must survive an arbitrary number of tiny fragments, not
    just a two-way split."""
    opener = "<think>"
    lines = [_content_chunk(ch) for ch in opener]
    lines.append(_content_chunk("deep reasoning"))
    lines.append(_content_chunk("</think>"))
    lines.append(_content_chunk("final answer"))
    lines.append("data:[DONE]")

    blob = _drive(monkeypatch, lines)
    events = _events(blob)

    thinking_text = "".join(text for text, is_thinking in events if is_thinking)
    plain_text = "".join(text for text, is_thinking in events if not is_thinking)

    assert "deep reasoning" in thinking_text, f"events={events!r}"
    assert plain_text == "final answer", f"events={events!r}"


def test_unsplit_think_tag_still_works(monkeypatch):
    """Baseline: the common case (the whole "<think>" arrives in one chunk)
    must be unaffected by the new buffering."""
    lines = [
        _content_chunk("<think>reasoning</think>answer"),
        "data:[DONE]",
    ]
    blob = _drive(monkeypatch, lines)
    events = _events(blob)

    thinking_text = "".join(text for text, is_thinking in events if is_thinking)
    plain_text = "".join(text for text, is_thinking in events if not is_thinking)

    assert "reasoning" in thinking_text
    assert plain_text == "answer"


def test_ordinary_short_first_chunk_is_not_held_forever(monkeypatch):
    """A model that doesn't think at all and happens to start with a short
    reply ("Hi") must not be delayed/mangled by the new hold-back buffer — it
    resolves immediately once the buffered text diverges from "<think"."""
    lines = [
        _content_chunk("Hi"),
        _content_chunk(" there!"),
        "data:[DONE]",
    ]
    blob = _drive(monkeypatch, lines)
    events = _events(blob)

    plain_text = "".join(text for text, is_thinking in events if not is_thinking)
    thinking_text = "".join(text for text, is_thinking in events if is_thinking)

    assert plain_text == "Hi there!"
    assert thinking_text == ""
