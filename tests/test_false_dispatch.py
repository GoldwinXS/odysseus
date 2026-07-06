"""The false-dispatch supervisor: catch a model that ENDS its turn claiming it
launched a background sub-agent when it never called spawn_agent (opus-4.8 as an
orchestrator narrated dispatches — even parroting the server ack — without
emitting the tool call, so the registry stayed empty).

Covers the detection primitives (agent_loop._FALSE_DISPATCH_RE and
subagent_runs.running_count); the running-count guard is what tells a real
"it's running in the background" from a hallucinated claim.
"""

import pytest

from src.agent_loop import _FALSE_DISPATCH_RE
from src import subagent_runs


@pytest.mark.parametrize("text", [
    # The verbatim server ack, parroted as prose without a real dispatch.
    "_Background sub-agent dispatched — its result will arrive here as a separate message when it finishes._",
    "I dispatched a fix agent to correct the barrel geometry.",
    "I'll dispatch a verification agent (Gemini vision).",
    "The sub-agent is now running in the background.",
    "Dispatching a single, narrowly-scoped verification agent.",
    "I spawned an agent to analyze the files.",
])
def test_false_dispatch_matches_dispatch_claims(text):
    assert _FALSE_DISPATCH_RE.search(text) is not None


@pytest.mark.parametrize("text", [
    # Legitimate references to a REAL prior run must not trip the supervisor.
    "The sub-agent finished and reported that all 240 configs pass.",
    "sub_9 completed the visual sign-off cleanly.",
    "I updated the map generation code and verified the tests.",
    "Let me check the logs for the error.",
    "The agent found the bug in units.js and fixed it.",
])
def test_false_dispatch_ignores_legitimate_text(text):
    assert _FALSE_DISPATCH_RE.search(text) is None


def test_running_count_distinguishes_real_from_hallucinated():
    subagent_runs._UPDATES.clear()
    try:
        # No runs for this session — a dispatch claim here is hallucinated.
        assert subagent_runs.running_count("chat-1") == 0
        subagent_runs._UPDATES["chat-1"] = [
            {"id": "sub_1", "status": "running", "queue_session": "_subagent_sub_1"},
            {"id": "sub_2", "status": "done", "queue_session": "_subagent_sub_2"},
        ]
        # One in flight → "it's running in the background" is TRUE → no nudge.
        assert subagent_runs.running_count("chat-1") == 1
        assert subagent_runs.running_count("other") == 0
        assert subagent_runs.running_count(None) == 0
    finally:
        subagent_runs._UPDATES.clear()
