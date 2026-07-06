"""model_interaction_tools.py - agent tools for talking to other models.

Owns the model-interaction tool implementations (chat_with_model, ask_teacher,
list_models) and their handler classes, registered in ``TOOL_HANDLERS``. Part
of the tool -> registry migration (#3629): the implementations were moved here
out of ``src.ai_interaction`` so dispatch flows through the registry instead of
the elif chain / dispatch_ai_tool in tool_execution.py.

Shared helpers that still live in ``src.ai_interaction`` and are used by tools
not yet migrated (``_resolve_model``, ``AI_CHAT_TIMEOUT``) are imported lazily
inside the functions to avoid an import cycle at module load.
"""
import asyncio
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


_TEACHER_SYSTEM_PROMPT = (
    "You are a senior AI mentor. A less capable model is stuck on a problem and asking for help. "
    "Provide clear, actionable guidance:\n"
    "1. Brief analysis of the problem\n"
    "2. Recommended approach (step by step)\n"
    "3. Key things to watch out for\n\n"
    "Be concise and practical. No preamble."
)


async def chat_with_model(content: str, session_id: Optional[str] = None, owner: Optional[str] = None) -> Dict:
    """Send a message to a specific model and return its response.

    Content format:
      Line 1: model_name (or model_name@endpoint_name)
      Line 2+: the message to send
    """
    from src.ai_interaction import _resolve_model, AI_CHAT_TIMEOUT
    from src.llm_core import llm_call_async

    lines = content.strip().split("\n", 1)
    if not lines or not lines[0].strip():
        return {"error": "First line must be the model name"}

    model_spec = lines[0].strip()
    message = lines[1].strip() if len(lines) > 1 else ""
    if not message:
        return {"error": "No message provided (line 2+ is the message)"}

    try:
        url, model, headers = await asyncio.to_thread(_resolve_model, model_spec, owner=owner)
    except ValueError as e:
        return {"error": str(e)}

    try:
        response = await llm_call_async(
            url, model,
            [{"role": "user", "content": message}],
            headers=headers,
            timeout=AI_CHAT_TIMEOUT,
        )
        # Truncate very long responses
        if len(response) > 10000:
            response = response[:10000] + "\n... (truncated)"
        return {"model": model, "response": response}
    except Exception as e:
        logger.error(f"chat_with_model failed: {e}")
        return {"error": f"Failed to get response from {model_spec}: {e}"}


async def ask_teacher(content: str, session_id: Optional[str] = None, owner: Optional[str] = None) -> Dict:
    """Ask a more capable model for help.

    Content format:
      Line 1: model_name (or 'auto')
      Line 2+: the problem description
    """
    from src.ai_interaction import _resolve_model, AI_CHAT_TIMEOUT
    from src.llm_core import llm_call_async
    from src.settings import get_setting

    lines = content.strip().split("\n", 1)
    model_spec = lines[0].strip() if lines else "auto"
    problem = lines[1].strip() if len(lines) > 1 else ""

    if not problem:
        return {"error": "No problem description provided"}

    if model_spec.lower() in ("auto", ""):
        model_spec = get_setting("teacher_model", "")
        if not model_spec:
            return {"error": "No teacher model configured. Specify a model name or set teacher_model in settings."}

    try:
        url, model, headers = await asyncio.to_thread(_resolve_model, model_spec, owner=owner)
    except ValueError as e:
        return {"error": str(e)}

    try:
        response = await llm_call_async(
            url, model,
            [
                {"role": "system", "content": _TEACHER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Problem:\n{problem}"},
            ],
            headers=headers,
            timeout=AI_CHAT_TIMEOUT,
        )
        if len(response) > 8000:
            response = response[:8000] + "\n... (truncated)"
        return {"model": model, "response": response, "teacher": True}
    except Exception as e:
        logger.error(f"ask_teacher failed: {e}")
        return {"error": f"Teacher call failed ({model_spec}): {e}"}


# ── Sub-agents ──────────────────────────────────────────────────────────────
# Hard guardrails (a runaway here can exhaust the machine — see the RAM
# incident): a sub-agent is a LEAF worker. It cannot spawn or orchestrate
# further (depth is capped at exactly 1 — no fork bombs), concurrency is
# bounded, each run is round- and time-limited.
_SUBAGENT_SEMAPHORE = asyncio.Semaphore(3)   # max concurrent sub-agents
_SUBAGENT_MAX_ROUNDS = 100                    # RUNAWAY BACKSTOP, not a working ceiling
                                              # — a sub-agent making genuine progress
                                              # must never be killed just for taking
                                              # many rounds. (Raised from 12, then 30: a
                                              # sub-agent investigating a 282KB JS file
                                              # deterministically hit the old 12-round
                                              # cap before finishing, causing the parent
                                              # to re-dispatch the same scope 3x instead
                                              # of getting one real answer.) Circling is
                                              # caught by _SUBAGENT_LOOP_REPEATS and the
                                              # stall/wall-clock watchdogs below, not by
                                              # this cap. Tunable via subagent_max_rounds
                                              # (settings.json may pin a lower value).
_SUBAGENT_TIMEOUT_S = 3600                    # wall-clock cap — a GENEROUS SAFETY
                                              # BACKSTOP, not the primary guard. Stall
                                              # detection (below) is what kills hangs;
                                              # this only stops a truly runaway run that
                                              # somehow keeps emitting. Default 1h; the
                                              # user does NOT want subagents hamstrung by
                                              # artificial time limits. Tunable via
                                              # subagent_timeout_seconds, and a value of
                                              # 0 (or negative) DISABLES the backstop
                                              # entirely (stall detection still applies).
_SUBAGENT_STALL_S = 180                       # PRIMARY guard: kill a sub-agent that emits
                                              # NOTHING (no delta/thinking/tool event) for
                                              # this long — that's a hung blocking command,
                                              # not slow-but-live work. A live worker streams
                                              # progress well inside this window. Tunable via
                                              # subagent_stall_timeout_seconds.
_SUBAGENT_LOOP_REPEATS = 3                    # kill if the SAME tool call (name+args)
                                              # repeats this many times consecutively —
                                              # a stuck loop burning rounds/time.

def _subagent_timeout() -> int:
    """Live wall-clock SAFETY BACKSTOP, in seconds. Tunable via
    subagent_timeout_seconds. Returns 0 to mean DISABLED (no wall-clock kill —
    stall detection is then the only time-based guard), so the user can turn the
    backstop off entirely. Falls back to the generous default on bad config."""
    try:
        from src.settings import get_setting
        raw = get_setting("subagent_timeout_seconds", _SUBAGENT_TIMEOUT_S)
        if raw is None:
            return _SUBAGENT_TIMEOUT_S
        v = int(raw)
        # <= 0 is an explicit "disable the backstop" signal, not a fallback.
        return v if v >= 0 else 0
    except Exception:
        return _SUBAGENT_TIMEOUT_S


def _subagent_stall_timeout() -> int:
    """Live silence-watchdog window: max seconds with no SSE activity before the
    sub-agent is treated as hung and killed. Tunable; falls back to the default."""
    try:
        from src.settings import get_setting
        v = int(get_setting("subagent_stall_timeout_seconds", _SUBAGENT_STALL_S) or _SUBAGENT_STALL_S)
        return v if v > 0 else _SUBAGENT_STALL_S
    except Exception:
        return _SUBAGENT_STALL_S


_SUBAGENT_WRAPUP_GRACE_S = 90                  # graceful wind-down window: when a
                                              # stall/wall/loop kill fires, first
                                              # STEER the worker to stop and write a
                                              # final summary, then give it this long
                                              # to comply before hard-killing. Tunable
                                              # via subagent_wrapup_grace_seconds; 0
                                              # disables the courtesy wrap-up (kill
                                              # immediately, old behaviour).
_SUBAGENT_WRAPUP_STALL_GRACE_S = 30            # SHORTER grace for a stall kill: a
                                              # truly wedged loop can't respond to a
                                              # steer, so we don't wait the full window
                                              # for it — attempt the steer but kill
                                              # fast if no NEW activity appears.


def _subagent_wrapup_grace(kill_reason: str) -> int:
    """Grace window (seconds) to let a sub-agent write a final summary after a
    wind-down steer, before it is hard-killed. Shorter for a stall kill (a wedged
    worker likely can't respond) than for a wall-clock/loop kill (the worker is
    alive, just over-budget). 0 disables the wrap-up entirely. Tunable via
    subagent_wrapup_grace_seconds (applies to the non-stall window; the stall
    window is derived as the min of that and the short stall grace)."""
    try:
        from src.settings import get_setting
        raw = get_setting("subagent_wrapup_grace_seconds", _SUBAGENT_WRAPUP_GRACE_S)
        base = _SUBAGENT_WRAPUP_GRACE_S if raw is None else int(raw)
    except Exception:
        base = _SUBAGENT_WRAPUP_GRACE_S
    if base <= 0:
        return 0   # wrap-up disabled → caller hard-kills immediately
    if kill_reason == "stall":
        # A stalled worker probably can't answer; give it only a short window.
        return min(base, _SUBAGENT_WRAPUP_STALL_GRACE_S)
    return base


def _subagent_max_rounds() -> int:
    """Live tool-loop round cap per sub-agent. Tunable via
    subagent_max_rounds. Falls back to the hardcoded default on bad/missing
    config. Clamped to >= 1 so a misconfig can't zero-out the loop."""
    try:
        from src.settings import get_setting
        raw = get_setting("subagent_max_rounds", _SUBAGENT_MAX_ROUNDS)
        if raw is None:
            return _SUBAGENT_MAX_ROUNDS
        v = int(raw)
        return v if v >= 1 else _SUBAGENT_MAX_ROUNDS
    except Exception:
        return _SUBAGENT_MAX_ROUNDS


def _subagent_bash_cap(stall_s: int) -> int:
    """Cap for bash *inside* this sub-agent, so one blocking foreground command
    (a dev server the model forgot to background) can't sit silent until the STALL
    watchdog fires. Keyed off the stall window (the primary guard), not the
    wall-clock backstop — which may be huge or disabled. Takes the smaller of the
    configured global bash timeout and the stall window, capped at 150s (a single
    agent bash step rarely needs more; a genuinely long build should stream output,
    which keeps the stall watchdog fed anyway)."""
    try:
        from src.settings import get_setting
        _cfg = int(get_setting("agent_bash_timeout_seconds", 600) or 600)
    except Exception:
        _cfg = 600
    return max(30, min(_cfg, max(1, stall_s), 150))
_SUBAGENT_RESULT_CAP = 30000                  # max chars of result delivered (the
                                              # parent sees this verbatim — keep it
                                              # generous so answers aren't truncated)
_SUBAGENT_MAX_TOKENS = 12000                  # per-round output cap. The loop default
                                              # (4096) cut long final summaries off
                                              # mid-word; sized to cover the char cap
                                              # above (~2.5 dense chars/token).
# Tools a sub-agent may NOT use: anything that would spawn/orchestrate more
# agents (recursion) or reach into other chats. ask_user/update_plan are also
# barred: a background sub-agent has no interactive channel, so a run that ends
# by asking the user a question just dead-ends with a question nobody can answer.
_SUBAGENT_DISABLED = frozenset({
    "spawn_agent", "manage_agents", "create_session", "send_to_session",
    "list_sessions", "manage_session", "pipeline", "chat_with_model", "ask_teacher",
    "ask_user", "update_plan",
})
# Tools EVERY sub-agent must reliably have, regardless of how its task is worded.
# A sub-agent's tools are otherwise picked by RAG from the task text alone (it has
# no session history / sticky tools to fall back on), so a task like "improve the
# buildings graphics" that never lexically mentions files would be dispatched with
# NO read_file/edit_file/bash — the worker then reports itself "blocked, I have no
# filesystem tools" while a sibling whose wording happened to retrieve them
# succeeds. These are unconditionally seeded so a dispatched worker can always
# read, edit, search, and run — the whole point of a sub-agent. (MCP tools like
# the browser are already unconditional; only native tools are RAG-gated.)
_SUBAGENT_TOOL_BASELINE = frozenset({
    "read_file", "write_file", "edit_file", "ls", "grep", "get_workspace", "bash",
    "web_search", "web_fetch",
})
# Reported back to the dispatcher/user verbatim, so the sub-agent MUST end with a
# written answer — otherwise a model that spends its whole round budget on tool
# calls (common with Gemini, which emits no narration between calls) delivers an
# empty "(no text output)" result. This is the #1 cause of "bizarre" empty/
# truncated deliveries.
_SUBAGENT_SYSTEM_PROMPT = (
    "You are a background sub-agent dispatched to carry out ONE specific task and "
    "report the result back. Work efficiently — you have a limited number of tool "
    "rounds. When you have enough to answer (or you are running low on rounds), "
    "STOP calling tools and write your final answer.\n\n"
    "CRITICAL: your final written message is the ONLY thing reported back to whoever "
    "dispatched you. If you end without writing a clear, self-contained summary of "
    "what you found or did, they receive NOTHING. Always finish with that summary — "
    "concise but complete, and understandable on its own without your tool history."
)


def _is_provider_error(text: str) -> bool:
    """Heuristic: does this failure reason point at the PROVIDER being down /
    rate-limited / out of credits, rather than the task itself failing? Waking the
    parent to "decide the next step" is pointless when the next spawn would hit the
    same wall — and worse, it's exactly the spawn/fail/resume token-burn loop the
    original _deliver comment guarded against. So we suppress the failure-resume in
    that case and just deliver the notice. Conservative substring match on the
    reasons the loop actually surfaces (429s, spend/credit caps, auth, provider
    unreachable)."""
    t = (text or "").lower()
    return any(k in t for k in (
        "rate-limit", "rate limit", "429", "quota", "spend cap", "spend-cap",
        "credit", "billing", "insufficient", "payment", "unauthorized", "401",
        "403", "forbidden", "api key", "provider error", "upstream error",
        "overloaded", "503", "502", "connection error", "unreachable",
    ))


class _SubagentLoopError(Exception):
    """Raised inside a sub-agent drain when the same tool call repeats
    consecutively past the loop-detection threshold — the worker is stuck. Carries
    the offending tool name + repeat count so the delivered failure can name it."""
    def __init__(self, tool: str, count: int):
        self.tool = tool
        self.count = count
        super().__init__(f"repeated identical {tool} calls x{count}")


def _looks_complete(text: str) -> bool:
    """Heuristic: does this read like a finished summary, or narration cut off
    mid-write? Used only to decide whether a round-cap sub-agent needs a wrap-up
    summary round. Truncated narration ends without terminal punctuation and/or
    trails a "Now Edit N: ..." / "Next, ..." step it never finished. Bias toward
    treating short unpunctuated tails as truncated — a needless summary round is
    cheaper than delivering "...Now Edit 6: remove the unused var"."""
    t = (text or "").strip()
    if not t:
        return False
    _tail = t.rsplit("\n", 1)[-1].strip()
    # A finished thought ends on terminal punctuation or a closing fence/quote.
    if t[-1] in ".!?)]\"'`" or t.endswith("```"):
        return True
    # Otherwise it's an unpunctuated tail — likely a step announced but not done
    # (e.g. "Now Edit 6: remove the unused var"). Treat as incomplete.
    return False


def _send_wrapup_steer(queue_session: str, kill_reason: str, stall_s: int, timeout_s: int) -> bool:
    """Enqueue a single wind-down steer into a running sub-agent's own steer
    queue, for either a STALL (silence) or WALL (wall-clock backstop) trigger.
    Both are check-in nudges, not unconditional stop orders: the watchdog
    cancels the pending kill and fully resets its clock if genuine new
    activity (tool calls, real progress) lands during the grace window — a
    merely-quiet-but-working (or long-but-still-progressing) agent must not be
    told to abandon a task it's actually still doing. A LOOP trigger is
    different: it fires from inside the drain loop on a confirmed stuck
    pattern (the same exact tool call repeated), so that message stays an
    unconditional stop order. Returns True if it was queued (a live drain loop
    exists), False if the sub-agent's turn is already over (nothing to steer).
    The steer lands via agent_runs.enqueue_steer → the sub-agent loop's
    round-boundary / final drain (agent_loop._inject_steering_messages), same
    mechanism as send_to_subagent, so the worker sees it as an in-turn user
    message."""
    if kill_reason == "stall":
        msg = (
            f"[system] Check-in: you've produced no visible activity for {stall_s}s. "
            "If you are still genuinely working (about to call a tool, mid-thought "
            "on a real next step), just continue normally — this is not an order to "
            "stop. If you are actually stuck or done, write a final, self-contained "
            "plain-text summary of what you found/did and what remains, then END "
            "YOUR TURN — that summary is what gets reported back if you go quiet "
            "again, so make it count."
        )
    elif kill_reason == "wall":
        msg = (
            f"[system] Check-in: you've been running for {timeout_s}s, a very long "
            "time for one task. If you are still making genuine progress, continue "
            "normally — this is not an order to stop, just a routine check that "
            "you're not stuck. If you're actually done or blocked, write a final, "
            "self-contained plain-text summary of what you found/did and what "
            "remains, then END YOUR TURN — that summary is what gets reported back "
            "if you don't respond, so make it count."
        )
    else:
        _why = "you appear stuck in a loop and are being wound down"
        msg = (
            f"[system] Your run is being wound down — reason: {_why}. "
            "IMMEDIATELY STOP all tool work. Do NOT call any more tools. "
            "Write a final, self-contained plain-text summary of what you found / did "
            "and what remains, then END YOUR TURN. This summary is the only thing "
            "reported back, so make it count — you have only a few seconds."
        )
    try:
        from src import agent_runs
        return agent_runs.enqueue_steer(queue_session, msg, kind="user")
    except Exception as e:
        logger.debug("[subagent-wrapup] steer enqueue failed for %s: %s", queue_session, e)
        return False


async def spawn_agent(
    content: str,
    session_id: Optional[str] = None,
    owner: Optional[str] = None,
    parent_disabled: Optional[set] = None,
) -> Dict:
    """Spawn a sub-agent that runs a full tool-using agent loop on a task IN THE
    BACKGROUND and reports its result back into this chat when it finishes.

    This returns immediately with an acknowledgement — it does NOT block the
    current turn. The user (and you) can keep chatting while the sub-agent works;
    its final result is posted into this session as a new message on completion
    (or an error/timeout notice if it fails). Do not wait for or poll it.

    Content: the task/instructions for the sub-agent. Optional first line(s)
    ``model: <name>`` overrides the model (defaults to this chat's model) and
    ``timeout: <seconds>`` overrides the wall-clock backstop (clamped 60..21600;
    stall detection still guards the run regardless).

    Use for a self-contained unit of work you want done and reported back —
    e.g. "read js/render/models.js, screenshot localhost:1338, and list what
    looks visually wrong." The sub-agent has the normal tools (files, shell,
    browser) but cannot itself spawn more agents.

    You can STEER a running sub-agent mid-flight with
    ``send_to_subagent(subagent_id, message)`` — e.g. to narrow its focus, add a
    constraint, or nudge it — and check on it or cancel it with ``manage_agents``.
    """
    import json
    from src.agent_loop import stream_agent_loop
    from src.ai_interaction import get_session_manager, _resolve_model
    from src.tool_index import get_tool_index, ALWAYS_AVAILABLE

    task = (content or "").strip()
    if not task:
        return {"error": "No task provided for the sub-agent"}

    # Optional leading directives, parsed off the first line(s) the same way:
    #   model: <name>        — override the model (defaults to this chat's model)
    #   timeout: <seconds>   — override the wall-clock cap (clamped 60..3600)
    # Both may appear (in either order) before the task body; each is consumed
    # from the front, so the task text starts at the first non-directive line.
    model_spec = None
    timeout_override: Optional[int] = None
    while True:
        low = task.lower()
        if low.startswith("model:"):
            first, _, rest = task.partition("\n")
            model_spec = first.split(":", 1)[1].strip()
            task = rest.strip()
            if not task:
                return {"error": "Sub-agent task was empty after the model: line"}
        elif low.startswith("timeout:"):
            first, _, rest = task.partition("\n")
            _raw = first.split(":", 1)[1].strip()
            try:
                # Clamp to a sane range: below 60s a real coding sub-agent can't
                # get anything done; above 21600s (6h) a runaway ties up a slot far
                # too long. This overrides the wall-clock BACKSTOP for this spawn;
                # stall detection still guards it. Junk (non-int) is ignored.
                timeout_override = max(60, min(21600, int(_raw)))
            except ValueError:
                timeout_override = None
            task = rest.strip()
            if not task:
                return {"error": "Sub-agent task was empty after the timeout: line"}
        else:
            break

    # Default: inherit endpoint/model from the parent chat.
    url = model = headers = None
    sm = get_session_manager()
    parent = sm.get_session(session_id) if (sm and session_id) else None
    if parent is not None:
        url = getattr(parent, "endpoint_url", None)
        model = getattr(parent, "model", None)
        headers = getattr(parent, "headers", None)
        # The cached session object can lack a resolved endpoint/headers (model
        # race, stale cache) even when it has a model name. Resolve the name to a
        # full endpoint so inheriting the parent's model doesn't fail with
        # "Could not resolve a model" — the sub-agent should Just Work when the
        # caller omits an explicit model.
        if model and not (url and headers):
            try:
                url, model, headers = await asyncio.to_thread(_resolve_model, model, owner=owner)
            except ValueError:
                pass
    # Configured default sub-agent model. When the caller did NOT pin a model,
    # a `subagent_model` setting (e.g. a cheap DeepSeek for routine background
    # work) overrides the inherited parent model. This is what stops the model
    # from wasting rounds guessing/hallucinating a model name to assign — by
    # default it specifies NO model and the harness routes to this one.
    # Explicit `model:` still wins (e.g. a vision task where the default can't
    # see images). Falls through to the inherited parent if unset/unresolvable.
    if not model_spec:
        try:
            from src.settings import get_setting
            _default_sub = (get_setting("subagent_model", "") or "").strip()
        except Exception:
            _default_sub = ""
        if _default_sub:
            try:
                url, model, headers = await asyncio.to_thread(_resolve_model, _default_sub, owner=owner)
            except ValueError:
                pass  # keep the inherited parent model as fallback
    if model_spec:
        try:
            url, model, headers = await asyncio.to_thread(_resolve_model, model_spec, owner=owner)
        except ValueError as e:
            return {"error": str(e)}
    if not (url and model):
        return {"error": "Could not resolve a model for the sub-agent"}

    # Bind locals for the detached runner — spawn_agent returns before it runs,
    # so it must not close over anything that could be rebound afterwards.
    _url, _model, _headers, _owner, _task = url, model, headers, owner, task
    _parent_session = session_id
    _timeout_override = timeout_override   # per-spawn wall-clock cap, or None
    _summary = _task.splitlines()[0][:120] if _task else ""

    # Fallback chain if the primary sub-agent model fails (rate-limit / spend
    # cap / provider error). Prefer a sub-agent-specific chain when the user
    # configured one in Settings → Sub-agents (subagent_model_fallbacks);
    # otherwise reuse the main chat's chain. Best-effort — empty = no fallback.
    try:
        from src.endpoint_resolver import (
            resolve_subagent_fallback_candidates, resolve_chat_fallback_candidates,
        )
        from src.settings import get_setting as _gs
        _sub_fb_cfg = _gs("subagent_model_fallbacks", []) or []
        if _sub_fb_cfg:
            _fallbacks = await asyncio.to_thread(resolve_subagent_fallback_candidates, _owner)
        else:
            _fallbacks = await asyncio.to_thread(resolve_chat_fallback_candidates, _owner)
    except Exception:
        _fallbacks = []

    # Inherit the parent turn's active workspace so the sub-agent's file/shell
    # tools are confined to (and resolve "the project" as) the same folder the
    # user is working in. spawn_agent runs inside the parent's tool-execution
    # context, where the workspace contextvar is set; capture it now, before the
    # detached runner starts.
    try:
        from src.tool_execution import get_active_workspace
        _workspace = get_active_workspace()
    except Exception:
        _workspace = None

    # Pre-select the sub-agent's tools with the coding baseline force-included, so
    # a worker is never dispatched without file/shell/search tools just because
    # its task wording didn't lexically retrieve them (the root cause of sub-
    # agents reporting themselves "blocked — no filesystem tools"). RAG still runs
    # on the task on top of the baseline, so task-specific tools are added too.
    # Effective disabled set = the leaf-worker baseline PLUS whatever the PARENT
    # turn had disabled. A child must never regain a capability the parent turn
    # withheld: the route computes disabled_tools per turn (global setting, chat-
    # mode escalation dropping bash/python/read_file/write_file, compare mode,
    # code-execution-off) and spawn_agent is ALWAYS_AVAILABLE, so without this a
    # parent with bash disabled could spawn a child WITH bash. (#security)
    _disabled = set(_SUBAGENT_DISABLED) | set(parent_disabled or ())
    _sub_tools = set(ALWAYS_AVAILABLE) | set(_SUBAGENT_TOOL_BASELINE)
    try:
        _tool_idx = get_tool_index()
        if _tool_idx:
            _sub_tools = await asyncio.to_thread(
                _tool_idx.get_tools_for_query, _task, 8, _sub_tools
            )
    except Exception as _tsel_err:
        logger.debug("sub-agent tool pre-selection fell back to baseline: %s", _tsel_err)
    _sub_tools -= _disabled

    # Clamp the per-round output cap to what the model can actually accept. A model
    # with a small context window (e.g. 8K) 400s every round on an unclamped 12K
    # output request. Only clamp when the window is PROVEN (endpoint-reported /
    # known table) — an unknown fallback isn't evidence the model is small, so we
    # keep the generous default. Reserve headroom for the prompt/history.
    _max_tokens = _SUBAGENT_MAX_TOKENS
    try:
        from src.model_context import get_context_length_known
        _ctx_len, _ctx_known = await asyncio.to_thread(get_context_length_known, _url, _model)
        if _ctx_known and _ctx_len:
            # Never ask for more output than ~half the window (leaving room for the
            # task + tool results), and never exceed the window itself.
            _max_tokens = min(_max_tokens, max(1024, _ctx_len // 2))
    except Exception as _clamp_err:
        logger.debug("sub-agent max_tokens clamp fell back to default: %s", _clamp_err)

    # Populated with the tracked run's id + ephemeral steer-queue session right
    # after subagent_runs.start() returns, so _deliver (server-side resume) and
    # _run_subagent (mid-flight steering) can read them by id. A plain dict works
    # because both read it at call time, well after start() has filled it in.
    _run_meta: Dict[str, Optional[str]] = {"id": None, "queue_session": None}
    # Holds the subagent_runs run record (a dict LIVE in that module's _UPDATES
    # store — mutating it here is visible to get_updates() with no extra
    # plumbing) once start() returns, so _drain's per-chunk handler can push
    # rounds_used/last_tool/last_activity_at/output_tail into it for a parent
    # checking in mid-run via manage_agents. Same call-time-not-definition-time
    # closure timing as _run_meta above: empty (never populated) on the
    # synchronous/no-session fallback, since there is no run record there.
    _rec_holder: Dict[str, Optional[dict]] = {"rec": None}

    def _deliver(error: Optional[str], partial: str) -> None:
        """Post the sub-agent's outcome as an assistant message into the parent
        session. Persists immediately (add_message -> _persist_message commits),
        so it renders on reload even if no client is currently connected.

        Server-side resume (Claude-Code parity): after persisting the outcome,
        ensure the parent model actually processes it, server-side — no open
        browser required:
          * If the parent session has a LIVE agent run, enqueue a steering entry
            (framed as untrusted context) so the running turn reacts next round.
          * Otherwise start a DETACHED main-agent resume turn in the parent
            session (registered in agent_runs so clients can attach/see it).

        FAILURE also resumes now (this fixes the hang-forever bug: a parent that
        spawned a subagent and ended its turn would wait forever if that subagent
        timed out, because failures never woke it). The failure wake is framed to
        say the subagent FAILED and why, so the parent can decide the next step —
        but it is guarded HARD against a spawn/fail/resume token loop: a tighter
        per-session failure cap (subagent_runs.can_failure_resume), and it is
        SKIPPED entirely when the reason looks like a provider/credits error (the
        exact concern of the original "never resume on error" comment).
        """
        if not _parent_session:
            return
        try:
            from core.models import ChatMessage
            sm2 = get_session_manager()
            logger.info("[subagent-deliver] session=%s error=%r partial_len=%d sm=%s",
                        _parent_session, error, len(partial or ""), "yes" if sm2 else "NONE")
            if not sm2:
                logger.error("[subagent-deliver] no session manager — result LOST for %s", _parent_session)
                return
            sess2 = sm2.get_session(_parent_session)
            if error:
                text = f"**Sub-agent failed** ({_model})\n\n{error}"
                if partial:
                    text += f"\n\nPartial output before it stopped:\n\n{partial}"
            else:
                text = f"**Sub-agent result** ({_model})\n\n" + (
                    partial or "(sub-agent produced no text output)"
                )
            sess2.add_message(ChatMessage("assistant", text, metadata={"model": _model, "subagent": True}))
            logger.info("[subagent-deliver] delivered to %s (%d chars)", _parent_session, len(text))
        except Exception as e:
            logger.error(f"spawn_agent delivery failed for session {_parent_session}: {e}", exc_info=True)
            return

        # Successful completions always drive a resume. Failures ALSO resume (so a
        # waiting parent is woken instead of hanging forever) EXCEPT when the reason
        # is a provider/credits problem — resuming then would just re-hit the wall
        # and burn tokens in a spawn/fail/resume loop (the original concern).
        if error and _is_provider_error(error):
            logger.info("[subagent-resume] provider-type failure for %s — delivered only, no resume", _parent_session)
            return
        try:
            _maybe_server_resume(partial or "", failed=bool(error), reason=error)
        except Exception as e:
            logger.error("[subagent-resume] resume dispatch failed for %s: %s", _parent_session, e, exc_info=True)

    def _maybe_server_resume(result_text: str, failed: bool = False, reason: Optional[str] = None) -> None:
        """Push the sub-agent's outcome to the parent model, server-side.

        Live turn → enqueue a steering entry (Feature 1's queue). No live turn →
        start a detached resume turn. Both frame the content via
        untrusted_context_message so fetched/quoted content can't speak with
        user/assistant authority. Capped + single-fire guarded via subagent_runs.

        ``failed`` frames the wake as a FAILURE notice (subagent timed out / got
        stuck / errored) and routes it through the tighter failure cap
        (can_failure_resume) instead of the success cap, so repeated failures can't
        ping-pong the parent.
        """
        from src import agent_runs, subagent_runs
        from src.prompt_security import untrusted_context_message

        sub_id = _run_meta.get("id")
        # Frame the outcome as untrusted context (may quote fetched web content).
        if failed:
            framed = untrusted_context_message(
                f"background sub-agent {sub_id or ''} ({_model}) FAILED",
                f"Background sub-agent {sub_id or ''} ({_model}) did NOT complete its task. "
                f"Reason: {reason or 'unknown failure'}.\n\n"
                f"Any partial output / activity it produced before stopping:\n\n{result_text or '(none)'}\n\n"
                "Decide the next step yourself: retry with a narrower task or a different "
                "model, do the work directly, or tell the user it could not be completed. "
                "Do NOT simply re-spawn the same task unchanged.",
            )["content"]
        else:
            framed = untrusted_context_message(
                f"background sub-agent {sub_id or ''} ({_model}) result",
                f"Background sub-agent {sub_id or ''} ({_model}) finished. Full result:\n\n{result_text}",
            )["content"]

        # 1) A turn is live for the parent session → steer into it (atomic:
        #    enqueue only succeeds if the run is still running). No cap needed: a
        #    live turn consumes the steer as one message, no new turn is spawned.
        if agent_runs.enqueue_steer(_parent_session, framed, kind="subagent"):
            if sub_id:
                subagent_runs.mark_resume(_parent_session, sub_id, "server")
            logger.info("[subagent-resume] steered %s into live turn for %s",
                        "FAILURE" if failed else "result", _parent_session)
            return

        # 2) No live turn → start a detached server-side resume turn, subject to
        #    the cap (failure path uses the tighter failure cap) and single-fire guard.
        _can = subagent_runs.can_failure_resume if failed else subagent_runs.can_server_resume
        if not _can(_parent_session):
            logger.info("[subagent-resume] resume capped/already-running for %s — delivered only", _parent_session)
            return
        if failed:
            subagent_runs.note_failure_resume(_parent_session)
        else:
            subagent_runs.note_server_resume(_parent_session)
        subagent_runs.set_resume_running(_parent_session, True)
        if sub_id:
            subagent_runs.mark_resume(_parent_session, sub_id, "server")

        async def _resume_turn() -> None:
            try:
                from src.chat_flows import start_server_resume_turn
                await start_server_resume_turn(_parent_session, framed, owner=_owner)
            except Exception as _e:
                logger.error("[subagent-resume] detached resume turn failed for %s: %s",
                             _parent_session, _e, exc_info=True)
            finally:
                subagent_runs.set_resume_running(_parent_session, False)

        try:
            asyncio.create_task(_resume_turn())
            logger.info("[subagent-resume] started detached server resume turn for %s", _parent_session)
        except Exception as _e:
            subagent_runs.set_resume_running(_parent_session, False)
            logger.error("[subagent-resume] could not schedule resume turn for %s: %s", _parent_session, _e)

    async def _summary_round(truncated: str, tools: int, last_tool: Optional[str]) -> str:
        """One final tool-free round so a sub-agent that burned all its rounds on
        tool calls still delivers a real summary instead of raw truncated
        narration. Mirrors the main loop's force-answer round (agent_loop.py:
        _force_answer -> tools=[]); here we get the same effect by running a
        fresh stream_agent_loop with an empty relevant-tools set and every tool
        disabled, so the model can only write prose. Returns the summary text, or
        "" if the round produced nothing (caller falls back to a synth header)."""
        _bits: list = []
        _prompt = (
            "You are wrapping up a background task you were already working on. You have "
            "hit your tool-round limit, so you can no longer call tools — write your final "
            "summary now.\n\n"
            f"Original task:\n{_task}\n\n"
            f"You ran {tools} tool call(s)"
            + (f" (last: {last_tool})" if last_tool else "")
            + ". Your work-in-progress notes so far were:\n"
            f"{truncated or '(no narration captured)'}\n\n"
            "In a few sentences, state plainly what you COMPLETED and what (if anything) "
            "REMAINS. Do not ask questions; do not call tools. This summary is the only "
            "thing reported back."
        )
        try:
            async def _drain_summary():
                async for _chunk in stream_agent_loop(
                    _url, _model,
                    [
                        {"role": "system", "content": _SUBAGENT_SYSTEM_PROMPT},
                        {"role": "user", "content": _prompt},
                    ],
                    headers=_headers,
                    owner=_owner,
                    session_id=None,
                    disabled_tools=set(_disabled) | set(_sub_tools),  # bar every tool
                    relevant_tools=set(),                              # no tools offered
                    workspace=_workspace,
                    max_rounds=1,
                    max_tokens=_max_tokens,
                    fallbacks=_fallbacks,
                ):
                    if _chunk.startswith("data: ") and not _chunk.startswith("data: [DONE]"):
                        try:
                            _d = json.loads(_chunk[6:])
                        except Exception:
                            continue
                        if "delta" in _d and not _d.get("thinking"):
                            _bits.append(_d["delta"])
            # Bound the wrap-up so a wedged summary round can't hang the run.
            await asyncio.wait_for(_drain_summary(), timeout=_subagent_timeout())
        except Exception as _sum_err:
            logger.warning("[subagent-run] summary round failed: %s", _sum_err)
            return ""
        _txt = "".join(_bits).strip()
        if _txt == "The model returned an empty response. Please try again or switch to a different model.":
            return ""
        return _txt

    async def _run_subagent(deliver: bool) -> Dict:
        """Run the leaf sub-agent to completion under the guardrails. When
        ``deliver`` is set (background mode) the result is also posted into the
        parent session; otherwise it is only returned (synchronous fallback)."""
        collected: list = []
        # Compact activity log of the tool calls seen, so a sub-agent that spends
        # all its time in tools (and delivers partial_len=0 text) can still report
        # "what it did before timing out". Capped in size below.
        activity: list = []
        # "rounds" starts at 1 (stream_agent_loop's own round numbering is
        # 1-based) and is bumped on each agent_step event, so a parent checking
        # in mid-run via manage_agents sees genuine progress, not just "still
        # running" with no sense of how far along it is.
        stats = {"tools": 0, "hit_cap": False, "error": None, "last_tool": None, "rounds": 1}
        # Watchdog scratch: last time ANY chunk arrived (stall detection), and the
        # consecutive-identical-tool-call run (loop detection). Mutable containers
        # so the nested _drain + the watchdog loop share them by reference.
        wd = {
            "last_activity": None,      # set to a loop-clock stamp on first chunk
            "loop_sig": None,           # signature of the last tool call
            "loop_count": 0,            # how many times it has repeated in a row
            "loop_tool": None,          # the tool name that is looping (for the msg)
            "wrapup_sent": False,       # one graceful wind-down steer, max, per run
        }

        def _note_tool_activity(tool: str, cmd: str) -> None:
            """Record a tool call in the activity log (capped) and update the
            consecutive-repeat counter used for loop detection."""
            if len(activity) < 60:  # cap entries; each line is trimmed below too
                _line = tool if not cmd else f"{tool}: {cmd}"
                activity.append(_line[:200])
            sig = f"{tool}\n{cmd}"
            if sig == wd["loop_sig"]:
                wd["loop_count"] += 1
            else:
                wd["loop_sig"] = sig
                wd["loop_count"] = 1
                wd["loop_tool"] = tool

        def _activity_log(limit: int = 3000) -> str:
            """The captured activity as a compact, size-bounded block for the
            delivered failure message. Empty when nothing ran."""
            if not activity:
                return ""
            body = "\n".join(f"  - {a}" for a in activity)
            if len(body) > limit:
                body = body[:limit] + "\n  … (activity log truncated)"
            return body

        # Ephemeral steer-queue session for THIS sub-agent (set by start()).
        # Threading it as session_id gives the loop a steer queue so the parent
        # can send_to_subagent() into a running worker; the queue is a bare
        # mailbox in agent_runs and a NON-PERSISTING ephemeral session in the
        # manager, so no chat rows are written under it (see subagent_runs). None
        # in the synchronous no-session fallback (no run record, no mid-flight
        # steering) — steering only applies to background runs.
        _queue_session = _run_meta.get("queue_session")

        async def _drain():
            import time as _time
            async for chunk in stream_agent_loop(
                _url, _model,
                [
                    {"role": "system", "content": _SUBAGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": _task},
                ],
                headers=_headers,
                owner=_owner,
                session_id=_queue_session,               # steer-queue mailbox (not persisted)
                disabled_tools=set(_disabled),           # leaf worker: no recursion, plus
                                                         # the parent turn's disabled policy
                relevant_tools=set(_sub_tools),          # coding baseline force-included
                workspace=_workspace,                    # inherit parent's project folder
                max_rounds=_subagent_max_rounds(),
                max_tokens=_max_tokens,                  # the default 4096 truncated long
                                                         # final summaries mid-word; clamped
                                                         # above to the model's real ceiling
                fallbacks=_fallbacks,
            ):
                # Any chunk = the sub-agent is alive; feed the stall watchdog.
                wd["last_activity"] = asyncio.get_event_loop().time()
                # Capture a real upstream failure (e.g. 429 rate-limit / spend cap)
                # so we can report WHY instead of the generic "empty response".
                if chunk.startswith("event: error"):
                    for _line in chunk.split("\n"):
                        if _line.startswith("data: "):
                            try:
                                _ed = json.loads(_line[6:])
                                stats["error"] = str(_ed.get("text") or _ed.get("error") or "upstream error")[:400]
                            except Exception:
                                stats["error"] = "upstream error"
                    continue
                if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                    try:
                        d = json.loads(chunk[6:])
                    except Exception:
                        continue
                    _t = d.get("type")
                    if _t == "tool_start":
                        stats["tools"] += 1
                        # Remember the last tool that ran so the round-cap
                        # summary/header can name it (see the exhaustion path).
                        _tool = d.get("tool")
                        if _tool:
                            stats["last_tool"] = _tool
                        # Log it + update loop detection. `command` is the compact
                        # display form of the args the loop already computes.
                        _note_tool_activity(_tool or "tool", str(d.get("command") or ""))
                        # Conservative loop guard: only EXACT consecutive repeats
                        # of the same tool+args. Raise a distinct error so the
                        # watchdog can kill with a "stuck in a loop" reason.
                        if wd["loop_count"] >= _SUBAGENT_LOOP_REPEATS:
                            raise _SubagentLoopError(wd["loop_tool"] or "a tool", wd["loop_count"])
                    elif _t == "rounds_exhausted":
                        stats["hit_cap"] = True
                    elif _t == "agent_step":
                        # agent_step's round is the round about to START (see
                        # stream_agent_loop's own docstring/emission sites), so
                        # this is exactly "rounds used so far" from the
                        # parent's point of view.
                        _rnd = d.get("round")
                        if isinstance(_rnd, int) and _rnd > stats["rounds"]:
                            stats["rounds"] = _rnd
                    # Accumulate visible answer text only (skip thinking tokens).
                    if "delta" in d and not d.get("thinking"):
                        collected.append(d["delta"])
                    # Push live progress into the shared run record (if any) so
                    # a parent calling manage_agents mid-run sees genuine status
                    # — rounds used, last tool, seconds since activity, and a
                    # tail of the most recent output — not just "still
                    # running". _rec_holder is populated once subagent_runs
                    # .start() returns (see below); empty before that (the
                    # synchronous/no-session fallback never populates it).
                    _rec = _rec_holder.get("rec")
                    if _rec is not None:
                        _rec["rounds_used"] = stats["rounds"]
                        _rec["last_tool"] = stats["last_tool"]
                        _rec["last_activity_at"] = _time.time()
                        _tail_text = "".join(collected)
                        if _tail_text:
                            _rec["output_tail"] = _tail_text[-300:]

        error = None
        # PRIMARY guard is stall/silence detection. The wall-clock cap is only a
        # generous SAFETY BACKSTOP: the per-spawn `timeout:` directive wins, else the
        # configured/default cap; a value of 0 (from a 0/negative setting) means the
        # backstop is DISABLED and only stall detection bounds the run.
        _stall_s = _subagent_stall_timeout()
        _timeout_s = _timeout_override if _timeout_override else _subagent_timeout()
        # Cap bash INSIDE this sub-agent so one blocking foreground command (a dev
        # server the model forgot to background) can't sit silent up to the stall
        # window. Keyed off the stall window (the primary guard), not the wall-clock
        # backstop which may be huge or disabled. Bound to this task's context.
        from src.agent_tools import subprocess_tools as _subproc
        _bash_tok = _subproc.set_subagent_bash_timeout(_subagent_bash_cap(_stall_s))
        try:
            async with _SUBAGENT_SEMAPHORE:
                # Watchdog: run the drain as a task and poll it. Kill (cancel) it on
                # (a) SILENCE for _stall_s — the primary guard: a hung/blocking
                # command; or (b) total runtime past the wall-clock BACKSTOP (unless
                # disabled). Loop detection kills from inside _drain. This beats a
                # blind wait_for: a hang dies ~a stall-window after it wedges instead
                # of running to the backstop, and the kill reasons are distinct below.
                _loop = asyncio.get_event_loop()
                _start = _loop.time()
                wd["last_activity"] = _start
                _drain_task = asyncio.ensure_future(_drain())
                _poll = min(5.0, max(1.0, _stall_s / 4))
                _kill_reason = None
                # Grace phase state: once a kill trigger fires we may FIRST steer
                # the worker to wrap up (write a final summary) and give it a grace
                # window to comply before hard-killing. _grace_deadline is set when
                # the wind-down steer goes out; while it is set the normal
                # stall/wall triggers are suspended (we're deliberately waiting on
                # the worker) and only the deadline — or the worker finishing —
                # ends the wait. One wrap-up attempt max (wd["wrapup_sent"]).
                _grace_deadline = None          # loop-clock time to hard-kill at
                _grace_last_activity = None     # activity stamp when grace started
                try:
                    while True:
                        done, _ = await asyncio.wait({_drain_task}, timeout=_poll)
                        if done:
                            _drain_task.result()   # re-raise any drain exception
                            break
                        _now = _loop.time()
                        if _grace_deadline is not None:
                            # In the graceful wind-down window. If the worker
                            # produced genuinely NEW activity since the wind-down
                            # steer went out, it woke up and is doing real work
                            # again — a nudged-awake agent is healthy, not one to
                            # kill. Abort the pending kill entirely (not just note
                            # it for the final message) and fully reset the stall
                            # clock so it gets a complete fresh window, exactly as
                            # if it had never stalled. Re-arm the wrap-up nudge too
                            # (wrapup_sent=False) so a LATER stall gets its own
                            # steer instead of silently hard-killing next time.
                            if wd["last_activity"] and wd["last_activity"] > _grace_last_activity:
                                logger.info(
                                    "[subagent-wrapup] %s: new activity during grace — "
                                    "kill aborted, stall clock reset",
                                    _queue_session,
                                )
                                _grace_deadline = None
                                _grace_last_activity = None
                                _kill_reason = None
                                wd["wrapup_sent"] = False
                                continue
                            # Still silent since the steer went out — _drain()
                            # timestamps last_activity on EVERY chunk (deltas,
                            # tool events, the final summary text alike), so any
                            # real output at all would already have re-armed the
                            # branch above. Nothing new yet: keep waiting out the
                            # deadline. (A STALL kill uses a short deadline
                            # precisely because a wedged worker won't produce new
                            # activity to justify the wait.)
                            if _now < _grace_deadline:
                                continue
                            # Grace expired with no new activity → fall through to
                            # hard-kill with the ORIGINAL reason recorded before
                            # the steer.
                        else:
                            # (a) Primary: silence for the whole stall window.
                            if _now - (wd["last_activity"] or _start) > _stall_s:
                                _kill_reason = "stall"
                            # (b) Backstop: only when enabled (_timeout_s > 0).
                            elif _timeout_s and _now - _start > _timeout_s:
                                _kill_reason = "wall"
                            if not _kill_reason:
                                continue
                            # A trigger fired. Try ONE check-in steer before
                            # killing: for stall/wall this asks the worker to
                            # either keep going (if it's genuinely still
                            # working) or wrap up with a final summary, then
                            # wait a grace window — new activity during that
                            # window cancels the kill outright (see above).
                            _grace = _subagent_wrapup_grace(_kill_reason)
                            if _grace and not wd["wrapup_sent"] and _queue_session:
                                if _send_wrapup_steer(_queue_session, _kill_reason, _stall_s, _timeout_s):
                                    wd["wrapup_sent"] = True
                                    _grace_deadline = _now + _grace
                                    _grace_last_activity = wd["last_activity"]
                                    logger.info(
                                        "[subagent-wrapup] %s: wind-down steer sent (reason=%s), grace=%ds",
                                        _queue_session, _kill_reason, _grace,
                                    )
                                    continue   # give the worker the grace window
                            # Wrap-up disabled / already attempted / no queue /
                            # enqueue failed (turn already ended) → hard-kill now.
                        # Hard-kill path (trigger with no grace, or grace expired).
                        _drain_task.cancel()
                        try:
                            await _drain_task
                        except (asyncio.CancelledError, Exception):
                            pass
                        break
                    if _kill_reason:
                        if not _drain_task.done():
                            _drain_task.cancel()
                            try:
                                await _drain_task
                            except (asyncio.CancelledError, Exception):
                                pass
                        _woke = wd["wrapup_sent"] and _grace_last_activity != wd["last_activity"]
                        _wrap_note = (
                            " (wrote a wrap-up summary before stopping)" if _woke
                            else (" (did not respond to the wind-down request)" if wd["wrapup_sent"] else "")
                        )
                        if _kill_reason == "stall":
                            error = (
                                f"Sub-agent stalled: no activity for {_stall_s}s "
                                f"(likely stuck on a blocking command){_wrap_note}"
                            )
                        else:
                            error = f"Sub-agent hit the {_timeout_s}s wall-clock backstop{_wrap_note}"
                except _SubagentLoopError as _le:
                    # Drain aborted itself: same tool call repeated too many times.
                    error = (
                        f"Sub-agent appears stuck in a loop (repeated identical "
                        f"{_le.tool} calls x{_le.count}) — stopped it"
                    )
                finally:
                    if not _drain_task.done():
                        _drain_task.cancel()
                        try:
                            await _drain_task
                        except (asyncio.CancelledError, Exception):
                            pass
        except asyncio.CancelledError:
            # Deliver whatever it produced before cancellation, then propagate so
            # the run manager records it as stopped/error and cleans up. Include the
            # activity log so a tool-only worker's "(nothing text)" still shows work.
            if deliver:
                _partial = "".join(collected).strip()
                _log = _activity_log()
                if _log:
                    _partial = (_partial + "\n\nWhat it did before it stopped:\n" + _log).strip()
                _deliver(error="Sub-agent was cancelled", partial=_partial[:_SUBAGENT_RESULT_CAP])
            raise
        except Exception as e:
            logger.error(f"spawn_agent run failed: {e}")
            error = f"Sub-agent failed: {e}"
        finally:
            _subproc.reset_subagent_bash_timeout(_bash_tok)

        result = "".join(collected).strip()
        # The loop's generic empty-response placeholder is not a real answer — drop
        # it so the real reason (captured below) surfaces instead.
        if result == "The model returned an empty response. Please try again or switch to a different model.":
            result = ""

        # Round-cap wrap-up: a sub-agent that spent all its rounds on tool calls
        # delivers whatever text happened to accumulate — often truncated
        # mid-edit ("...Now Edit 6: remove the unused var"), so the parent/user
        # never learn the edits succeeded. When it hit the cap with pending work,
        # run ONE tool-free summary round and deliver THAT instead. If the summary
        # round yields nothing, prepend a synthesized header so the truncation is
        # at least labelled. (Skipped on error paths — those report their own
        # reason; the empty+cap case below already explains itself.)
        if stats["hit_cap"] and not error and result and not _looks_complete(result):
            _summary = await _summary_round(result, stats["tools"], stats["last_tool"])
            if _summary:
                result = _summary
            else:
                _hdr = (
                    f"(Hit the {_subagent_max_rounds()}-round limit; {stats['tools']} tool "
                    f"call(s) ran"
                    + (f", last: {stats['last_tool']}" if stats["last_tool"] else "")
                    + ". Work happened but the summary was cut off mid-write.)\n\n"
                )
                result = _hdr + result

        # Cap generously so the PARENT model receives the sub-agent's FULL answer
        # (the delivered message IS the parent's context on auto-resume); only
        # genuinely huge outputs get trimmed, with a clear marker.
        if len(result) > _SUBAGENT_RESULT_CAP:
            result = result[:_SUBAGENT_RESULT_CAP] + f"\n\n… [sub-agent result truncated at {_SUBAGENT_RESULT_CAP} chars]"
        # Informative fallback when the model wrote no final answer — otherwise the
        # user just sees "(no text output)" with no idea why (the #1 bizarre case).
        if not result and not error:
            if stats["error"]:
                error = f"the sub-agent's model failed — {stats['error']}"
            elif stats["hit_cap"]:
                result = (
                    f"(The sub-agent ran {stats['tools']} tool call(s) but hit its "
                    f"{_subagent_max_rounds()}-round limit before writing a summary. "
                    "Try a narrower task, or spawn it with an explicit fast model.)"
                )
            elif stats["tools"]:
                result = (
                    f"(The sub-agent ran {stats['tools']} tool call(s) but produced no "
                    "written summary of what it found.)"
                )
            else:
                result = "(The sub-agent produced no output.)"
        # On any failure path (timeout / stall / loop / upstream error), a
        # tool-working sub-agent often has partial_len=0 visible text — so attach
        # the compact activity log ("What it did before it stopped") to the partial
        # delivered alongside the failure notice. This is the salvage that makes a
        # killed-mid-work run still report the tool steps it completed.
        if error:
            _log = _activity_log()
            if _log:
                _work = "What it did before it stopped:\n" + _log
                result = (result + "\n\n" + _work).strip() if result else _work
                if len(result) > _SUBAGENT_RESULT_CAP:
                    result = result[:_SUBAGENT_RESULT_CAP] + f"\n\n… [truncated at {_SUBAGENT_RESULT_CAP} chars]"

        logger.info("[subagent-run] finished session=%s deliver=%s error=%r tools=%d cap=%s result_len=%d",
                    _parent_session, deliver, error, stats["tools"], stats["hit_cap"], len(result))
        if deliver:
            _deliver(error=error, partial=result)
        if error:
            return {"error": error, "partial_result": result[:4000]}
        return {"model": _model, "result": result or "(sub-agent produced no text output)"}

    # No parent session to deliver into (e.g. a direct/ephemeral call): fall back
    # to synchronous execution so the result is still returned to the caller.
    if not _parent_session:
        return await _run_subagent(deliver=False)

    # Background mode: register a detached run and return immediately so the main
    # turn is not blocked. The result is delivered into this session on completion.
    from src import subagent_runs
    if not subagent_runs.can_start(_owner):
        return {
            "error": (
                f"Too many sub-agents already running (per-owner limit "
                f"{subagent_runs._MAX_PER_OWNER}, global {subagent_runs._MAX_TOTAL}). "
                "Wait for one to finish before spawning another."
            )
        }
    rec = subagent_runs.start(
        _parent_session, _summary, _model, lambda: _run_subagent(deliver=True), owner=_owner
    )
    # Let _deliver flag this run for server-side resume by id when it finishes,
    # and let _run_subagent thread the steer-queue session into the loop.
    _run_meta["id"] = rec["id"]
    _run_meta["queue_session"] = rec.get("queue_session")
    # Let _drain's per-chunk handler push live progress into this SAME record
    # object (it lives in subagent_runs._UPDATES) so manage_agents can report
    # genuine mid-run status to the parent.
    _rec_holder["rec"] = rec
    return {
        "result": (
            f"Sub-agent dispatched (id={rec['id']}, model={_model}) and now running in the "
            "background.\n\n"
            "YOUR WORK FOR THIS REQUEST IS DONE. Do NOT call any more tools. Do NOT spawn "
            "another agent. Do NOT start doing the sub-agent's task yourself. The sub-agent "
            "will post its own result into this chat when it finishes, and that does not "
            "require your turn to stay open. (You can call `manage_agents` any time to "
            "check what's running or cancel it — but do NOT poll it in a loop.)\n\n"
            "Now reply to the user with ONE short sentence saying the agent is working, then "
            "STOP — end your turn immediately."
        ),
        "background": True,
        "subagent_id": rec["id"],
    }


async def list_models(content: str, session_id: Optional[str] = None, owner: Optional[str] = None) -> Dict:
    """List all available models across configured endpoints.

    Content = optional filter keyword.
    """
    import json
    import httpx
    from src.database import SessionLocal, ModelEndpoint
    from src.llm_core import _detect_provider, ANTHROPIC_MODELS
    from src.auth_helpers import owner_filter
    from src.endpoint_resolver import resolve_endpoint_runtime, build_headers, build_models_url

    keyword = content.strip().lower() if content.strip() else None

    db = SessionLocal()
    try:
        query = db.query(ModelEndpoint).filter(ModelEndpoint.is_enabled == True)
        if owner:
            query = owner_filter(query, ModelEndpoint, owner)
        endpoints = query.all()
        if not endpoints:
            return {"results": "No enabled model endpoints configured."}

        result_lines = []
        total_models = 0

        for ep in endpoints:
            try:
                base, api_key = resolve_endpoint_runtime(ep, owner=owner)
            except Exception:
                continue
            provider = _detect_provider(base)
            headers = build_headers(api_key, base)

            model_ids = []
            if provider == "anthropic":
                model_ids = list(ANTHROPIC_MODELS)
            else:
                try:
                    models_url = build_models_url(base)
                    if models_url:
                        r = httpx.get(models_url, headers=headers, timeout=5)
                        r.raise_for_status()
                        data = r.json()
                        model_ids = [m.get("id") for m in (data.get("data") or []) if m.get("id")]
                        if not model_ids:
                            model_ids = [
                                m.get("name") or m.get("model")
                                for m in (data.get("models") or [])
                                if m.get("name") or m.get("model")
                            ]
                    else:
                        model_ids = json.loads(ep.cached_models or "[]")
                except Exception:
                    model_ids = ["(endpoint offline)"]

            # Normalize to strings before any filtering/rendering. cached_models
            # (and, defensively, a malformed /models payload) can carry None or
            # non-string entries; the keyword filter's `m.lower()` would raise
            # AttributeError on those, and the except-guard turns a single bad
            # entry into a whole-tool failure ({"error": ...}) that hides every
            # other endpoint's models. Coerce + drop empties so one junk row
            # can't take down the listing.
            model_ids = [str(m).strip() for m in model_ids if m]

            if keyword:
                _ep_name = (ep.name or "").lower()
                model_ids = [m for m in model_ids if keyword in m.lower() or keyword in _ep_name]

            if model_ids:
                result_lines.append(f"\n**{ep.name or base}** ({provider}):")
                for mid in model_ids:
                    result_lines.append(f"  - `{mid}`")
                    total_models += 1

        if not result_lines:
            return {"results": "No models found" + (f" matching '{keyword}'" if keyword else "") + "."}

        header = f"Available models ({total_models} total):"
        return {"results": header + "\n".join(result_lines)}
    except Exception as e:
        logger.error(f"list_models failed: {e}")
        return {"error": str(e)}
    finally:
        db.close()


async def manage_agents(content: str, session_id: Optional[str] = None, owner: Optional[str] = None) -> Dict:
    """See which background sub-agents are running in THIS chat, or cancel one.

    Content:
      (empty) or "list"      → list running / recently-finished sub-agents
      "stop <id>" / "cancel <id>" → cancel a running sub-agent by its id (e.g. sub_3)

    Use this to check on work you dispatched with spawn_agent — like a manager
    checking in on a report: each running entry shows rounds used so far,
    seconds since its last activity, the last tool it called, and a tail of
    its most recent output, so you can judge "it's going fine" vs. deciding to
    steer it (``send_to_subagent``) or stop it (this tool, ``stop <id>``).
    """
    import time
    from src import subagent_runs

    if not session_id:
        return {"error": "manage_agents can only be used inside a chat session."}

    action = (content or "").strip()
    low = action.lower()

    if low.startswith("stop") or low.startswith("cancel"):
        parts = action.split(None, 1)
        sub_id = parts[1].strip() if len(parts) > 1 else ""
        if not sub_id:
            return {"error": "Which sub-agent? Use: manage_agents stop <id> (e.g. stop sub_3)."}
        stopped = subagent_runs.stop(session_id, sub_id)
        if stopped:
            return {"results": f"Cancelling sub-agent {sub_id}. It will post a cancellation notice with any partial output into this chat."}
        return {"results": f"No running sub-agent {sub_id} found (it may have already finished — check the chat for its result)."}

    # Default: list. Running entries carry live progress (rounds used, seconds
    # since activity, last tool, a tail of recent output) so the parent model
    # can genuinely assess "is it going fine?" mid-run instead of only seeing
    # "running Ns" with no sense of progress — like a manager checking in.
    upd = subagent_runs.get_updates(session_id)
    now = upd.get("now") or time.time()
    running, finished = [], []
    for u in upd.get("updates", []):
        el = int(max(0, now - (u.get("started_at") or now)))
        summ = (u.get("summary") or "").strip()
        model = u.get("model") or "?"
        if u.get("status") == "running":
            _bits = [f"  - {u['id']} [{model}] running {el}s"]
            _rounds = u.get("rounds_used")
            if _rounds is not None:
                _bits.append(f"round {_rounds}")
            _since = u.get("seconds_since_activity")
            if _since is not None:
                _bits.append(f"last activity {_since}s ago")
            _last_tool = u.get("last_tool")
            if _last_tool:
                _bits.append(f"last tool: {_last_tool}")
            _line = ", ".join(_bits) + f" — {summ}"
            _tail = (u.get("output_tail") or "").strip()
            if _tail:
                _line += f"\n    recent output: …{_tail[-300:]}"
            running.append(_line)
        else:
            st = u.get("status")
            if u.get("error"):
                st = f"{st}: {u['error']}"
            finished.append(f"  - {u['id']} [{model}] {st} — {summ}")
    if not running and not finished:
        return {"results": "No background sub-agents are running or recently finished in this chat."}
    out = []
    if running:
        out.append(f"{len(running)} running sub-agent(s) (cancel with `manage_agents` then `stop <id>`, steer with `send_to_subagent`):")
        out.extend(running)
    if finished:
        out.append("Recently finished (results already delivered into this chat):")
        out.extend(finished)
    return {"results": "\n".join(out)}


# ---------------------------------------------------------------------------
# Handler classes registered in TOOL_HANDLERS
# ---------------------------------------------------------------------------

class ChatWithModelTool:
    async def execute(self, content: str, ctx: dict) -> Dict:
        return await chat_with_model(content, ctx.get("session_id"), owner=ctx.get("owner"))


class AskTeacherTool:
    async def execute(self, content: str, ctx: dict) -> Dict:
        return await ask_teacher(content, ctx.get("session_id"), owner=ctx.get("owner"))


class ListModelsTool:
    async def execute(self, content: str, ctx: dict) -> Dict:
        return await list_models(content, ctx.get("session_id"), owner=ctx.get("owner"))


class SpawnAgentTool:
    async def execute(self, content: str, ctx: dict) -> Dict:
        return await spawn_agent(
            content, ctx.get("session_id"), owner=ctx.get("owner"),
            parent_disabled=ctx.get("disabled_tools"),
        )


class ManageAgentsTool:
    async def execute(self, content: str, ctx: dict) -> Dict:
        return await manage_agents(content, ctx.get("session_id"), owner=ctx.get("owner"))
