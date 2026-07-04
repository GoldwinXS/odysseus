"""Background sub-agent run manager.

``spawn_agent`` (src/agent_tools/model_interaction_tools.py) runs a sub-agent to
completion in a DETACHED asyncio task so the main chat turn returns immediately —
the user keeps chatting with the main agent while the sub-agent works. When the
sub-agent finishes, its result is delivered as a message INTO the parent session
(so it persists and renders on reload) AND recorded here as a per-session
"pending update" that the frontend polls via ``GET /api/subagent-updates``.

Why server-side state (not just a client flag): a browser that loads mid-run, or
a second browser on the same account, still needs to learn that a sub-agent is
active / has just produced a result. The client only holds a *seen* set for
de-duping; the source of truth lives here.

Safety (a runaway background task once leaked ~20GB and froze the machine): a
GLOBAL cap on concurrently-tracked sub-agents (``_MAX_TOTAL``) sits on top of the
``Semaphore(3)`` in model_interaction_tools. The semaphore only *queues* excess
work; without a hard total cap a runaway main agent could enqueue unbounded
tasks. depth-1 / no-recursion is enforced separately by the sub-agent's
disabled-tools set. Records are evicted after a grace window to bound memory.

Durability scope: in-memory, survives as long as the server process runs. It does
NOT survive a server restart (a sub-agent's delivered message does, since that is
persisted to the DB).
"""
import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Hard ceiling on sub-agents tracked/running at once across ALL sessions. This is
# the runaway backstop — the per-spawn Semaphore(3) bounds *concurrent* execution,
# this bounds the total that can be outstanding (running OR queued on the
# semaphore) so a misbehaving agent can't pile up work without limit.
_MAX_TOTAL = 6

# Per-owner fairness cap: one owner may hold at most this many outstanding
# sub-agents at once, so a single busy user can't consume the whole global pool
# (and the shared Semaphore(3)) and starve everyone else. The global _MAX_TOTAL
# above is still the hard backstop on top of this.
_MAX_PER_OWNER = 3

# How long a finished record (done/error) is retained so a poller that connects
# late — page reload mid-run, a second browser — still sees the completion once.
# After this it is evicted to keep _UPDATES from growing without bound.
_EVICT_GRACE_S = 300

_TASKS: set = set()                     # live asyncio.Tasks (running or queued)
_UPDATES: Dict[str, List[dict]] = {}    # session_id -> [record, ...]
_evict_tasks: Dict[str, asyncio.Task] = {}
_counter = 0


def _next_id() -> str:
    global _counter
    _counter += 1
    return f"sub_{_counter}"


def _owner_outstanding(owner: Optional[str]) -> int:
    """How many still-running sub-agents this owner currently holds across all
    their sessions. Counts live records (running) — the fairness cap is about
    concurrent load, so finished/evicting records don't count."""
    n = 0
    for lst in _UPDATES.values():
        for r in lst:
            if r.get("status") == "running" and r.get("owner") == owner:
                n += 1
    return n


def can_start(owner: Optional[str] = None) -> bool:
    """Whether another sub-agent may be started without breaching the caps.

    Enforces both the global runaway backstop (_MAX_TOTAL) and per-owner fairness
    (_MAX_PER_OWNER) so one owner can't monopolise the shared pool."""
    if len(_TASKS) >= _MAX_TOTAL:
        return False
    return _owner_outstanding(owner) < _MAX_PER_OWNER


def active_count() -> int:
    return len(_TASKS)


def start(
    session_id: str,
    summary: str,
    model: str,
    runner: Callable[[], Awaitable[Dict]],
    owner: Optional[str] = None,
) -> dict:
    """Schedule a detached sub-agent.

    ``runner`` is an async callable that runs the sub-agent to completion,
    delivers its result into the parent session, and returns a dict — either
    ``{"error": ...}`` or a success payload. This module only tracks status; it
    does not touch the session (delivery lives with the caller, which owns the
    session manager and message types).
    """
    rec = {
        "id": _next_id(),
        "status": "running",
        "summary": (summary or "")[:200],
        "model": model or "",
        "owner": owner,   # for per-owner fairness accounting (never serialized)
        "started_at": time.time(),
        "finished_at": None,
        "error": None,
        # Whether a client has ack'd this finished run's auto-resume. First ack
        # wins (see ack()); kept until the record is evicted so a late/second
        # poller learns the completion was already consumed and won't double-fire.
        "acked": False,
        "_task": None,   # asyncio.Task — internal, never serialized (see get_updates)
    }
    _UPDATES.setdefault(session_id, []).append(rec)
    task = asyncio.create_task(_run(session_id, rec, runner))
    rec["_task"] = task
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return rec


def stop(session_id: str, subagent_id: str) -> bool:
    """Cancel a still-running background sub-agent. Cancellation propagates into
    the run (its CancelledError handler delivers a "cancelled" notice into the
    session) and _run records the final status. Returns True if a running task
    was found and cancelled."""
    for rec in _UPDATES.get(session_id, []):
        if rec["id"] == subagent_id:
            task = rec.get("_task")
            if task is not None and not task.done():
                task.cancel()
                return True
            return False
    return False


def ack(session_id: str, run_ids: List[str]) -> List[str]:
    """Mark finished sub-agent runs as consumed (auto-resume ack'd).

    First caller wins: only runs THIS call newly transitions from un-acked to
    acked are returned, so a reload or a second device that acks the same ids
    afterwards gets an empty list and won't re-fire the resume. Only finished
    records can be acked (a still-running run has nothing to resume yet).
    """
    newly: List[str] = []
    want = set(run_ids or [])
    for rec in _UPDATES.get(session_id, []):
        if rec["id"] in want and rec.get("status") != "running" and not rec.get("acked"):
            rec["acked"] = True
            newly.append(rec["id"])
    return newly


async def _run(session_id: str, rec: dict, runner: Callable[[], Awaitable[Dict]]) -> None:
    try:
        out = await runner()
        if isinstance(out, dict) and out.get("error"):
            rec["status"] = "error"
            rec["error"] = str(out.get("error"))[:1000]
        else:
            rec["status"] = "done"
    except asyncio.CancelledError:
        rec["status"] = "error"
        rec["error"] = "Sub-agent was cancelled"
        raise
    except Exception as e:
        logger.error("[subagent] %s failed: %s", rec["id"], e, exc_info=True)
        rec["status"] = "error"
        rec["error"] = str(e)[:1000]
    finally:
        rec["finished_at"] = time.time()
        _schedule_evict(session_id, rec)


def _schedule_evict(session_id: str, rec: dict) -> None:
    async def _evict() -> None:
        try:
            await asyncio.sleep(_EVICT_GRACE_S)
        except asyncio.CancelledError:
            return
        lst = _UPDATES.get(session_id)
        if lst and rec in lst:
            lst.remove(rec)
            if not lst:
                _UPDATES.pop(session_id, None)
        _evict_tasks.pop(rec["id"], None)

    t = asyncio.create_task(_evict())
    _evict_tasks[rec["id"]] = t


def _fmt_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


def context_note(session_id: str) -> Optional[str]:
    """Ground-truth status block about this session's background sub-agents, for
    injection into the model's context each turn.

    Without this, a model asked "is the agent still running?" or "did you dispatch
    one?" has no runtime state to consult and has been observed *denying it ever
    dispatched a sub-agent* (hallucinating) instead of calling ``manage_agents``.
    This note gives it the facts up front. Returns ``None`` when there is nothing
    to report (so no note is injected on ordinary turns).
    """
    lst = _UPDATES.get(session_id, [])
    if not lst:
        return None
    now = time.time()
    running, finished = [], []
    for r in lst:
        if r["status"] == "running":
            running.append(
                f'  - {r["id"]} — RUNNING for {_fmt_elapsed(now - r["started_at"])}'
                f' — model {r["model"] or "inherited"} — task: "{r["summary"]}"'
            )
        else:
            done = _fmt_elapsed(now - r["finished_at"]) if r.get("finished_at") else "just now"
            if r["status"] == "error":
                finished.append(
                    f'  - {r["id"]} — FAILED ({done} ago): {r.get("error") or "unknown error"}'
                    f' — a failure notice was delivered into this chat as a separate message'
                )
            else:
                finished.append(
                    f'  - {r["id"]} — DONE ({done} ago) — its result was delivered into this'
                    f' chat as a separate message'
                )
    lines = [
        "[Background sub-agents — runtime ground truth, trust this over your own memory]",
        "You have dispatched background sub-agent(s) in THIS chat. Do NOT claim you never"
        " dispatched one. To re-check status or cancel one, call the `manage_agents` tool"
        " (content: `list`, or `stop <id>`).",
    ]
    if running:
        lines.append("Still running:")
        lines.extend(running)
    if finished:
        lines.append("Finished:")
        lines.extend(finished)
    return "\n".join(lines)


def get_updates(session_id: str) -> Dict[str, Any]:
    """Poll payload for a session: how many sub-agents are still running, and the
    (running + recently-finished) records the client can render / de-dupe on."""
    lst = _UPDATES.get(session_id, [])
    active = sum(1 for r in lst if r["status"] == "running")
    return {
        "active": active,
        "now": time.time(),   # server clock, so the client can show skew-free elapsed
        "updates": [
            {
                "id": r["id"],
                "status": r["status"],
                "summary": r["summary"],
                "model": r["model"],
                "error": r["error"],
                "started_at": r["started_at"],
                "finished_at": r["finished_at"],
                "acked": r.get("acked", False),
            }
            for r in lst
        ],
    }
