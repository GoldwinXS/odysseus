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
# the runaway backstop — the execution semaphore bounds *concurrent* execution,
# this bounds the total that can be outstanding (running OR queued on the
# semaphore) so a misbehaving agent can't pile up work without limit.
# Both caps are settings-tunable (subagent_max_total / subagent_max_per_owner);
# the constants are fallbacks. They were hardcoded at 6/3 — the literal source
# of the user's "only 3 sub-agents" ceiling on a single-user box.
_MAX_TOTAL = 16

# Per-owner fairness cap: one owner may hold at most this many outstanding
# sub-agents at once, so a single busy user can't consume the whole global pool
# and starve everyone else. The global _MAX_TOTAL above is still the hard
# backstop on top of this.
_MAX_PER_OWNER = 8


def _setting_int(key: str, fallback: int, lo: int = 1, hi: int = 200) -> int:
    """Settings-tunable integer with a clamped range and a constant fallback."""
    try:
        from src.settings import get_setting
        v = int(get_setting(key, fallback) or fallback)
        return max(lo, min(v, hi))
    except Exception:
        return fallback


def max_total() -> int:
    return _setting_int("subagent_max_total", _MAX_TOTAL, 1, 64)


def max_per_owner() -> int:
    return _setting_int("subagent_max_per_owner", _MAX_PER_OWNER, 1, 64)

# How long a finished record (done/error) is retained so a poller that connects
# late — page reload mid-run, a second browser — still sees the completion once.
# After this it is evicted to keep _UPDATES from growing without bound.
_EVICT_GRACE_S = 300

_TASKS: set = set()                     # live asyncio.Tasks (running or queued)
_UPDATES: Dict[str, List[dict]] = {}    # session_id -> [record, ...]
_evict_tasks: Dict[str, asyncio.Task] = {}
_counter = 0

# Server-side sub-agent resume (Claude-Code parity: the parent model always
# processes a sub-agent result, server-side, even with no browser open).
# _resume_count caps consecutive server-fired resumes per session so a
# down/rate-limited provider (which finishes as status=error and would itself
# be delivered) can't drive an unbounded spawn/finish/resume token loop.
# _resume_running guards against starting a second resume turn while one is
# already going. Both reset on genuine user activity (a normal send / steer).
_MAX_SERVER_RESUMES = 3
_resume_count: Dict[str, int] = {}      # session_id -> consecutive server resumes
_resume_running: set = set()            # session_ids with a resume turn in flight

# Failure-driven resumes are capped SEPARATELY and more tightly than success
# resumes. A subagent that FAILS (timeout / stall / loop / upstream error) should
# still wake the parent once so it can decide the next step (the hang-forever bug),
# but repeated failures must not ping-pong the parent into a spawn/fail/resume
# token loop — especially against a down/rate-limited provider. This cap bounds
# consecutive failure-driven resumes per session; it shares the _resume_running
# in-flight guard with the success path and resets on genuine user activity.
_MAX_FAILURE_RESUMES = 2
_failure_resume_count: Dict[str, int] = {}   # session_id -> consecutive failure resumes

# Prefix for a sub-agent's ephemeral STEER-QUEUE session id (mid-flight steering
# via send_to_subagent). It is NOT a real chat session — no DB row exists for it.
# The agent loop's steering-injection path (agent_loop._inject_steering_messages)
# calls sm.get_session(session_id) and sess.add_message() on it; both must be
# harmless for this id. We satisfy that by registering an in-memory,
# NON-PERSISTING Session object under this id in the session-manager cache (see
# _register_ephemeral_session), so get_session finds it (no KeyError from the
# DB-load path) and add_message appends to history WITHOUT writing a chat_messages
# row under the fake session. Kept in sync with the check other layers do on the
# id prefix.
_QUEUE_SESSION_PREFIX = "_subagent_"


def is_queue_session(session_id: Optional[str]) -> bool:
    """True for a sub-agent's ephemeral steer-queue session id (never a real
    chat session). Used to skip DB persistence for steered messages."""
    return bool(session_id) and session_id.startswith(_QUEUE_SESSION_PREFIX)


def parent_session_for_queue(queue_session: Optional[str]) -> Optional[str]:
    """Reverse-map a sub-agent's ephemeral queue session id back to the real
    parent chat session that spawned it. Returns None if unknown (e.g. the run
    already evicted). Used so work a sub-agent starts (e.g. a background job)
    is delivered into a persisted chat, not the queue session that is torn down
    the moment the sub-agent ends."""
    if not is_queue_session(queue_session):
        return None
    for parent, recs in _UPDATES.items():
        for rec in recs:
            if rec.get("queue_session") == queue_session:
                return parent
    return None


def _register_ephemeral_session(queue_session: str, model: str) -> None:
    """Register a NON-PERSISTING in-memory Session under the sub-agent's queue id
    so the loop's steering path can get_session()/add_message() it without either
    raising KeyError (unknown id → DB load → KeyError) or writing chat rows to the
    DB under a fake session. Best-effort: if the session manager or core models
    aren't importable (unit tests that stub them), we skip silently — steering
    still enqueues/drains via agent_runs; only the loop-side persistence guard is
    unavailable, and that path is itself guarded by is_queue_session at delivery."""
    try:
        from core.models import Session, get_session_manager_instance
    except Exception:
        return
    sm = get_session_manager_instance()
    if sm is None or not hasattr(sm, "sessions"):
        return
    if queue_session in sm.sessions:
        return

    class _EphemeralSubagentSession(Session):
        """A queue-only session that never persists. add_message appends to the
        in-memory history so the loop's assumptions hold, but does NOT call the
        session manager's _persist_message (which would drop the write AND pop
        this object from the cache, breaking later steer rounds)."""
        def add_message(self, message):   # type: ignore[override]
            self.history.append(message)
            self.message_count = len(self.history)

    try:
        sm.sessions[queue_session] = _EphemeralSubagentSession(
            id=queue_session, name="(sub-agent steer queue)",
            endpoint_url="", model=model or "", history=[],
        )
    except Exception as e:
        logger.debug("[subagent] ephemeral session register skipped: %s", e)


def _drop_ephemeral_session(queue_session: str) -> None:
    """Remove the ephemeral queue session from the manager cache on teardown."""
    try:
        from core.models import get_session_manager_instance
        sm = get_session_manager_instance()
        if sm is not None and hasattr(sm, "sessions"):
            sm.sessions.pop(queue_session, None)
    except Exception:
        pass


def can_failure_resume(session_id: str) -> bool:
    """Whether a FAILURE-driven server resume may fire for this session.

    False when a resume turn is already running or the (tighter) failure cap is
    hit. Independent of the success cap so a normal successful delivery isn't
    starved by prior failures and vice-versa."""
    if session_id in _resume_running:
        return False
    return _failure_resume_count.get(session_id, 0) < _MAX_FAILURE_RESUMES


def note_failure_resume(session_id: str) -> None:
    """Record that a failure-driven server resume just fired (consumes cap)."""
    _failure_resume_count[session_id] = _failure_resume_count.get(session_id, 0) + 1


def can_server_resume(session_id: str) -> bool:
    """Whether another server-side resume may fire for this session.

    False when the per-session cap is hit or a resume turn is already running.
    """
    if session_id in _resume_running:
        return False
    return _resume_count.get(session_id, 0) < _MAX_SERVER_RESUMES


def note_server_resume(session_id: str) -> None:
    """Record that a server-side resume just fired (consumes cap budget)."""
    _resume_count[session_id] = _resume_count.get(session_id, 0) + 1


def set_resume_running(session_id: str, running: bool) -> None:
    """Mark whether a server-side resume turn is currently in flight."""
    if running:
        _resume_running.add(session_id)
    else:
        _resume_running.discard(session_id)


def note_user_activity(session_id: str) -> None:
    """Reset the server-resume caps (success AND failure) on genuine user activity
    (normal send / steer / ack). Mirrors the frontend's resetSubagentAutoResume."""
    _resume_count.pop(session_id, None)
    _failure_resume_count.pop(session_id, None)
    _auto_continue_count.pop(session_id, None)


# Round-cap auto-continue (the "please continue" fix). When a turn exhausts
# max_rounds mid-task, the server fires a hidden continue turn itself instead
# of only rendering a Continue button that needs an open, watched browser —
# audit of 2026-07-09 found the manual button was the ONLY path, and users
# were typing "hello?"/"please continue" to do the watchdog's job by hand.
# Capped per session so a task that can't finish doesn't burn rounds forever
# (3 auto-continues x 20 rounds on top of the original 20 = 80 rounds max);
# resets on genuine user activity like the resume caps above.
_MAX_AUTO_CONTINUES = 3
_auto_continue_count: Dict[str, int] = {}   # session_id -> consecutive auto-continues


def can_auto_continue(session_id: str) -> bool:
    """Whether a server-side round-cap auto-continue may fire for this session."""
    if session_id in _resume_running:
        return False
    _cap = _setting_int("agent_auto_continue_max", _MAX_AUTO_CONTINUES, 0, 50)
    return _auto_continue_count.get(session_id, 0) < _cap


def note_auto_continue(session_id: str) -> None:
    """Record that a round-cap auto-continue just fired (consumes cap budget)."""
    _auto_continue_count[session_id] = _auto_continue_count.get(session_id, 0) + 1


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

    Enforces both the global runaway backstop (max_total) and per-owner fairness
    (max_per_owner) so one owner can't monopolise the shared pool."""
    if len(_TASKS) >= max_total():
        return False
    return _owner_outstanding(owner) < max_per_owner()


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
    _id = _next_id()
    rec = {
        "id": _id,
        "status": "running",
        "summary": (summary or "")[:200],
        "model": model or "",
        "owner": owner,   # for per-owner fairness accounting (never serialized)
        # Ephemeral steer-queue session id for this sub-agent (mid-flight steering
        # via send_to_subagent). Threaded into stream_agent_loop(session_id=...) so
        # the loop drains steers; NOT a real chat session (never serialized).
        "queue_session": f"{_QUEUE_SESSION_PREFIX}{_id}",
        "started_at": time.time(),
        "finished_at": None,
        "error": None,
        # Whether a client has ack'd this finished run's auto-resume. First ack
        # wins (see ack()); kept until the record is evicted so a late/second
        # poller learns the completion was already consumed and won't double-fire.
        "acked": False,
        # How this completion's parent-model reaction is being handled. The
        # delivery path (model_interaction_tools._deliver) sets this to
        # "server" once it has enqueued a steer into a live turn OR started a
        # detached resume turn itself; the frontend reads it from get_updates
        # and MUST NOT client-fire an auto-resume when it is "server".
        "resume": None,   # None | "server"
        # Live mid-run progress — pushed by model_interaction_tools' _drain
        # per-chunk handler (mutating this SAME dict object, since it lives
        # here in _UPDATES) so a parent checking in via manage_agents sees
        # genuine status instead of just "still running". None/absent until
        # the sub-agent's first chunk arrives.
        "rounds_used": None,        # highest agent_step round seen so far
        "last_tool": None,          # name of the most recent tool_start
        "last_activity_at": None,   # time.time() of the most recent chunk
        "output_tail": None,        # last ~300 chars of visible output so far
        "_task": None,   # asyncio.Task — internal, never serialized (see get_updates)
    }
    _UPDATES.setdefault(session_id, []).append(rec)
    # Stand up the steer mailbox + ephemeral queue session BEFORE the run starts,
    # so a send_to_subagent that races an early round still lands. Both are torn
    # down in _run's finally.
    _queue = rec["queue_session"]
    try:
        from src import agent_runs
        agent_runs.register_steer_mailbox(_queue)
    except Exception as e:
        logger.debug("[subagent] steer mailbox register skipped for %s: %s", _queue, e)
    _register_ephemeral_session(_queue, model)
    task = asyncio.create_task(_run(session_id, rec, runner))
    rec["_task"] = task
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return rec


def running_count(session_id: Optional[str]) -> int:
    """How many sub-agent runs are currently in flight for this parent session.
    Used to tell a genuine "it's running in the background" statement (a real
    run exists) from a hallucinated dispatch claim (none registered)."""
    return sum(1 for rec in _UPDATES.get(session_id or "", [])
               if rec.get("status") == "running")


def find_running(session_id: str, subagent_id: str) -> Optional[dict]:
    """Return the run record for a RUNNING sub-agent in this session, or None if
    it doesn't exist / already finished. Used by send_to_subagent to validate a
    steer target before enqueuing."""
    for rec in _UPDATES.get(session_id, []):
        if rec["id"] == subagent_id:
            return rec if rec.get("status") == "running" else None
    return None


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


def mark_resume(session_id: str, subagent_id: str, mode: str = "server") -> None:
    """Flag a finished run's completion as handled server-side, so no client
    fires a duplicate auto-resume. Exposed via get_updates (`resume`)."""
    for rec in _UPDATES.get(session_id, []):
        if rec["id"] == subagent_id:
            rec["resume"] = mode
            # A server-side resume also consumes/records against the cap only
            # once actually fired; the delivery path calls note_server_resume
            # explicitly. Here we just record the mode on the record.
            return


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
        # Tear down the steer mailbox + ephemeral queue session — the sub-agent
        # is no longer running, so it can't be steered any more.
        _queue = rec.get("queue_session")
        if _queue:
            try:
                from src import agent_runs
                agent_runs.close_steer_mailbox(_queue)
            except Exception:
                pass
            _drop_ephemeral_session(_queue)
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
    (running + recently-finished) records the client can render / de-dupe on.

    Also carries live mid-run progress (rounds_used, last_tool, output_tail,
    seconds_since_activity) so a parent model checking in via manage_agents can
    say "it's going fine" or decide to steer/stop, rather than seeing only a
    bare running/done status."""
    lst = _UPDATES.get(session_id, [])
    active = sum(1 for r in lst if r["status"] == "running")
    _now = time.time()
    return {
        "active": active,
        "now": _now,   # server clock, so the client can show skew-free elapsed
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
                # "server" => the parent-model reaction is being handled
                # server-side (steer into a live turn or a detached resume
                # turn). The client must NOT client-fire an auto-resume for it.
                "resume": r.get("resume"),
                # Live progress (None until the sub-agent's first chunk lands).
                "rounds_used": r.get("rounds_used"),
                "last_tool": r.get("last_tool"),
                "output_tail": r.get("output_tail"),
                "seconds_since_activity": (
                    max(0, int(_now - r["last_activity_at"]))
                    if r.get("last_activity_at") else None
                ),
            }
            for r in lst
        ],
    }
