"""Server-side chat turn starters that don't originate from an HTTP request.

Currently: ``start_server_resume_turn`` — after a background sub-agent delivers
its result, give the PARENT model a turn to react WITHOUT needing an open
browser (Claude-Code parity). This is the headless equivalent of the client's
``autoResumeAfterSubagent``: it persists a hidden resume prompt, runs the agent
loop, persists the assistant reply, and registers the run in ``agent_runs`` so
any client can attach/resume and watch it stream.

Kept out of routes/chat_routes.py so it has no FastAPI ``Request`` dependency —
the delivery path (a detached asyncio task) can call it directly.
"""
import asyncio
import json
import logging
from typing import AsyncGenerator, List, Dict, Optional

logger = logging.getLogger(__name__)

# Hidden resume prompt persisted as a user message. Its leading text MUST match
# the frontend history-hide filter (static/js/sessions.js _renderHistoryMessage:
# `trimmed.startsWith('The background sub-agent you dispatched has finished')`)
# so the bubble is suppressed on reload — same convention the client auto-resume
# uses. The sub-agent's result is already the assistant message directly above
# this prompt in history, so the model sees it in context.
SERVER_RESUME_PROMPT = (
    "The background sub-agent you dispatched has finished — its result is in the "
    "message directly above. Review it, incorporate the findings, and continue: "
    "give the outcome or the next step. Do not dispatch another sub-agent unless "
    "it is genuinely necessary. If it is not, continue directly from these "
    "findings; if a genuine gap remains, re-dispatch ONE sub-agent with a "
    "narrower, more specific scope and an explicitly higher max_rounds."
)

# Hidden prompt for the round-cap AUTO-CONTINUE turn (server-fired equivalent of
# the frontend Continue button's message — keep the two in spirit-sync). Persisted
# with {"hidden": True} metadata so no bubble renders.
AUTO_CONTINUE_PROMPT = (
    "You hit the per-turn step limit before finishing — the task is not complete. "
    "Continue from exactly where you left off and keep going until it is done. "
    "Do NOT repeat work that already succeeded; check the tool history above for "
    "what is already done."
)


async def start_server_resume_turn(
    session_id: str,
    framed_result: Optional[str] = None,
    owner: Optional[str] = None,
    prompt: Optional[str] = None,
    prompt_meta: Optional[Dict] = None,
) -> None:
    """Start a detached main-agent turn in ``session_id`` reacting to a just-
    delivered sub-agent result (or, with ``prompt`` overridden, any other
    server-fired reason — e.g. the round-cap auto-continue).

    Persists a hidden resume prompt, then registers an ``agent_runs`` run that
    streams the agent loop and persists the assistant reply on completion — so
    the parent model processes the result server-side and any client can attach
    to watch. ``framed_result`` is the untrusted-framed result text; when set it
    is appended to the model's message list so the model has the full result
    even if history trimming would drop the delivered message. It is NOT
    persisted separately (the delivered assistant message already is).
    """
    from core.models import ChatMessage, get_session_manager
    from src import agent_runs
    from src.agent_loop import stream_agent_loop
    from routes.chat_helpers import save_assistant_response, resolve_session_auth
    from src.settings import get_setting
    from src.agent_tools import MAX_AGENT_ROUNDS

    sm = get_session_manager()
    if not sm:
        logger.error("[server-resume] no session manager — cannot resume %s", session_id)
        return
    sess = sm.get_session(session_id)
    if sess is None:
        logger.error("[server-resume] session %s not found — cannot resume", session_id)
        return
    if not (getattr(sess, "model", "") or "").strip():
        logger.warning("[server-resume] session %s has no model — skipping resume", session_id)
        return

    # Ensure auth headers are populated for the upstream call (mirrors the route).
    try:
        resolve_session_auth(sess, session_id, owner=owner)
    except Exception as _e:
        logger.debug("[server-resume] resolve_session_auth skipped: %s", _e)

    # Persist the hidden resume prompt as a user message (so history is truthful
    # and the model sees it on later turns too); the content prefix makes the
    # frontend hide the bubble.
    try:
        sess.add_message(ChatMessage("user", prompt or SERVER_RESUME_PROMPT, metadata=(
            prompt_meta or {"hidden": True, "server_resume": True}
        )))
    except Exception as _e:
        logger.warning("[server-resume] failed to persist resume prompt: %s", _e)

    # Build the model's message list from session history. stream_agent_loop
    # adds its own system prompt / tool scaffolding.
    try:
        messages: List[Dict] = list(sess.get_context_messages())
    except Exception:
        messages = []
    if framed_result and not messages:
        # Fallback only: history normally already contains the delivered
        # "Sub-agent result" assistant message (persisted by _deliver before we
        # run) and the resume prompt directs the model to it — appending the
        # framed copy on top would send the same result twice. Only when the
        # history build failed does the framed copy become the model's sole
        # view of the result.
        messages.append({"role": "user", "content": framed_result})

    try:
        _max_rounds = int(get_setting("agent_max_rounds", MAX_AGENT_ROUNDS) or MAX_AGENT_ROUNDS)
    except (TypeError, ValueError):
        _max_rounds = MAX_AGENT_ROUNDS
    _max_rounds = max(1, min(_max_rounds, 200))
    try:
        _tool_budget = int(get_setting("agent_max_tool_calls", 0))
    except (TypeError, ValueError):
        _tool_budget = 0

    # Configured fallback chain, so a rate-limited primary still resumes.
    try:
        from src.endpoint_resolver import resolve_chat_fallback_candidates
        _fallbacks = resolve_chat_fallback_candidates(owner=owner)
    except Exception:
        _fallbacks = []

    # Session's reasoning-effort pref: a server-fired turn (sub-agent resume /
    # auto-continue) previously dropped the selector entirely, so a session set
    # to high effort silently continued at default. Same pref the composer
    # restores from (/api/session/{id}/prefs).
    _reasoning_effort = None
    try:
        from core.database import SessionLocal as _SL, Session as _DbSess
        _db = _SL()
        try:
            _row = _db.query(_DbSess.prefs).filter(_DbSess.id == session_id).first()
        finally:
            _db.close()
        if _row and isinstance(_row[0], dict):
            _val = str(_row[0].get("reasoning_effort") or "").strip().lower()
            if _val and _val != "default":
                _reasoning_effort = _val
    except Exception:
        pass

    async def _resume_stream() -> AsyncGenerator[str, None]:
        full_response = ""
        last_metrics = None
        _rounds_exhausted = False
        try:
            async for chunk in stream_agent_loop(
                sess.endpoint_url,
                sess.model,
                messages,
                headers=sess.headers,
                max_rounds=_max_rounds,
                max_tool_calls=_tool_budget,
                session_id=session_id,
                owner=owner,
                fallbacks=_fallbacks,
                reasoning_effort=_reasoning_effort,
                # A resume prompt reads as low-signal (no domain keywords), but
                # it must run the full loop with history so the model reports
                # the sub-agent's work instead of collapsing to a "Hey." stub.
                suppress_low_signal=True,
            ):
                if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                    try:
                        data = json.loads(chunk[6:])
                    except json.JSONDecodeError:
                        yield chunk
                        continue
                    if "delta" in data and not data.get("thinking"):
                        full_response += data["delta"]
                    elif data.get("type") == "metrics":
                        last_metrics = data.get("data", {})
                    elif data.get("type") == "rounds_exhausted":
                        # This server-fired turn ALSO hit the round cap. Chain
                        # another auto-continue (still bounded by the per-session
                        # cap) and tell the client so it shows "continuing…"
                        # instead of a manual button.
                        from src import subagent_runs as _sar
                        _rounds_exhausted = True
                        if _sar.can_auto_continue(session_id):
                            data["auto_continue"] = True
                            yield f'data: {json.dumps(data)}\n\n'
                            continue
                    yield chunk
                elif chunk == "data: [DONE]\n\n":
                    if full_response.strip():
                        try:
                            save_assistant_response(
                                sess, sm, session_id, full_response, last_metrics,
                            )
                        except Exception as _e:
                            logger.error("[server-resume] save failed for %s: %s", session_id, _e)
                    if _rounds_exhausted:
                        maybe_schedule_auto_continue(session_id, owner)
                    yield chunk
                else:
                    yield chunk
        except Exception as e:
            logger.error("[server-resume] stream error for %s: %s", session_id, e, exc_info=True)
            yield f'event: error\ndata: {json.dumps({"error": str(e), "status": 500})}\n\n'
            yield "data: [DONE]\n\n"

    # Register the run so any client can attach/resume and watch it stream, and
    # so it survives a client that never connects.
    agent_runs.start(session_id, _resume_stream())
    logger.info("[server-resume] detached resume run registered for %s", session_id)


# Strong refs so fire-and-forget auto-continue tasks aren't GC'd mid-flight.
_auto_continue_tasks: set = set()


def maybe_schedule_auto_continue(session_id: str, owner: Optional[str] = None) -> None:
    """Consume one auto-continue budget slot and schedule the continue turn.

    Called at [DONE]-time of a turn that hit the round cap (route path AND the
    server-resume path above, so continues chain until the cap). The short delay
    lets the finishing run drain out of agent_runs before the new one registers —
    agent_runs.start() supersedes a live run for the same session, and we don't
    want to race the final frames of the one that is ending."""
    try:
        from src import subagent_runs as _sar
        if not _sar.can_auto_continue(session_id):
            logger.info("[auto-continue] cap reached for %s — leaving the manual button", session_id)
            return
        _sar.note_auto_continue(session_id)
    except Exception:
        logger.exception("[auto-continue] cap check failed for %s", session_id)
        return

    async def _go():
        try:
            await asyncio.sleep(1.5)
            await start_server_resume_turn(
                session_id,
                owner=owner,
                prompt=AUTO_CONTINUE_PROMPT,
                prompt_meta={"hidden": True, "auto_continue": True},
            )
            logger.info("[auto-continue] continue turn started for %s", session_id)
        except Exception:
            logger.exception("[auto-continue] failed to start continue turn for %s", session_id)

    t = asyncio.create_task(_go())
    _auto_continue_tasks.add(t)
    t.add_done_callback(_auto_continue_tasks.discard)
