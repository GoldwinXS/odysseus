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
_SUBAGENT_MAX_ROUNDS = 12                     # tool-loop rounds per sub-agent
_SUBAGENT_TIMEOUT_S = 240                     # wall-clock cap per sub-agent
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

    Content: the task/instructions for the sub-agent. An optional first line
    ``model: <name>`` overrides the model (defaults to this chat's model).

    Use for a self-contained unit of work you want done and reported back —
    e.g. "read js/render/models.js, screenshot localhost:1338, and list what
    looks visually wrong." The sub-agent has the normal tools (files, shell,
    browser) but cannot itself spawn more agents.
    """
    import json
    from src.agent_loop import stream_agent_loop
    from src.ai_interaction import get_session_manager, _resolve_model
    from src.tool_index import get_tool_index, ALWAYS_AVAILABLE

    task = (content or "").strip()
    if not task:
        return {"error": "No task provided for the sub-agent"}

    # Optional `model: <name>` override on the first line.
    model_spec = None
    if task.lower().startswith("model:"):
        first, _, rest = task.partition("\n")
        model_spec = first.split(":", 1)[1].strip()
        task = rest.strip()
        if not task:
            return {"error": "Sub-agent task was empty after the model: line"}

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
    _summary = _task.splitlines()[0][:120] if _task else ""

    # Give the sub-agent the SAME fallback chain the main chat uses, so a
    # rate-limited / spend-capped primary (e.g. Gemini 429 "exceeded monthly
    # spending cap") falls back to another model instead of failing with an
    # empty result. Best-effort — an empty chain just means no fallback.
    try:
        from src.endpoint_resolver import resolve_chat_fallback_candidates
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

    # Populated with the tracked run's id right after subagent_runs.start()
    # returns, so _deliver (defined here, invoked later) can flag the run for
    # server-side resume by id. A plain dict works because _deliver reads it at
    # call time, well after start() has filled it in.
    _run_meta: Dict[str, Optional[str]] = {"id": None}

    def _deliver(error: Optional[str], partial: str) -> None:
        """Post the sub-agent's outcome as an assistant message into the parent
        session. Persists immediately (add_message -> _persist_message commits),
        so it renders on reload even if no client is currently connected.

        Server-side resume (Claude-Code parity): after persisting a SUCCESSFUL
        result, ensure the parent model actually processes it, server-side —
        no open browser required:
          * If the parent session has a LIVE agent run, enqueue a steering entry
            (framed as untrusted context) so the running turn reacts next round.
          * Otherwise start a DETACHED main-agent resume turn in the parent
            session (registered in agent_runs so clients can attach/see it).
        Never resume on error/cancelled/timeout — those deliver the notice only.
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

        # Only successful completions drive a resume; error/cancelled/timeout
        # deliver the notice and stop (resuming on those would burn tokens on a
        # spawn/fail/resume loop against a down provider).
        if error:
            return
        try:
            _maybe_server_resume(partial or "")
        except Exception as e:
            logger.error("[subagent-resume] resume dispatch failed for %s: %s", _parent_session, e, exc_info=True)

    def _maybe_server_resume(result_text: str) -> None:
        """Push the sub-agent's result to the parent model, server-side.

        Live turn → enqueue a steering entry (Feature 1's queue). No live turn →
        start a detached resume turn. Both frame the result via
        untrusted_context_message so fetched/quoted content can't speak with
        user/assistant authority. Capped + single-fire guarded via subagent_runs.
        """
        from src import agent_runs, subagent_runs
        from src.prompt_security import untrusted_context_message

        sub_id = _run_meta.get("id")
        # Frame the result as untrusted context (may quote fetched web content).
        framed = untrusted_context_message(
            f"background sub-agent {sub_id or ''} ({_model}) result",
            f"Background sub-agent {sub_id or ''} ({_model}) finished. Full result:\n\n{result_text}",
        )["content"]

        # 1) A turn is live for the parent session → steer into it (atomic:
        #    enqueue only succeeds if the run is still running).
        if agent_runs.enqueue_steer(_parent_session, framed, kind="subagent"):
            if sub_id:
                subagent_runs.mark_resume(_parent_session, sub_id, "server")
            logger.info("[subagent-resume] steered result into live turn for %s", _parent_session)
            return

        # 2) No live turn → start a detached server-side resume turn, subject to
        #    the cap and single-fire guard.
        if not subagent_runs.can_server_resume(_parent_session):
            logger.info("[subagent-resume] resume capped/already-running for %s — delivered only", _parent_session)
            return
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
            await asyncio.wait_for(_drain_summary(), timeout=_SUBAGENT_TIMEOUT_S)
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
        stats = {"tools": 0, "hit_cap": False, "error": None, "last_tool": None}

        async def _drain():
            async for chunk in stream_agent_loop(
                _url, _model,
                [
                    {"role": "system", "content": _SUBAGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": _task},
                ],
                headers=_headers,
                owner=_owner,
                session_id=None,                        # ephemeral — not persisted
                disabled_tools=set(_disabled),           # leaf worker: no recursion, plus
                                                         # the parent turn's disabled policy
                relevant_tools=set(_sub_tools),          # coding baseline force-included
                workspace=_workspace,                    # inherit parent's project folder
                max_rounds=_SUBAGENT_MAX_ROUNDS,
                max_tokens=_max_tokens,                  # the default 4096 truncated long
                                                         # final summaries mid-word; clamped
                                                         # above to the model's real ceiling
                fallbacks=_fallbacks,
            ):
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
                        if d.get("tool"):
                            stats["last_tool"] = d["tool"]
                    elif _t == "rounds_exhausted":
                        stats["hit_cap"] = True
                    # Accumulate visible answer text only (skip thinking tokens).
                    if "delta" in d and not d.get("thinking"):
                        collected.append(d["delta"])

        error = None
        try:
            async with _SUBAGENT_SEMAPHORE:
                await asyncio.wait_for(_drain(), timeout=_SUBAGENT_TIMEOUT_S)
        except asyncio.TimeoutError:
            error = f"Sub-agent timed out after {_SUBAGENT_TIMEOUT_S}s"
        except asyncio.CancelledError:
            # Deliver whatever it produced before cancellation, then propagate so
            # the run manager records it as stopped/error and cleans up.
            if deliver:
                _deliver(error="Sub-agent was cancelled", partial="".join(collected).strip()[:_SUBAGENT_RESULT_CAP])
            raise
        except Exception as e:
            logger.error(f"spawn_agent run failed: {e}")
            error = f"Sub-agent failed: {e}"

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
                    f"(Hit the {_SUBAGENT_MAX_ROUNDS}-round limit; {stats['tools']} tool "
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
                    f"{_SUBAGENT_MAX_ROUNDS}-round limit before writing a summary. "
                    "Try a narrower task, or spawn it with an explicit fast model.)"
                )
            elif stats["tools"]:
                result = (
                    f"(The sub-agent ran {stats['tools']} tool call(s) but produced no "
                    "written summary of what it found.)"
                )
            else:
                result = "(The sub-agent produced no output.)"
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
    # Let _deliver flag this run for server-side resume by id when it finishes.
    _run_meta["id"] = rec["id"]
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

    Use this to check on work you dispatched with spawn_agent, or to stop a
    sub-agent that is taking too long or is no longer needed.
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

    # Default: list.
    upd = subagent_runs.get_updates(session_id)
    now = upd.get("now") or time.time()
    running, finished = [], []
    for u in upd.get("updates", []):
        el = int(max(0, now - (u.get("started_at") or now)))
        summ = (u.get("summary") or "").strip()
        model = u.get("model") or "?"
        if u.get("status") == "running":
            running.append(f"  - {u['id']} [{model}] running {el}s — {summ}")
        else:
            st = u.get("status")
            if u.get("error"):
                st = f"{st}: {u['error']}"
            finished.append(f"  - {u['id']} [{model}] {st} — {summ}")
    if not running and not finished:
        return {"results": "No background sub-agents are running or recently finished in this chat."}
    out = []
    if running:
        out.append(f"{len(running)} running sub-agent(s) (cancel with `manage_agents` then `stop <id>`):")
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
