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
# Tools a sub-agent may NOT use: anything that would spawn/orchestrate more
# agents (recursion) or reach into other chats.
_SUBAGENT_DISABLED = frozenset({
    "spawn_agent", "create_session", "send_to_session", "list_sessions",
    "manage_session", "pipeline", "chat_with_model", "ask_teacher",
})


async def spawn_agent(content: str, session_id: Optional[str] = None, owner: Optional[str] = None) -> Dict:
    """Spawn a sub-agent that runs a full tool-using agent loop on a task and
    returns its final result to the caller.

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
    if model_spec:
        try:
            url, model, headers = await asyncio.to_thread(_resolve_model, model_spec, owner=owner)
        except ValueError as e:
            return {"error": str(e)}
    if not (url and model):
        return {"error": "Could not resolve a model for the sub-agent"}

    collected: list = []

    async def _drain():
        async for chunk in stream_agent_loop(
            url, model,
            [{"role": "user", "content": task}],
            headers=headers,
            owner=owner,
            session_id=None,                       # ephemeral — not persisted
            disabled_tools=set(_SUBAGENT_DISABLED),  # leaf worker: no recursion
            max_rounds=_SUBAGENT_MAX_ROUNDS,
        ):
            if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                try:
                    d = json.loads(chunk[6:])
                except Exception:
                    continue
                # Accumulate visible answer text only (skip thinking tokens).
                if "delta" in d and not d.get("thinking"):
                    collected.append(d["delta"])

    try:
        async with _SUBAGENT_SEMAPHORE:
            await asyncio.wait_for(_drain(), timeout=_SUBAGENT_TIMEOUT_S)
    except asyncio.TimeoutError:
        partial = "".join(collected).strip()
        return {
            "error": f"Sub-agent timed out after {_SUBAGENT_TIMEOUT_S}s",
            "partial_result": partial[:4000],
        }
    except Exception as e:
        logger.error(f"spawn_agent failed: {e}")
        return {"error": f"Sub-agent failed: {e}"}

    result = "".join(collected).strip()
    if len(result) > 8000:
        result = result[:8000] + "\n... (truncated)"
    return {"model": model, "result": result or "(sub-agent produced no text output)"}


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

            if keyword:
                model_ids = [m for m in model_ids if keyword in m.lower() or keyword in (ep.name or "").lower()]

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
        return await spawn_agent(content, ctx.get("session_id"), owner=ctx.get("owner"))
