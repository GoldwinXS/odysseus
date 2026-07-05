"""discovery_tools.py — the search_tools tool.

Odysseus RAG-selects only ~8 relevant tools per turn; the rest are invisible
to the model (it sees a small name-sample hint). ``search_tools`` is an
always-available tool that lets the model discover a capability it needs and
unlock the matching tools for the NEXT round.

The matching logic (embedding retrieve() merged with an embedding-free keyword
scan) lives in ``src.tool_index``; this handler is a thin wrapper that reads
the query, threads the current turn's already-loaded set and the disabled set
through, and returns a lean text list plus a ``tools`` key. The agent loop
unions that ``tools`` key into ``_relevant_tools`` so the next round's native
schema list and fenced prompt both include them (mirrors the skill-unlock path
in agent_loop.py).
"""
import json
import logging
from typing import Dict

from src.constants import SEARCH_TOOLS_MAX_RESULTS

logger = logging.getLogger(__name__)


def _parse_query(content: str) -> str:
    """Extract the query from a tool-call body.

    Native models send ``{"query": "..."}`` (function_call_to_tool_block's
    generic json.dumps fallback); fenced local models may send raw JSON or a
    bare string. All spellings collapse to a plain query string; anything
    unparseable is treated as the query itself.
    """
    raw = (content or "").strip()
    if not raw:
        return ""
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw
        if isinstance(data, dict):
            for key in ("query", "q", "search", "text", "description"):
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            return ""
        return str(data).strip()
    return raw


async def search_tools(content: str, ctx: Dict) -> Dict:
    """Discover tools outside this turn's RAG-selected set and unlock them."""
    from src.tool_index import format_search_tools_result

    query = _parse_query(content)
    # Already-loaded this turn (don't re-list tools the model can already call)
    # and admin-disabled tools (never surface/unlock an off tool).
    exclude = set(ctx.get("relevant_tools") or set())
    disabled = set(ctx.get("disabled_tools") or set())
    try:
        result = format_search_tools_result(
            query,
            limit=SEARCH_TOOLS_MAX_RESULTS,
            exclude=exclude,
            disabled=disabled,
        )
    except Exception as e:  # never let discovery crash the turn
        logger.warning("search_tools failed: %s", e)
        return {"error": f"search_tools: {e}", "exit_code": 1}
    logger.info(
        "[search_tools] query=%r unlocked=%s", query, result.get("tools"),
    )
    return result


class SearchToolsTool:
    async def execute(self, content: str, ctx: dict) -> Dict:
        return await search_tools(content, ctx)
