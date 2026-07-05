"""search_tools — the always-available tool-discovery / unlock tool.

Odysseus RAG-selects only ~8 tools per turn; the rest are invisible to the
model. search_tools lets a model look up a capability it needs and unlock the
matching tools for the next round. These tests exercise it through the real
layers, not just the handler:

  * matching (incl. the mandatory keyword fallback when embeddings are DOWN),
  * exclusion of disabled + already-loaded tools,
  * REGISTRATION across TOOL_TAGS / ALWAYS_AVAILABLE / FUNCTION_TOOL_SCHEMAS /
    TOOL_HANDLERS, plus the parse + native-convert layer the manage_agents bug
    bypassed,
  * the next-round unlock (mirrors the skill-unlock block in agent_loop.py).
"""
import asyncio
import sys
from unittest.mock import MagicMock

for mod in ['src.agent_tools', 'src.tool_parsing', 'src.tool_schemas', 'src.tool_execution']:
    sys.modules.pop(mod, None)
for mod in [
    'sqlalchemy', 'sqlalchemy.orm', 'sqlalchemy.ext', 'sqlalchemy.ext.declarative',
    'sqlalchemy.ext.hybrid', 'sqlalchemy.sql', 'sqlalchemy.sql.expression',
    'src.database', 'core.models', 'core.database', 'core.auth',
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import src.agent_tools  # noqa: E402, F401
from src.agent_tools import TOOL_TAGS, TOOL_HANDLERS  # noqa: E402
from src.tool_parsing import parse_tool_blocks  # noqa: E402
from src.tool_schemas import function_call_to_tool_block, FUNCTION_TOOL_SCHEMAS  # noqa: E402
from src.tool_index import ALWAYS_AVAILABLE  # noqa: E402
import src.tool_index as ti  # noqa: E402


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _no_embeddings(monkeypatch):
    """Force retrieve() unavailable so only the keyword fallback contributes."""
    monkeypatch.setattr(ti, "get_tool_index", lambda: None)


# ── Registration (the manage_agents-bug trap) ──

def test_registered_in_all_four_places():
    # A tool missing from ANY of these is silently dropped from some path.
    assert "search_tools" in TOOL_TAGS
    assert "search_tools" in ALWAYS_AVAILABLE
    assert "search_tools" in TOOL_HANDLERS
    assert any(
        s.get("function", {}).get("name") == "search_tools"
        for s in FUNCTION_TOOL_SCHEMAS
    )


def test_fenced_block_parses_to_tool_block():
    blocks = parse_tool_blocks("```search_tools\ncalendar\n```")
    assert [(b.tool_type, b.content) for b in blocks] == [("search_tools", "calendar")]


def test_native_convert_yields_valid_block():
    # function_call_to_tool_block('search_tools', {'query': 'calendar'}) — the
    # parse-layer check the manage_agents bug bypassed.
    block = function_call_to_tool_block("search_tools", '{"query": "calendar"}')
    assert block is not None
    assert block.tool_type == "search_tools"
    assert block.content == '{"query": "calendar"}'


def test_native_convert_empty_and_alias_keys():
    # No args → empty query (browse). Alias keys normalise to query.
    assert function_call_to_tool_block("search_tools", "{}").content == '{"query": ""}'
    assert function_call_to_tool_block(
        "search_tools", '{"q": "serve a model"}'
    ).content == '{"query": "serve a model"}'


# ── Matching, incl. the mandatory keyword fallback ──

def test_query_matches_calendar_tool_with_embeddings_down(monkeypatch):
    _no_embeddings(monkeypatch)
    names = ti.search_tools_catalog("add an event to my calendar", limit=8)
    assert "manage_calendar" in names
    assert "search_tools" not in names  # never lists itself


def test_query_matches_email_tool_with_embeddings_down(monkeypatch):
    _no_embeddings(monkeypatch)
    names = ti.search_tools_catalog("send an email to my boss", limit=8)
    assert "send_email" in names


def test_results_are_capped(monkeypatch):
    _no_embeddings(monkeypatch)
    names = ti.search_tools_catalog("model server download serve email calendar", limit=8)
    assert len(names) <= 8


def test_empty_query_returns_full_catalog(monkeypatch):
    _no_embeddings(monkeypatch)
    names = ti.search_tools_catalog("", limit=8)
    # Browse mode is not capped and is alphabetical.
    assert len(names) > 8
    assert names == sorted(names)
    assert "manage_calendar" in names and "send_email" in names
    assert "search_tools" not in names


def test_excludes_disabled_and_already_loaded(monkeypatch):
    _no_embeddings(monkeypatch)
    names = ti.search_tools_catalog(
        "send an email",
        limit=8,
        exclude={"send_email"},      # already loaded this turn
        disabled={"reply_to_email"},  # admin-disabled
    )
    assert "send_email" not in names       # already available → not re-listed
    assert "reply_to_email" not in names   # disabled → never surfaced


def test_output_is_lean_one_line_per_tool_no_schemas(monkeypatch):
    _no_embeddings(monkeypatch)
    res = ti.format_search_tools_result("add a calendar event", limit=8)
    assert res["exit_code"] == 0
    assert "manage_calendar" in res["tools"]
    # One line per tool, prefixed "- ", and NO JSON param schemas in the text.
    for name in res["tools"]:
        assert f"- {name} " in res["output"]
    assert '"parameters"' not in res["output"]
    assert '"type": "object"' not in res["output"]
    # Trailing line telling the model the tools are now callable.
    assert "call them normally" in res["output"].lower()


# ── Handler path (through TOOL_HANDLERS) ──

def test_handler_returns_tools_and_respects_disabled(monkeypatch):
    _no_embeddings(monkeypatch)
    handler = TOOL_HANDLERS["search_tools"]
    ctx = {"disabled_tools": {"manage_calendar"}, "relevant_tools": set()}
    res = _run(handler('{"query": "calendar event"}', ctx))
    assert res["exit_code"] == 0
    assert "manage_calendar" not in res["tools"]  # disabled tool never unlocked


def test_handler_empty_query_browses(monkeypatch):
    _no_embeddings(monkeypatch)
    handler = TOOL_HANDLERS["search_tools"]
    res = _run(handler("{}", {"disabled_tools": set(), "relevant_tools": set()}))
    assert len(res["tools"]) > 8


def test_handler_is_idempotent(monkeypatch):
    _no_embeddings(monkeypatch)
    handler = TOOL_HANDLERS["search_tools"]
    ctx = {"disabled_tools": set(), "relevant_tools": set()}
    a = _run(handler('{"query": "serve a model"}', ctx))
    b = _run(handler('{"query": "serve a model"}', ctx))
    assert a["tools"] == b["tools"]


# ── Next-round unlock (mirrors the skill-unlock block in agent_loop.py) ──

def _apply_unlock(relevant_tools, result, disabled_tools):
    """The exact union the agent loop performs after a search_tools block:
    add returned names into the selection, skipping disabled + already-present.
    This is the seam that proves the native schema filter and fenced prompt
    assembly would include them next round (both read _relevant_tools)."""
    st_tools = result.get("tools") or []
    new = {
        t for t in st_tools
        if t and t not in relevant_tools and t not in (disabled_tools or set())
    }
    relevant_tools.update(new)
    return new


def test_next_round_tool_set_includes_discovered(monkeypatch):
    _no_embeddings(monkeypatch)
    handler = TOOL_HANDLERS["search_tools"]
    relevant = {"search_tools", "manage_memory"}  # this turn's tiny set
    res = _run(handler('{"query": "add a calendar event"}',
                        {"disabled_tools": set(), "relevant_tools": relevant}))
    _apply_unlock(relevant, res, set())
    # Next round's native schema filter (~agent_loop 3228) and the fenced prompt
    # both filter on _relevant_tools — so manage_calendar is now callable.
    assert "manage_calendar" in relevant
    schema_names = {s.get("function", {}).get("name") for s in FUNCTION_TOOL_SCHEMAS}
    next_round_schemas = [n for n in schema_names if n in relevant]
    assert "manage_calendar" in next_round_schemas


def test_unlock_never_adds_disabled_tool(monkeypatch):
    _no_embeddings(monkeypatch)
    handler = TOOL_HANDLERS["search_tools"]
    relevant = {"search_tools"}
    disabled = {"manage_calendar"}
    res = _run(handler('{"query": "calendar event"}',
                       {"disabled_tools": disabled, "relevant_tools": relevant}))
    _apply_unlock(relevant, res, disabled)
    assert "manage_calendar" not in relevant
