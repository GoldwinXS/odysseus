"""Tool selection is stateless per-turn, so a session deep into file editing
could lose read_file/edit_file/bash on a follow-up phrased about the outcome
("make the grid look better") — the model then says it "has no file access
this turn" and asks the user to paste the file.

_session_used_tools re-derives the tools a session has already invoked (from
persisted tool_events) so those capabilities can be unioned back every turn
and never disappear.
"""
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

for mod in ['src.agent_tools', 'src.tool_parsing', 'src.tool_schemas', 'src.tool_execution']:
    sys.modules.pop(mod, None)
for mod in [
    'sqlalchemy', 'sqlalchemy.orm', 'sqlalchemy.ext', 'sqlalchemy.ext.declarative',
    'sqlalchemy.ext.hybrid', 'sqlalchemy.sql', 'sqlalchemy.sql.expression',
    'src.database', 'core.models', 'core.auth',
]:
    sys.modules.setdefault(mod, MagicMock())

import src.agent_tools  # noqa: E402, F401
import src.agent_loop as al  # noqa: E402


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def all(self):
        return self._rows


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *a, **k):
        return _FakeQuery(self._rows)

    def close(self):
        pass


def _patch_db(monkeypatch, metas):
    """Point _session_used_tools at canned assistant-message metadata rows."""
    rows = [(m,) for m in metas]
    # ChatMessage columns must support ORM ops (`.timestamp.desc()`,
    # `col == val`) — MagicMock handles any attribute/comparison chain.
    fake_core_db = SimpleNamespace(SessionLocal=lambda: _FakeDB(rows), ChatMessage=MagicMock())
    monkeypatch.setitem(sys.modules, 'core.database', fake_core_db)


def test_used_file_tools_are_recovered(monkeypatch):
    _patch_db(monkeypatch, [
        '{"tool_events": [{"tool": "bash", "command": "ls"}, {"tool": "edit_file", "command": "{}"}]}',
        '{"tool_events": [{"tool": "read_file", "command": "{}"}, {"tool": "grep", "command": "{}"}]}',
    ])
    used = al._session_used_tools("sess-1")
    assert {"bash", "edit_file", "read_file", "grep"} <= used


def test_legacy_and_unknown_names_are_dropped(monkeypatch):
    # "note" is a legacy tool_events label with no current tag; it must not
    # be re-offered. Real tags around it still stick.
    _patch_db(monkeypatch, [
        '{"tool_events": [{"tool": "note"}, {"tool": "bash", "command": "x"}]}',
    ])
    used = al._session_used_tools("sess-1")
    assert "bash" in used
    assert "note" not in used


def test_escaped_tool_key_in_output_is_not_matched(monkeypatch):
    # A tool's OUTPUT that literally contains {"tool": "rm"} is stored escaped
    # (\"tool\": ...). It must NOT be read as the session having used "rm"
    # (which isn't a tag anyway) or any tool — only the real event key counts.
    _patch_db(monkeypatch, [
        '{"tool_events": [{"tool": "bash", "command": "cat x",'
        ' "output": "the file said \\"tool\\": \\"read_file\\" somewhere"}]}',
    ])
    used = al._session_used_tools("sess-1")
    assert used == {"bash"}  # NOT read_file (that was only inside escaped output)


def test_empty_or_missing_session_id(monkeypatch):
    _patch_db(monkeypatch, [])
    assert al._session_used_tools(None) == set()
    assert al._session_used_tools("") == set()


def test_no_tool_events_returns_empty(monkeypatch):
    _patch_db(monkeypatch, ['{"response_time": 1.2, "model": "glm-5.2"}'])
    assert al._session_used_tools("sess-1") == set()
