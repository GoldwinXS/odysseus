"""FIX A — post-edit syntax gate on edit_file / write_file.

After a successful write the harness runs a cheap by-extension syntax/parse
check and, on failure, appends the checker's error to the tool result so the
model sees it and fixes it next round. The check is best-effort and NON-FATAL:
the file is always saved (exit_code 0) regardless of the check, and a missing
`node` silently skips the .js check.

These tests target the checker directly (`_syntax_check_written_file`,
`_augment_with_syntax_check`) and through the real EditFileTool/WriteFileTool
handlers, patching subprocess where needed so they never depend on node being
installed.
"""
import json
import os
import tempfile

import pytest

from src.agent_tools import filesystem_tools as ft
from src.agent_tools.filesystem_tools import (
    EditFileTool,
    WriteFileTool,
    _syntax_check_written_file,
    _augment_with_syntax_check,
)


def _tmp(name: str) -> str:
    # Isolated-checker tests call the checker directly (no path confinement).
    return os.path.join(tempfile.gettempdir(), name)


def _ws(name: str) -> str:
    # End-to-end handler tests go through _resolve_tool_path, which confines
    # writes to allowed roots. /tmp is one (maps to C:\tmp on this host), same
    # as the existing tests/test_edit_file.py.
    return os.path.join("/tmp", name)


# ── The checker in isolation ──────────────────────────────────────────────
def test_bad_python_reports_py_compile_error():
    p = _tmp("pesc_bad.py")
    with open(p, "w", encoding="utf-8") as f:
        f.write("def f(:\n    return 1\n")  # syntax error
    try:
        err = _syntax_check_written_file(p)
        assert err is not None
        # py_compile surfaces a SyntaxError; the message names the file/line.
        assert "SyntaxError" in err or "invalid syntax" in err
    finally:
        os.unlink(p)


def test_clean_python_reports_no_error():
    p = _tmp("pesc_clean.py")
    with open(p, "w", encoding="utf-8") as f:
        f.write("def f():\n    return 1\n")
    try:
        assert _syntax_check_written_file(p) is None
    finally:
        os.unlink(p)


def test_bad_json_reports_parse_error():
    p = _tmp("pesc_bad.json")
    with open(p, "w", encoding="utf-8") as f:
        f.write('{"a": 1,}')  # trailing comma → invalid JSON
    try:
        err = _syntax_check_written_file(p)
        assert err is not None and "JSON parse error" in err
    finally:
        os.unlink(p)


def test_clean_json_reports_no_error():
    p = _tmp("pesc_clean.json")
    with open(p, "w", encoding="utf-8") as f:
        f.write('{"a": 1}')
    try:
        assert _syntax_check_written_file(p) is None
    finally:
        os.unlink(p)


def test_unknown_extension_is_not_checked():
    p = _tmp("pesc_note.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("this is not code {{{")
    try:
        assert _syntax_check_written_file(p) is None
    finally:
        os.unlink(p)


def test_js_check_skipped_when_node_missing(monkeypatch):
    # node absent → .js check silently skips (returns None), never errors.
    monkeypatch.setattr(ft.shutil, "which", lambda name: None)
    called = {"ran": False}

    def _boom(*a, **k):
        called["ran"] = True
        raise AssertionError("subprocess must not run when node is missing")

    monkeypatch.setattr(ft.subprocess, "run", _boom)
    p = _tmp("pesc_skip.js")
    with open(p, "w", encoding="utf-8") as f:
        f.write("function ( {")  # broken JS, but node is 'missing'
    try:
        assert _syntax_check_written_file(p) is None
        assert called["ran"] is False
    finally:
        os.unlink(p)


def test_js_check_reports_error_when_node_present(monkeypatch):
    # Simulate node being present and reporting a syntax error, without
    # depending on node actually being installed.
    monkeypatch.setattr(ft.shutil, "which", lambda name: "/usr/bin/node")

    class _P:
        returncode = 1
        stderr = "SyntaxError: Unexpected token"
        stdout = ""

    monkeypatch.setattr(ft.subprocess, "run", lambda *a, **k: _P())
    p = _tmp("pesc_bad.js")
    with open(p, "w", encoding="utf-8") as f:
        f.write("function ( {")
    try:
        err = _syntax_check_written_file(p)
        assert err is not None and "SyntaxError" in err
    finally:
        os.unlink(p)


def test_checker_never_raises_on_subprocess_failure(monkeypatch):
    # If launching the checker blows up, the gate swallows it (returns None).
    def _raise(*a, **k):
        raise OSError("cannot spawn")

    monkeypatch.setattr(ft.subprocess, "run", _raise)
    p = _tmp("pesc_raise.py")
    with open(p, "w", encoding="utf-8") as f:
        f.write("def f(:\n")
    try:
        assert _syntax_check_written_file(p) is None
    finally:
        os.unlink(p)


# ── The setting gate on the augment wrapper ────────────────────────────────
def test_augment_disabled_by_setting(monkeypatch):
    import src.settings as settings
    monkeypatch.setattr(settings, "get_setting",
                        lambda key, default=None: False if key == "agent_post_edit_syntax_check" else default)
    called = {"ran": False}
    monkeypatch.setattr(ft, "_syntax_check_written_file",
                        lambda p: (called.__setitem__("ran", True) or "err"))
    res = {"output": "Wrote file", "exit_code": 0}
    out = _augment_with_syntax_check(res, _tmp("whatever.py"))
    assert "syntax_warning" not in out
    assert called["ran"] is False  # setting off → checker not even called


def test_augment_enabled_appends_warning(monkeypatch):
    import src.settings as settings
    monkeypatch.setattr(settings, "get_setting", lambda key, default=None: True)
    monkeypatch.setattr(ft, "_syntax_check_written_file", lambda p: "SyntaxError: bad")
    res = {"output": "Edited file", "exit_code": 0}
    out = _augment_with_syntax_check(res, _tmp("x.py"))
    assert out["syntax_warning"] == "SyntaxError: bad"
    assert "Syntax check FAILED" in out["output"]
    assert out["exit_code"] == 0  # non-fatal: still a success


# ── End-to-end through the real handlers ───────────────────────────────────
@pytest.mark.asyncio
async def test_write_file_bad_python_carries_warning():
    p = _ws("pesc_wf_bad.py")
    res = await WriteFileTool().execute(
        json.dumps({"path": p, "content": "def broken(:\n    pass\n"}), {}
    )
    try:
        # File is written regardless — success + non-fatal warning.
        assert res["exit_code"] == 0
        assert os.path.exists(p)
        assert "syntax_warning" in res
        assert "SyntaxError" in res["syntax_warning"] or "invalid syntax" in res["syntax_warning"]
    finally:
        if os.path.exists(p):
            os.unlink(p)


@pytest.mark.asyncio
async def test_write_file_clean_python_no_warning():
    p = _ws("pesc_wf_clean.py")
    res = await WriteFileTool().execute(
        json.dumps({"path": p, "content": "def ok():\n    return 1\n"}), {}
    )
    try:
        assert res["exit_code"] == 0
        assert "syntax_warning" not in res
    finally:
        os.unlink(p)


@pytest.mark.asyncio
async def test_edit_file_into_bad_python_carries_warning():
    p = _ws("pesc_ef_bad.py")
    with open(p, "w", encoding="utf-8") as f:
        f.write("def f():\n    return 1\n")
    res = await EditFileTool().execute(
        json.dumps({"path": p, "old_string": "return 1", "new_string": "return ("}), {}
    )
    try:
        assert res["exit_code"] == 0  # edit saved
        assert "syntax_warning" in res
    finally:
        os.unlink(p)


@pytest.mark.asyncio
async def test_edit_file_setting_off_skips_check(monkeypatch):
    import src.settings as settings
    monkeypatch.setattr(settings, "get_setting",
                        lambda key, default=None: False if key == "agent_post_edit_syntax_check" else default)
    p = _ws("pesc_ef_off.py")
    with open(p, "w", encoding="utf-8") as f:
        f.write("def f():\n    return 1\n")
    res = await EditFileTool().execute(
        json.dumps({"path": p, "old_string": "return 1", "new_string": "return ("}), {}
    )
    try:
        assert res["exit_code"] == 0
        assert "syntax_warning" not in res  # gate disabled
    finally:
        os.unlink(p)
