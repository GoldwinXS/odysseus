"""Reliability regressions for background shell jobs (see src/bg_jobs.py,
src/bg_monitor.py, src/tool_execution.py launch path).

Covers three bugs seen live:
  * a long-lived server (`python -m http.server`) was reaped at the 1h cap and
    reported "failed" — a server staying up is success, not failure;
  * a job launched inside a sub-agent was keyed to that sub-agent's ephemeral
    steer-queue session, which is torn down when the sub-agent ends, so the
    follow-up target vanished;
  * that vanished session made bg_monitor's get_session raise KeyError, which
    (uncaught) left followed_up=False and retried the same job every tick.
"""

import asyncio
import json

import pytest

from src import bg_jobs, bg_monitor
from src import subagent_runs


# --- server / long-running command detection --------------------------------

@pytest.mark.parametrize("cmd", [
    "python -m http.server 1338",
    "python3 -m http.server",
    "uvicorn app:app --port 7000",
    "npm run dev",
    "pnpm dev",
    "yarn start",
    "npx vite",
    "flask run",
    "php -S 0.0.0.0:8000",
    "node server.js",
    "streamlit run app.py",
    "cd C:\\ClaudeSessions\\webrts\npython -m http.server 1338",  # server on line 2
])
def test_looks_long_running_detects_servers(cmd):
    assert bg_jobs.looks_long_running(cmd) is True


@pytest.mark.parametrize("cmd", [
    "ls -la",
    "python train.py",
    "pip install numpy",
    "git status",
    "echo http.server",              # mentions it, doesn't launch it
    "python manage.py migrate",
    "npm install",
    "python -m pytest tests/",
    "python train.py\necho \"starting uvicorn soon\"",  # server name only in prose
    "",
])
def test_looks_long_running_rejects_normal_commands(cmd):
    assert bg_jobs.looks_long_running(cmd) is False


@pytest.mark.parametrize("cmd", [
    "flask --app app run",
    "flask run",
    "python -m flask run",
    "python manage.py runserver",
    "python -m streamlit run x.py",
    "python -m gunicorn app:app",
])
def test_looks_long_running_detects_more_server_launchers(cmd):
    assert bg_jobs.looks_long_running(cmd) is True


# --- server detection in PYTHON SOURCE (the ```python``` tool) ---------------

@pytest.mark.parametrize("src", [
    "import socketserver\ns=socketserver.TCPServer(('',8000),h)\ns.serve_forever()",
    "app.run(host='0.0.0.0', port=5000)",
    "import uvicorn; uvicorn.run(app)",
    "from http.server import HTTPServer\nHTTPServer(('',8000),H).serve_forever()",
    "loop.run_forever()",
])
def test_looks_long_running_python_detects_servers(src):
    assert bg_jobs.looks_long_running_python(src) is True


@pytest.mark.parametrize("src", [
    "print(1 + 1)",
    "def serve_forever(): pass",    # a definition, not a call — must NOT match
    "x = [i for i in range(10)]",
    "df.run(mode=1)",               # unrelated .run() without a port
    "",
])
def test_looks_long_running_python_rejects_compute(src):
    assert bg_jobs.looks_long_running_python(src) is False


# --- runaway reaper leaves uncapped servers alone ---------------------------

def _seed_store(tmp_path, monkeypatch, rec):
    store = tmp_path / "bg_jobs.json"
    store.write_text(json.dumps({rec["id"]: rec}), encoding="utf-8")
    monkeypatch.setattr(bg_jobs, "_STORE", store)
    monkeypatch.setattr(bg_jobs, "_JOBS_DIR", tmp_path)
    # Never touch real processes in a unit test.
    monkeypatch.setattr(bg_jobs, "_kill", lambda pid: None)
    monkeypatch.setattr(bg_jobs, "_pid_alive", lambda pid: True)
    return store


def test_uncapped_server_is_not_reaped(tmp_path, monkeypatch):
    rec = {
        "id": "srv1", "session_id": "chat-1", "command": "python -m http.server 1339",
        "status": "running", "pid": 4242, "started_at": 0.0, "ended_at": None,
        "exit_code": None, "max_runtime_s": 0, "followed_up": False,
        "exit_path": str(tmp_path / "srv1.exit"), "log_path": str(tmp_path / "srv1.log"),
    }
    _seed_store(tmp_path, monkeypatch, rec)
    jobs = bg_jobs.refresh()
    # max_runtime_s=0 means uncapped: even started_at=0 (long past) must NOT reap.
    assert jobs["srv1"]["status"] == "running"
    assert not jobs["srv1"].get("timed_out")


def test_capped_runaway_is_still_reaped(tmp_path, monkeypatch):
    rec = {
        "id": "run1", "session_id": "chat-1", "command": "python train.py",
        "status": "running", "pid": 4242, "started_at": 0.0, "ended_at": None,
        "exit_code": None, "max_runtime_s": 1, "followed_up": False,
        "exit_path": str(tmp_path / "run1.exit"), "log_path": str(tmp_path / "run1.log"),
    }
    _seed_store(tmp_path, monkeypatch, rec)
    jobs = bg_jobs.refresh()
    assert jobs["run1"]["status"] == "failed"
    assert jobs["run1"].get("timed_out") is True


# --- ephemeral queue session -> real parent ---------------------------------

def test_parent_session_for_queue_resolves_and_guards():
    subagent_runs._UPDATES.clear()
    subagent_runs._UPDATES["parent-chat"] = [{"id": "sub_1", "queue_session": "_subagent_sub_1"}]
    try:
        assert subagent_runs.parent_session_for_queue("_subagent_sub_1") == "parent-chat"
        assert subagent_runs.parent_session_for_queue("_subagent_unknown") is None
        # A real chat id is not a queue session — never reverse-mapped.
        assert subagent_runs.parent_session_for_queue("parent-chat") is None
        assert subagent_runs.parent_session_for_queue(None) is None
    finally:
        subagent_runs._UPDATES.clear()


# --- monitor no longer loops forever on a missing session -------------------

def test_followup_handles_missing_session_without_looping(monkeypatch):
    class _RaisingSM:
        def get_session(self, sid):
            raise KeyError(sid)  # unknown id -> DB load -> KeyError (real behavior)

    monkeypatch.setattr("src.ai_interaction.get_session_manager", lambda: _RaisingSM())
    # A vanished ephemeral session must be treated as "handled" (return True) so
    # mark_followed_up fires and the monitor stops retrying it every tick.
    handled = asyncio.run(bg_monitor._run_followup(
        {"id": "j1", "session_id": "_subagent_sub_9"}
    ))
    assert handled is True
