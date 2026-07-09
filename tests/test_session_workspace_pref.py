"""Workspace is a SESSION property, authoritative across devices — the same
conversation runs in the same working folder no matter where you open it.
Regression for the "workspace changed when I switched device" bug: the folder
lived in per-device localStorage and was posted per turn, so device B silently
ran the same conversation unconfined (or in a different folder).

Covers routes.chat_routes._resolve_request_workspace precedence: the persisted
prefs.workspace wins over the posted field; the posted field only seeds
sessions that never had the pref set.
"""

import pytest

from routes import chat_routes


class _Req:
    pass


@pytest.fixture(autouse=True)
def _admin_and_identity_vet(monkeypatch):
    # Privilege check passes; vet_workspace is identity for testable paths.
    import src.tool_security as ts
    import src.tool_execution as te
    monkeypatch.setattr(ts, "owner_is_admin_or_single_user", lambda user: True)
    monkeypatch.setattr(te, "vet_workspace", lambda p: p)
    monkeypatch.setattr(chat_routes, "get_current_user", lambda request: "admin")
    yield


def _with_pref(monkeypatch, value):
    monkeypatch.setattr(chat_routes, "_get_session_workspace_pref", lambda sid: value)


def test_session_pref_beats_posted_value(monkeypatch):
    _with_pref(monkeypatch, r"C:\projects\alpha")
    ws, rejected = chat_routes._resolve_request_workspace(_Req(), r"C:\other\folder", session_id="s1")
    assert ws == r"C:\projects\alpha"
    assert rejected == ""


def test_explicit_empty_pref_means_no_workspace(monkeypatch):
    # '' = the user explicitly cleared the workspace on some device: another
    # device's posted (stale) folder must NOT resurrect confinement.
    _with_pref(monkeypatch, "")
    ws, rejected = chat_routes._resolve_request_workspace(_Req(), r"C:\stale\folder", session_id="s1")
    assert ws == ""
    assert rejected == ""


def test_absent_pref_falls_back_to_posted(monkeypatch):
    _with_pref(monkeypatch, None)
    ws, rejected = chat_routes._resolve_request_workspace(_Req(), r"C:\projects\beta", session_id="s1")
    assert ws == r"C:\projects\beta"


def test_no_pref_no_posted_is_unconfined(monkeypatch):
    _with_pref(monkeypatch, None)
    ws, rejected = chat_routes._resolve_request_workspace(_Req(), "", session_id="s1")
    assert ws == ""
    assert rejected == ""
