"""Regression: single-message persist path had no ordering-tie guard.

chat_messages has no sequence column — history replay relies solely on
``ORDER BY timestamp``. The bulk ``replace_messages()`` path already guards
against same-instant collisions by offsetting each row with
``timedelta(microseconds=i)``, but the single-message ``_persist_message()``
path used a bare ``datetime.utcnow()`` with no check against the session's
last persisted timestamp — two messages persisted in the same microsecond (or
one persisted with a clock that moved backward relative to the last row) would
tie or invert on reload.

Fix: before writing, ``_persist_message`` now queries the session's most
recent persisted timestamp and, if the new message's timestamp would tie or
precede it, bumps it 1 microsecond past that last timestamp.
"""
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

from core.models import ChatMessage
from core.session_manager import SessionManager
import core.session_manager as SM


def _manager_with(sessions):
    manager = SessionManager.__new__(SessionManager)
    manager.sessions = dict(sessions)
    return manager


def _session_local(parent_row, last_timestamp):
    """Mock SessionLocal() whose query chain answers both queries
    _persist_message makes: the parent-session lookup (query().filter().first())
    and the last-timestamp lookup (query().filter().order_by().limit().scalar())."""
    db = MagicMock()

    def _query(*args, **kwargs):
        q = MagicMock()
        # Parent-session lookup: db.query(DbSession).filter(...).first()
        q.filter.return_value.first.return_value = parent_row
        # Last-timestamp lookup: db.query(DbChatMessage.timestamp).filter(...)
        #   .order_by(...).limit(1).scalar()
        q.filter.return_value.order_by.return_value.limit.return_value.scalar.return_value = last_timestamp
        return q

    db.query.side_effect = _query
    return MagicMock(return_value=db), db


def test_persist_message_bumps_timestamp_past_tie(monkeypatch):
    """A new message whose computed timestamp ties the last persisted row's
    timestamp must be bumped 1us past it, not written as a tie."""
    parent = SimpleNamespace(message_count=0, last_accessed=None, last_message_at=None)
    tie_time = datetime(2026, 7, 6, 12, 0, 0, 500000)
    session_local, db = _session_local(parent, last_timestamp=tie_time)
    monkeypatch.setattr(SM, "SessionLocal", session_local)
    monkeypatch.setattr(SM, "datetime", SimpleNamespace(
        utcnow=lambda: tie_time,
        now=datetime.now,
    ))

    message = ChatMessage("assistant", "second message, same instant")
    manager = _manager_with({"sid": SimpleNamespace(history=[message])})

    manager._persist_message("sid", message)

    db.add.assert_called_once()
    added_row = db.add.call_args[0][0]
    assert added_row.timestamp == tie_time + timedelta(microseconds=1)
    assert added_row.timestamp > tie_time


def test_persist_message_bumps_timestamp_when_clock_goes_backward(monkeypatch):
    """A new message computed EARLIER than the last persisted row (clock skew /
    fast successive calls) must still be bumped strictly past the last row,
    not merely left at its own (earlier) value."""
    parent = SimpleNamespace(message_count=0, last_accessed=None, last_message_at=None)
    later_last_time = datetime(2026, 7, 6, 12, 0, 1, 0)
    earlier_new_time = datetime(2026, 7, 6, 12, 0, 0, 0)
    session_local, db = _session_local(parent, last_timestamp=later_last_time)
    monkeypatch.setattr(SM, "SessionLocal", session_local)
    monkeypatch.setattr(SM, "datetime", SimpleNamespace(
        utcnow=lambda: earlier_new_time,
        now=datetime.now,
    ))

    message = ChatMessage("assistant", "arrives with a stale/behind clock reading")
    manager = _manager_with({"sid": SimpleNamespace(history=[message])})

    manager._persist_message("sid", message)

    added_row = db.add.call_args[0][0]
    assert added_row.timestamp == later_last_time + timedelta(microseconds=1)


def test_persist_message_leaves_timestamp_alone_when_strictly_after_last(monkeypatch):
    """The common case — plenty of wall-clock time since the last message —
    must NOT be perturbed; the natural timestamp is used as-is."""
    parent = SimpleNamespace(message_count=0, last_accessed=None, last_message_at=None)
    last_time = datetime(2026, 7, 6, 12, 0, 0, 0)
    new_time = datetime(2026, 7, 6, 12, 5, 0, 0)
    session_local, db = _session_local(parent, last_timestamp=last_time)
    monkeypatch.setattr(SM, "SessionLocal", session_local)
    monkeypatch.setattr(SM, "datetime", SimpleNamespace(
        utcnow=lambda: new_time,
        now=datetime.now,
    ))

    message = ChatMessage("user", "arrives well after the last message")
    manager = _manager_with({"sid": SimpleNamespace(history=[message])})

    manager._persist_message("sid", message)

    added_row = db.add.call_args[0][0]
    assert added_row.timestamp == new_time


def test_persist_message_no_prior_rows_uses_natural_timestamp(monkeypatch):
    """First message in a session — no prior row to tie against — must use the
    natural timestamp unchanged (last_timestamp=None is the empty-history case)."""
    parent = SimpleNamespace(message_count=0, last_accessed=None, last_message_at=None)
    new_time = datetime(2026, 7, 6, 12, 0, 0, 0)
    session_local, db = _session_local(parent, last_timestamp=None)
    monkeypatch.setattr(SM, "SessionLocal", session_local)
    monkeypatch.setattr(SM, "datetime", SimpleNamespace(
        utcnow=lambda: new_time,
        now=datetime.now,
    ))

    message = ChatMessage("user", "first message in a brand new session")
    manager = _manager_with({"sid": SimpleNamespace(history=[message])})

    manager._persist_message("sid", message)

    added_row = db.add.call_args[0][0]
    assert added_row.timestamp == new_time
