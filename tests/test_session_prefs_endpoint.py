"""Session prefs endpoints (GET/POST /api/session/{id}/prefs) — the small
free-form per-session preference blob added for the reasoning-effort
selector. Follows the isolated in-memory-SQLite + direct-route-call pattern
from test_archived_sessions_model_filter.py.
"""
import asyncio
import sys
import tempfile
import types
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import configure_mappers, sessionmaker
from sqlalchemy.pool import NullPool

import core.database as cdb
from core.database import Session as DbSession

# Force eager SQLAlchemy mapper configuration NOW, against the registry that is
# current at import time. Without this, `Session.messages` (a
# relationship("ChatMessage", ...) string reference) resolves lazily on first
# use — and some other test file in the suite does
# `importlib.reload(core.database)`, which creates a fresh Base/registry that
# never gets a chance to fully configure before this file's own DbSession
# instances trigger a lazy resolve against a half-configured mapper state,
# raising "expression 'ChatMessage' failed to locate a name" (order-dependent
# failure, not a regression in the new prefs endpoint itself — see
# test_truncate_message_count_regression.py's _make_manager for the reload).
configure_mappers()

_TMPDB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_ENGINE = create_engine(
    f"sqlite:///{_TMPDB.name}",
    connect_args={"check_same_thread": False},
    poolclass=NullPool,
)
cdb.Base.metadata.create_all(_ENGINE)
_TS = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False)


def _route(router, path, method="GET"):
    for r in router.routes:
        if r.path == path and method in getattr(r, "methods", set()):
            return r.endpoint
    raise AssertionError(f"route not found: {method} {path}")


def _stub_multipart_if_missing(monkeypatch):
    """See test_archived_sessions_model_filter.py — setup_session_routes()
    registers Form()-based routes that otherwise require python-multipart
    to be importable just to register, even though this test never posts
    multipart data itself."""
    try:
        import python_multipart  # noqa: F401
        return
    except ImportError:
        pass
    stub = types.ModuleType("python_multipart")
    stub.__version__ = "0.0.20"
    monkeypatch.setitem(sys.modules, "python_multipart", stub)


def _run(coro_or_value):
    """set_session_prefs/get_session_prefs handlers are async; the archived-
    sessions test's endpoints happen to be sync. Await if we got a coroutine.

    Uses asyncio.run() (a fresh, private event loop per call) rather than
    asyncio.get_event_loop().run_until_complete() — the latter depends on
    ambient "current event loop" state that another test earlier in the same
    pytest process may have left closed/inconsistent, which silently produced
    a "coroutine was never awaited" RuntimeWarning and a None result here
    instead of actually running the handler (order-dependent flakiness, not a
    bug in the prefs endpoint itself)."""
    if asyncio.iscoroutine(coro_or_value):
        return asyncio.run(coro_or_value)
    return coro_or_value


class _FakeRequest:
    """Minimal stand-in for FastAPI's Request — only .json() is used by
    set_session_prefs; GET doesn't touch the request body at all."""
    def __init__(self, body=None):
        self._body = body

    async def json(self):
        if self._body is _RAISE:
            raise ValueError("bad json")
        return self._body


_RAISE = object()


@pytest.fixture
def prefs_endpoints(monkeypatch):
    import routes.session_routes as sr
    from unittest.mock import MagicMock

    _stub_multipart_if_missing(monkeypatch)
    monkeypatch.setattr(sr, "SessionLocal", _TS)
    monkeypatch.setattr(sr, "effective_user", lambda request: "alice")
    router = sr.setup_session_routes(MagicMock(), {})
    get_fn = _route(router, "/api/session/{session_id}/prefs", "GET")
    post_fn = _route(router, "/api/session/{session_id}/prefs", "POST")
    return get_fn, post_fn


def _seed_session(owner="alice", prefs=None):
    sid = str(uuid.uuid4())
    db = _TS()
    try:
        db.add(DbSession(id=sid, owner=owner, name="chat", endpoint_url="http://localhost",
                          model="gpt-4o", prefs=prefs))
        db.commit()
    finally:
        db.close()
    return sid


def test_get_prefs_defaults_to_empty_dict(prefs_endpoints):
    get_fn, _ = prefs_endpoints
    sid = _seed_session()
    res = _run(get_fn(request=_FakeRequest(), session_id=sid))
    assert res == {"prefs": {}}


def test_post_sets_reasoning_effort(prefs_endpoints):
    get_fn, post_fn = prefs_endpoints
    sid = _seed_session()
    res = _run(post_fn(request=_FakeRequest({"reasoning_effort": "high"}), session_id=sid))
    assert res == {"status": "success", "prefs": {"reasoning_effort": "high"}}
    # Persisted — a fresh GET sees it.
    res2 = _run(get_fn(request=_FakeRequest(), session_id=sid))
    assert res2 == {"prefs": {"reasoning_effort": "high"}}


def test_post_merges_rather_than_replaces(prefs_endpoints):
    get_fn, post_fn = prefs_endpoints
    sid = _seed_session(prefs={"some_future_key": "keep-me"})
    _run(post_fn(request=_FakeRequest({"reasoning_effort": "low"}), session_id=sid))
    res = _run(get_fn(request=_FakeRequest(), session_id=sid))
    assert res == {"prefs": {"some_future_key": "keep-me", "reasoning_effort": "low"}}


def test_post_normalizes_case_and_whitespace(prefs_endpoints):
    get_fn, post_fn = prefs_endpoints
    sid = _seed_session()
    _run(post_fn(request=_FakeRequest({"reasoning_effort": "  MEDIUM  "}), session_id=sid))
    res = _run(get_fn(request=_FakeRequest(), session_id=sid))
    assert res == {"prefs": {"reasoning_effort": "medium"}}


def test_post_rejects_invalid_effort_value(prefs_endpoints):
    _, post_fn = prefs_endpoints
    sid = _seed_session()
    with pytest.raises(Exception) as exc_info:
        _run(post_fn(request=_FakeRequest({"reasoning_effort": "extreme"}), session_id=sid))
    assert getattr(exc_info.value, "status_code", None) == 400


def test_post_ignores_unrecognized_keys(prefs_endpoints):
    _, post_fn = prefs_endpoints
    sid = _seed_session()
    with pytest.raises(Exception) as exc_info:
        _run(post_fn(request=_FakeRequest({"unknown_key": "value"}), session_id=sid))
    assert getattr(exc_info.value, "status_code", None) == 400


def test_post_rejects_non_object_body(prefs_endpoints):
    _, post_fn = prefs_endpoints
    sid = _seed_session()
    with pytest.raises(Exception) as exc_info:
        _run(post_fn(request=_FakeRequest(["not", "a", "dict"]), session_id=sid))
    assert getattr(exc_info.value, "status_code", None) == 400


def test_post_rejects_invalid_json(prefs_endpoints):
    _, post_fn = prefs_endpoints
    sid = _seed_session()
    with pytest.raises(Exception) as exc_info:
        _run(post_fn(request=_FakeRequest(_RAISE), session_id=sid))
    assert getattr(exc_info.value, "status_code", None) == 400


def test_get_missing_session_404s(prefs_endpoints):
    get_fn, _ = prefs_endpoints
    with pytest.raises(Exception) as exc_info:
        _run(get_fn(request=_FakeRequest(), session_id="does-not-exist"))
    assert getattr(exc_info.value, "status_code", None) == 404


def test_post_missing_session_404s(prefs_endpoints):
    _, post_fn = prefs_endpoints
    with pytest.raises(Exception) as exc_info:
        _run(post_fn(request=_FakeRequest({"reasoning_effort": "high"}), session_id="does-not-exist"))
    assert getattr(exc_info.value, "status_code", None) == 404
