"""Mode is a SESSION property, authoritative across devices — not a per-browser
toggle. Regression for the "opened it on my phone and lost all the tools" bug:
a device whose local toggle defaulted to chat could silently downgrade an agent
session because the server trusted the client's posted mode.

Covers routes.chat_routes._resolve_session_mode (the precedence rule the send
path uses at request time).
"""

import pytest

from routes.chat_routes import _resolve_session_mode


@pytest.mark.parametrize("persisted,client,expected", [
    # Existing session: the PERSISTED mode wins, whatever the device sent.
    ("agent", "chat", "agent"),   # the actual bug: phone sent chat, session is agent
    ("agent", "agent", "agent"),
    ("chat", "agent", "chat"),    # symmetric: a chat session isn't upgraded by a stray agent toggle
    ("chat", "chat", "chat"),
    # Brand-new / no persisted mode yet: the client's field SEEDS the session.
    (None, "agent", "agent"),
    (None, "chat", "chat"),
    ("", "agent", "agent"),
    # 'research'/'research_pending' aren't agent/chat → fall through to client.
    ("research", "agent", "agent"),
    ("research_pending", "chat", "chat"),
    # Nothing usable anywhere → safe default is chat (no tools).
    (None, "", "chat"),
    (None, "garbage", "chat"),
])
def test_resolve_session_mode_prefers_persisted(persisted, client, expected):
    assert _resolve_session_mode(persisted, client) == expected
