"""Group-chat context fetching for @-mention wake.

Verifies the ``_fetch_chat_context`` / ``_fetch_channel_context`` family
that hydrates prior chat history into ``MessageEvent.channel_context``
when the bot is @-mentioned in a group / channel. The behaviour is
platform-specific:

  - BlueBubbles: REST GET on the chat GUID, in-memory TTL cache.
  - Slack: ``conversations.history`` on the channel, TTL cache.
  - Telegram: handled by the upstream ``observe_unmentioned_group_messages``
    mechanism; this file only verifies the wiring is intact.

These tests run against the adapters directly with the slack_bolt / ptb /
httpx transports replaced by mocks so no live credentials are required.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# BlueBubbles
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _make_bb_adapter():
    """Build a BlueBubblesAdapter wired to a mock REST client + config."""
    from gateway.platforms.bluebubbles import BlueBubblesAdapter

    adapter = BlueBubblesAdapter.__new__(BlueBubblesAdapter)
    adapter.server_url = "http://127.0.0.1:1234"
    adapter.password = "test-pw"
    adapter.client = MagicMock()
    adapter.bot_username = "Bianka"
    adapter.require_mention = True
    adapter.send_read_receipts = False
    from collections import OrderedDict
    adapter._chat_context_cache = OrderedDict()
    # Mention patterns matching "@bianka" so the test message is admitted.
    from gateway.platforms.helpers import compile_mention_patterns
    adapter._mention_patterns = compile_mention_patterns(
        [r"(?<![\w@])@?bianka\b[,:\-]?"],
        log_prefix="[bluebubbles:test] ",
    )
    return adapter


def test_bluebubbles_fetch_chat_context_returns_group_history():
    adapter = _make_bb_adapter()
    adapter.client.get = AsyncMock(
        return_value=_FakeResponse({
            "status": 200,
            "message": "Success",
            # BlueBubbles /chat/<guid>/message with sort=DESC returns
            # newest-first. ``_fetch_chat_context`` reverses the slice
            # so the rendered block is oldest-to-newest.
            "data": [
                {
                    "guid": "msg-003",
                    "text": "@bianka qué opinas?",
                    "isFromMe": False,
                    "handle": {"address": "+34666555444"},
                    "dateCreated": 1700000020000,
                },
                {
                    "guid": "msg-002",
                    "text": "segundo mensaje",
                    "isFromMe": False,
                    "handle": {"address": "+34611222333"},
                    "dateCreated": 1700000010000,
                },
                {
                    "guid": "msg-001",
                    "text": "primer mensaje del grupo",
                    "isFromMe": False,
                    "handle": {"address": "+34666555444"},
                    "dateCreated": 1700000000000,
                },
            ],
        })
    )

    content = asyncio.run(
        adapter._fetch_chat_context(
            chat_guid="any;+;chat42",
            current_message_guid="msg-003",
            limit=20,
        )
    )

    assert "primer mensaje del grupo" in content
    assert "segundo mensaje" in content
    assert "@bianka qué opinas" not in content  # trigger must be excluded
    assert "+34666555444" in content
    assert "+34611222333" in content
    # Must be chronological (oldest first), not reversed.
    pos1 = content.index("primer mensaje")
    pos2 = content.index("segundo mensaje")
    assert pos1 < pos2


def test_bluebubbles_fetch_chat_context_skips_dm():
    adapter = _make_bb_adapter()
    adapter.client.get = AsyncMock()

    content = asyncio.run(
        adapter._fetch_chat_context(
            chat_guid="any;-;+34611222333",  # DM, not a group
            current_message_guid="msg-x",
            limit=20,
        )
    )

    assert content == ""
    adapter.client.get.assert_not_called()


def test_bluebubbles_fetch_chat_context_caches_within_ttl():
    adapter = _make_bb_adapter()
    adapter.client.get = AsyncMock(
        return_value=_FakeResponse({
            "status": 200,
            "message": "Success",
            "data": [{
                "guid": "msg-a",
                "text": "hola",
                "isFromMe": False,
                "handle": {"address": "+34600000000"},
            }],
        })
    )
    guid = "any;+;chat99"

    first = asyncio.run(adapter._fetch_chat_context(chat_guid=guid))
    second = asyncio.run(adapter._fetch_chat_context(chat_guid=guid))

    # Second call served from cache, so only ONE outbound HTTP request.
    assert adapter.client.get.call_count == 1
    assert first == second


def test_bluebubbles_fetch_chat_context_handles_rest_error():
    adapter = _make_bb_adapter()
    adapter.client.get = AsyncMock(side_effect=RuntimeError("connection refused"))

    content = asyncio.run(
        adapter._fetch_chat_context(chat_guid="any;+;chat42")
    )

    assert content == ""  # graceful degradation


def test_bluebubbles_fetch_chat_context_renders_oldest_to_newest():
    """The rendered block is oldest-to-newest regardless of API sort order.

    BlueBubbles' ``/chat/<guid>/message`` returns DESC (newest-first);
    the fetcher reverses the kept slice so the rendered block reads
    oldest-to-newest (the natural reading order).
    """
    adapter = _make_bb_adapter()
    adapter.client.get = AsyncMock(
        return_value=_FakeResponse({
            "status": 200,
            "data": [
                # DESC order from the real endpoint — newest first.
                {"guid": "new", "text": "third", "isFromMe": False,
                 "handle": {"address": "+34600000003"}},
                {"guid": "mid", "text": "second", "isFromMe": False,
                 "handle": {"address": "+34600000002"}},
                {"guid": "old", "text": "first", "isFromMe": False,
                 "handle": {"address": "+34600000001"}},
            ],
        })
    )

    content = asyncio.run(
        adapter._fetch_chat_context(chat_guid="any;+;chat-order", limit=20)
    )

    pos_first = content.index("first")
    pos_second = content.index("second")
    pos_third = content.index("third")
    assert pos_first < pos_second < pos_third


def test_bluebubbles_webhook_skips_group_without_at_mention():
    """Group messages without @bianka must be silently dropped."""
    from gateway.platforms.bluebubbles import BlueBubblesAdapter

    adapter = _make_bb_adapter()
    adapter.client = MagicMock()
    # Stub build_source so we don't pull in session-resolution.
    adapter.build_source = lambda **kw: SimpleNamespace(**kw)
    adapter.handle_message = AsyncMock()

    request = SimpleNamespace(
        query={"password": "test-pw"},
        headers={},
    )

    async def fake_read():
        return b'{"type":"new-message","data":{"text":"hola gente","isFromMe":false,"chatGuid":"any;+;chat42","handle":{"address":"+34666555444"},"guid":"msg-z"}}'

    request.read = fake_read

    response = asyncio.run(adapter._handle_webhook(request))

    # Acknowledged without dispatching the agent.
    assert response.text == "ok"
    adapter.handle_message.assert_not_called()


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------


def _make_slack_adapter():
    """Build a SlackAdapter wired to a mock slack_sdk client."""
    from plugins.platforms.slack.adapter import SlackAdapter

    adapter = SlackAdapter.__new__(SlackAdapter)
    adapter._bot_user_id = "U_BIANKA"
    adapter._team_bot_user_ids = {"T10BBEYTC": "U_BIANKA"}
    adapter._channel_context_cache = {}
    adapter._CHANNEL_CONTEXT_CACHE_TTL = 30.0
    adapter._CHANNEL_CONTEXT_CACHE_MAX = 2500
    adapter._is_sender_authorized = lambda *_a, **_kw: True
    adapter._resolve_user_name = AsyncMock(return_value="someone")
    # Pull in the real _format_channel_context — it doesn't need slack_bolt.
    return adapter


def test_slack_channel_context_fetches_history_and_renders():
    adapter = _make_slack_adapter()
    fake_client = MagicMock()
    fake_client.conversations_history = AsyncMock(return_value={
        "messages": [
            {"ts": "1700000020.000", "text": "@bianka ayuda", "user": "U_HUMAN"},
            {"ts": "1700000010.000", "text": "primer mensaje", "user": "U_HUMAN2"},
            {"ts": "1700000005.000", "text": "otro mensaje", "user": "U_HUMAN"},
        ],
    })
    adapter._get_client = lambda *_a, **_kw: fake_client

    content = asyncio.run(
        adapter._fetch_channel_context(
            channel_id="C123",
            current_ts="1700000020.000",
            team_id="T10BBEYTC",
            limit=20,
        )
    )

    assert "primer mensaje" in content
    assert "otro mensaje" in content
    assert "@bianka ayuda" not in content  # current trigger must be excluded
    # The channel-context header must be present.
    assert "Channel context" in content


def test_slack_channel_context_skips_dm():
    adapter = _make_slack_adapter()
    adapter._get_client = MagicMock()

    content = asyncio.run(
        adapter._fetch_channel_context(
            channel_id="D_DIRECT_MSG",
            current_ts="1700000020.000",
        )
    )

    assert content == ""


def test_slack_channel_context_caches_within_ttl():
    adapter = _make_slack_adapter()
    fake_client = MagicMock()
    fake_client.conversations_history = AsyncMock(return_value={
        "messages": [{"ts": "1.0", "text": "x", "user": "U"}],
    })
    adapter._get_client = lambda *_a, **_kw: fake_client

    asyncio.run(adapter._fetch_channel_context(channel_id="C123", current_ts=""))
    asyncio.run(adapter._fetch_channel_context(channel_id="C123", current_ts=""))

    assert fake_client.conversations_history.call_count == 1


def test_slack_channel_context_handles_rate_limit_with_retry():
    adapter = _make_slack_adapter()
    fake_client = MagicMock()

    call_count = {"n": 0}

    async def flaky_history(*_a, **_kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("429 rate_limited")
        return {"messages": [{"ts": "1.0", "text": "x", "user": "U"}]}

    fake_client.conversations_history = flaky_history
    adapter._get_client = lambda *_a, **_kw: fake_client

    content = asyncio.run(
        adapter._fetch_channel_context(channel_id="C123", current_ts="")
    )

    assert "Channel context" in content
    assert call_count["n"] == 2  # one retry after rate-limit


# ---------------------------------------------------------------------------
# Telegram (smoke test only — observe-mode is upstream-tested)
# ---------------------------------------------------------------------------


def test_telegram_observer_config_field_exists():
    """The Telegram adapter reads ``observe_unmentioned_group_messages`` from config.

    This test guards against accidental rename of that config key.
    """
    from plugins.platforms.telegram import adapter as tg_adapter

    src = open(tg_adapter.__file__).read()
    assert "observe_unmentioned_group_messages" in src
    assert "_observe_unmentioned_group_message" in src
    assert "_apply_telegram_group_observe_attribution" in src