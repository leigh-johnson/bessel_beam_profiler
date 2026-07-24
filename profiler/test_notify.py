"""
Tests for the Slack webhook notifier: payload format, success detection,
and — critically — that every failure mode logs a warning and returns
False instead of raising (a Slack outage must never abort a scan).
"""

import io
import json
import logging
import urllib.error

import pytest

import notify


class FakeResponse:
    def __init__(self, status=200, body=b"ok"):
        self.status = status
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_sends_expected_payload_and_returns_true(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode())
        captured["content_type"] = request.get_header("Content-type")
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)

    ok = notify.send_slack_message(
        ":bell: Scan with placement ID `jul22-001` finished.",
        "https://hooks.slack.com/services/T00/B00/xyz",
    )

    assert ok is True
    assert captured["url"] == "https://hooks.slack.com/services/T00/B00/xyz"
    assert captured["body"] == {
        "text": ":bell: Scan with placement ID `jul22-001` finished."
    }
    assert captured["content_type"] == "application/json"


def test_non_ok_response_warns_and_returns_false(monkeypatch, caplog):
    monkeypatch.setattr(
        notify.urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(status=404, body=b"no_service"),
    )

    with caplog.at_level(logging.WARNING, logger="notify"):
        ok = notify.send_slack_message("hi", "https://hooks.slack.com/x")

    assert ok is False
    assert any("404" in r.message for r in caplog.records)


def test_network_error_warns_and_returns_false(monkeypatch, caplog):
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)

    with caplog.at_level(logging.WARNING, logger="notify"):
        ok = notify.send_slack_message("hi", "https://hooks.slack.com/x")

    assert ok is False
    assert any("scan unaffected" in r.message for r in caplog.records)


def test_missing_webhook_url_warns_and_returns_false(caplog):
    with caplog.at_level(logging.WARNING, logger="notify"):
        ok = notify.send_slack_message("hi", "")

    assert ok is False
    assert any("no webhook URL" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Startup configuration notice
# ---------------------------------------------------------------------------


def test_config_notice_warns_when_nothing_is_set():
    level, message = notify.slack_config_notice(None, None, "#logs-bessel-beam")

    assert level == logging.WARNING
    assert "DISABLED" in message
    assert "SLACK_BOT_TOKEN" in message and "SLACK_WEBHOOK_URL" in message


def test_config_notice_reports_webhook_only_mode():
    level, message = notify.slack_config_notice(None, "https://hooks/x", "#c")

    assert level == logging.INFO
    assert "webhook" in message
    assert "streaming" in message  # mentions what is NOT available


def test_config_notice_reports_bot_token_mode():
    level, message = notify.slack_config_notice("xoxb-x", None, "#logs-bessel-beam")

    assert level == logging.INFO
    assert "thread streaming" in message
    assert "#logs-bessel-beam" in message


# ---------------------------------------------------------------------------
# Web API (chat.postMessage)
# ---------------------------------------------------------------------------


def make_api_response(payload: dict) -> FakeResponse:
    return FakeResponse(status=200, body=json.dumps(payload).encode())


def test_post_message_returns_ts_and_sends_thread_fields(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["auth"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data.decode())
        return make_api_response({"ok": True, "ts": "1753222222.000100"})

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)

    ts = notify.post_message(
        "xoxb-test",
        "#logs-bessel-beam",
        "hello",
        thread_ts="1753200000.000200",
        reply_broadcast=True,
    )

    assert ts == "1753222222.000100"
    assert captured["url"] == notify.SLACK_POST_MESSAGE_URL
    assert captured["auth"] == "Bearer xoxb-test"
    assert captured["body"]["channel"] == "#logs-bessel-beam"
    assert captured["body"]["thread_ts"] == "1753200000.000200"
    assert captured["body"]["reply_broadcast"] is True


def test_post_message_api_error_warns_and_returns_none(monkeypatch, caplog):
    monkeypatch.setattr(
        notify.urllib.request,
        "urlopen",
        lambda request, timeout: make_api_response(
            {"ok": False, "error": "channel_not_found"}
        ),
    )

    with caplog.at_level(logging.WARNING, logger="notify"):
        ts = notify.post_message("xoxb-test", "#nope", "hello")

    assert ts is None
    # The warning names the channel and gives an error-specific hint.
    assert any(
        "channel_not_found" in r.message and "#nope" in r.message
        for r in caplog.records
    )


def test_post_message_not_in_channel_hints_at_invite(monkeypatch, caplog):
    monkeypatch.setattr(
        notify.urllib.request,
        "urlopen",
        lambda request, timeout: make_api_response(
            {"ok": False, "error": "not_in_channel"}
        ),
    )

    with caplog.at_level(logging.WARNING, logger="notify"):
        assert notify.post_message("xoxb-test", "#logs-bessel-beam", "hi") is None

    assert any(
        "/invite @BesselBot in #logs-bessel-beam" in r.message
        for r in caplog.records
    )


def test_post_message_network_error_warns_and_returns_none(monkeypatch, caplog):
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)

    with caplog.at_level(logging.WARNING, logger="notify"):
        assert notify.post_message("xoxb-test", "#c", "hello") is None

    assert any("scan unaffected" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# SlackLogHandler (threaded log streaming)
# ---------------------------------------------------------------------------


def make_record(message: str, name: str = "auto_scan", level=logging.INFO):
    return logging.LogRecord(name, level, "path.py", 1, message, None, None)


def make_handler(monkeypatch, posts: list, post_result="1.0"):
    def fake_post(token, channel, text, thread_ts=None, reply_broadcast=False, timeout_s=10.0):
        posts.append({"text": text, "thread_ts": thread_ts})
        return post_result

    monkeypatch.setattr(notify, "post_message", fake_post)

    handler = notify.SlackLogHandler(
        "xoxb-test",
        "#logs-bessel-beam",
        thread_ts="1753200000.000200",
        start_worker=False,  # tests flush synchronously
        max_consecutive_failures=2,
    )
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    return handler


def test_handler_batches_records_into_one_thread_reply(monkeypatch):
    posts = []
    handler = make_handler(monkeypatch, posts)

    handler.emit(make_record("slice 1 done"))
    handler.emit(make_record("slice 2 done"))
    handler.flush_now()

    assert len(posts) == 1  # batched, not per-line
    assert posts[0]["thread_ts"] == "1753200000.000200"
    assert posts[0]["text"] == (
        "```INFO auto_scan: slice 1 done\nINFO auto_scan: slice 2 done```"
    )


def test_handler_skips_its_own_module_records(monkeypatch):
    posts = []
    handler = make_handler(monkeypatch, posts)

    handler.emit(make_record("Slack post failed", name="notify"))
    handler.flush_now()

    assert posts == []  # loop prevention


def test_handler_disables_itself_after_repeated_failures(monkeypatch, capsys):
    posts = []
    handler = make_handler(monkeypatch, posts, post_result=None)  # all fail

    handler.emit(make_record("a"))
    handler.flush_now()
    handler.emit(make_record("b"))
    handler.flush_now()

    assert handler._disabled is True
    assert "disabled after" in capsys.readouterr().err

    handler.emit(make_record("c"))  # dropped silently once disabled
    handler.flush_now()
    assert len(posts) == 2  # no further attempts


def test_chunking_splits_long_batches_on_line_boundaries():
    lines = ["x" * 1500 for _ in range(5)]

    chunks = notify.SlackLogHandler._chunk(lines)

    assert len(chunks) == 3  # 2+2+1 lines per <=3500-char chunk
    assert all(len(chunk) <= notify.MAX_MESSAGE_CHARS for chunk in chunks)
    assert sum(chunk.count("x") for chunk in chunks) == 7500
