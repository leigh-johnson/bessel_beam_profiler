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
