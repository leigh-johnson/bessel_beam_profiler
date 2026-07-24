"""
Slack notifications for scan events: webhook pings, Web API messages, and
threaded log streaming.

Two transports:

  * Incoming webhook (SLACK_WEBHOOK_URL): simplest ping. Webhooks can post
    INTO an existing thread but can never START one — the response never
    reveals the created message's ts, and threads are keyed by the
    parent's ts.
  * Bot token (SLACK_BOT_TOKEN, chat:write scope, bot invited to the
    channel): chat.postMessage returns ts, enabling the thread pattern —
    one "scan started" parent per placement, log lines streamed as
    batched replies, and the finish ping as a reply_broadcast (thread +
    channel).

Both credentials are SECRETS: environment variables or flags, never the
repo.

SlackLogHandler streams logging records into a thread: batched (one Slack
message per ~10 s, not per line — rate limits), asynchronous (a daemon
worker posts; a wedged network never stalls the scan loop), loop-proof
(records from THIS module's logger are skipped, so a failed post's
warning cannot recurse), and self-disabling after repeated failures.

Failures everywhere are logged and swallowed BY DESIGN — Slack being down
must never abort a running scan. stdlib only, no dependencies.
"""

from __future__ import annotations

from typing import Optional
import json
import logging
import queue
import sys
import threading
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

SLACK_POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"

# Slack's text limit is 40k, but huge messages render badly; chunk batches.
MAX_MESSAGE_CHARS = 3500


def send_slack_message(text: str, webhook_url: str, timeout_s: float = 10.0) -> bool:
    """
    POST a message to a Slack incoming webhook. Returns True when Slack
    acknowledged it; False (with a logged warning) on any failure.
    """

    if not webhook_url:
        logger.warning("Slack notification skipped: no webhook URL configured.")
        return False

    request = urllib.request.Request(
        webhook_url,
        data=json.dumps({"text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8", errors="replace").strip()

            if response.status == 200 and body == "ok":
                logger.info(f"Slack notification sent: {text}")
                return True

            logger.warning(
                f"Slack webhook returned HTTP {response.status}: {body[:200]}"
            )
            return False

    except (urllib.error.URLError, OSError) as ex:
        logger.warning(f"Slack notification failed (scan unaffected): {ex}")
        return False


def slack_config_notice(
    bot_token: Optional[str],
    webhook_url: Optional[str],
    channel: str,
) -> tuple[int, str]:
    """
    One-line description of the active Slack notification setup, for
    logging at scan startup. Returns (logging level, message) — WARNING
    when notifications are entirely disabled, so a forgotten env var in a
    fresh shell is visible before a multi-hour scan starts.
    """

    if bot_token:
        return (
            logging.INFO,
            f"Slack: thread streaming + pings to {channel} (bot token set).",
        )

    if webhook_url:
        return (
            logging.INFO,
            "Slack: finish/failure pings via webhook (no bot token — log "
            "streaming and threads disabled).",
        )

    return (
        logging.WARNING,
        "Slack notifications DISABLED: neither SLACK_BOT_TOKEN nor "
        "SLACK_WEBHOOK_URL is set in this shell. No ping will be sent "
        "when the placement finishes.",
    )


def post_message(
    token: str,
    channel: str,
    text: str,
    thread_ts: Optional[str] = None,
    reply_broadcast: bool = False,
    timeout_s: float = 10.0,
) -> Optional[str]:
    """
    chat.postMessage via the Web API. Returns the posted message's ts
    (usable as thread_ts for replies), or None (with a logged warning) on
    any failure.
    """

    if not token:
        logger.warning("Slack post skipped: no bot token configured.")
        return None

    payload: dict = {"channel": channel, "text": text}

    if thread_ts is not None:
        payload["thread_ts"] = thread_ts
        payload["reply_broadcast"] = reply_broadcast

    request = urllib.request.Request(
        SLACK_POST_MESSAGE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {token}",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as ex:
        logger.warning(f"Slack post failed (scan unaffected): {ex}")
        return None

    if not body.get("ok"):
        error = body.get("error", body)
        hint = {
            "not_in_channel": "the bot is not a member — run /invite "
            f"@BesselBot in {channel}",
            "channel_not_found": f"no channel named {channel} visible to "
            "the bot — check --slack-channel / SLACK_CHANNEL",
            "invalid_auth": "the bot token is wrong or revoked",
            "missing_scope": "the token lacks the chat:write scope — add "
            "it and reinstall the app",
        }.get(
            error,
            "check the bot token scope chat:write and that the bot is "
            "invited to the channel",
        )
        logger.warning(
            f"Slack API rejected the message to {channel}: {error} ({hint})."
        )
        return None

    return body.get("ts")


class SlackLogHandler(logging.Handler):
    """
    Stream logging records into a Slack thread as batched code-block
    replies. See the module docstring for the safety properties.
    """

    def __init__(
        self,
        token: str,
        channel: str,
        thread_ts: str,
        level: int = logging.INFO,
        flush_interval_s: float = 10.0,
        max_consecutive_failures: int = 5,
        start_worker: bool = True,
    ):
        super().__init__(level)

        self.token = token
        self.channel = channel
        self.thread_ts = thread_ts
        self.flush_interval_s = flush_interval_s
        self.max_consecutive_failures = max_consecutive_failures

        self._queue: "queue.SimpleQueue[str]" = queue.SimpleQueue()
        self._stop = threading.Event()
        self._consecutive_failures = 0
        self._disabled = False
        self._worker: Optional[threading.Thread] = None

        if start_worker:
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="slack-log-streamer",
                daemon=True,
            )
            self._worker.start()

    # -- logging.Handler interface -------------------------------------

    def emit(self, record: logging.LogRecord) -> None:
        # Loop prevention: this module logs warnings when Slack posts
        # fail; shipping those to Slack would recurse forever.
        if record.name == __name__ or self._disabled:
            return

        try:
            self._queue.put(self.format(record))
        except Exception:  # noqa: BLE001 - logging must never raise
            self.handleError(record)

    def close(self) -> None:
        self._stop.set()

        if self._worker is not None:
            self._worker.join(timeout=5.0)

        self.flush_now()  # final drain of anything still queued
        super().close()

    # -- worker ---------------------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            stopped = self._stop.wait(timeout=self.flush_interval_s)
            self.flush_now()

            if stopped:
                return

    def flush_now(self) -> None:
        """Drain the queue and post everything as chunked replies."""

        lines: list[str] = []

        while True:
            try:
                lines.append(self._queue.get_nowait())
            except queue.Empty:
                break

        if not lines or self._disabled:
            return

        for chunk in self._chunk(lines):
            ts = post_message(
                self.token,
                self.channel,
                f"```{chunk}```",
                thread_ts=self.thread_ts,
            )

            if ts is None:
                self._consecutive_failures += 1

                if (
                    self._consecutive_failures >= self.max_consecutive_failures
                    and not self._disabled
                ):
                    self._disabled = True
                    # stderr, not logging: this handler must not feed itself.
                    print(
                        "SlackLogHandler: disabled after "
                        f"{self._consecutive_failures} consecutive failures; "
                        "scan continues, logs remain in scan.log.",
                        file=sys.stderr,
                    )
                return

            self._consecutive_failures = 0

    @staticmethod
    def _chunk(lines: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        size = 0

        for line in lines:
            if current and size + len(line) + 1 > MAX_MESSAGE_CHARS:
                chunks.append("\n".join(current))
                current, size = [], 0

            current.append(line[:MAX_MESSAGE_CHARS])
            size += len(line) + 1

        if current:
            chunks.append("\n".join(current))

        return chunks
