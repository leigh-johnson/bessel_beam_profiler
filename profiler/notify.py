"""
Slack incoming-webhook notifications for scan events.

The webhook URL is a SECRET (anyone holding it can post to the channel):
it lives in the SLACK_WEBHOOK_URL environment variable or the
--slack-webhook flag, never in the repo. Webhook URLs are bound to the
channel chosen when the webhook was created (e.g. #proj-bessel-beam), so
no channel is specified here.

Failures are logged and swallowed BY DESIGN — a Slack outage or bad URL
must never abort a running scan. stdlib only (urllib), no dependencies.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


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
