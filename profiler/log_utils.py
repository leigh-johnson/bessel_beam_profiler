"""
Logging configuration for the profiler CLIs.

The CLI entry point calls configure_cli_logging() once (console, INFO by
default). Dataset subcommands additionally attach a FileHandler so the
same log lines land in a scan.log next to the data they describe — the
durable record for post-morteming long unattended runs.
"""

from __future__ import annotations

from pathlib import Path
import logging

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger(__name__)


def configure_cli_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )


def add_file_log(path: Path) -> logging.FileHandler:
    """
    Attach a FileHandler for `path` to the root logger and return it, so
    everything logged (any module, any level >= the root level) is also
    written to the file. Pair with remove_file_log when a command handles
    several run directories in one process.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    logging.getLogger().addHandler(handler)
    logger.info(f"Also logging to {path}")

    return handler


def remove_file_log(handler: logging.FileHandler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()
