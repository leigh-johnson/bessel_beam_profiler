"""
Non-blocking single-key reads from the launching terminal, so the free
alignment mode can be jogged from the COMMAND LINE — no need to focus
the preview window, and it works over SSH with --no-display.

POSIX only (termios/tty). On a non-tty stdin (piped input, some IDEs,
Windows) the reader silently degrades: poll() returns [] and the
preview-window keys remain the only input path.

Usage:

    with TerminalKeyReader() as keys:
        while streaming:
            for key in keys.poll():   # "up", "left", "q", ",", ...
                ...

The context manager puts the terminal in cbreak mode (keys arrive
immediately, unbuffered, no Enter needed; Ctrl-C still works) and
ALWAYS restores the saved terminal state on exit.
"""

from __future__ import annotations

from typing import Optional
import logging
import os
import sys

logger = logging.getLogger(__name__)


# Arrow keys arrive as 3-byte ANSI escape sequences (CSI A..D).
ESCAPE_SEQUENCES: dict[bytes, str] = {
    b"\x1b[A": "up",
    b"\x1b[B": "down",
    b"\x1b[C": "right",
    b"\x1b[D": "left",
    # Application-cursor mode variants (some terminals after tput smkx).
    b"\x1bOA": "up",
    b"\x1bOB": "down",
    b"\x1bOC": "right",
    b"\x1bOD": "left",
}


def parse_key_bytes(data: bytes) -> list[str]:
    """
    Decode a raw stdin read into key names: arrow escape sequences map
    to "up"/"down"/"left"/"right", printable characters map to
    themselves (case preserved), everything else is dropped.
    """

    keys: list[str] = []
    i = 0
    while i < len(data):
        if data[i:i + 1] == b"\x1b":
            sequence = data[i:i + 3]
            name = ESCAPE_SEQUENCES.get(sequence)
            if name is not None:
                keys.append(name)
                i += 3
                continue
            # Unknown escape: swallow the whole CSI/SS3 sequence
            # (ESC [ params final, e.g. Home = ESC[1~) so its bytes
            # don't decode as spurious printable keys.
            if data[i + 1:i + 2] in (b"[", b"O"):
                j = i + 2
                while j < len(data) and 0x30 <= data[j] <= 0x3F:
                    j += 1
                i = j + 1  # + the final byte
            else:
                i += 1  # lone ESC
            continue

        char = chr(data[i])
        i += 1
        if char.isprintable():
            keys.append(char)
    return keys


class TerminalKeyReader:
    """Context manager: cbreak-mode stdin with non-blocking poll()."""

    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stdin
        self._fd: Optional[int] = None
        self._saved_state = None
        self.active = False

    def __enter__(self) -> "TerminalKeyReader":
        try:
            import termios
            import tty

            fd = self.stream.fileno()
            if not os.isatty(fd):
                raise OSError("stdin is not a tty")

            self._saved_state = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            self._fd = fd
            self.active = True
        except Exception as ex:  # noqa: BLE001 - degrade, don't die
            logger.info(
                f"Terminal key input unavailable ({ex}); use the "
                "preview-window keys instead."
            )
            self.active = False
        return self

    def __exit__(self, *_exc) -> None:
        if self.active and self._fd is not None:
            import termios

            try:
                termios.tcsetattr(
                    self._fd, termios.TCSADRAIN, self._saved_state
                )
            except Exception as ex:  # noqa: BLE001
                logger.warning(f"Could not restore the terminal state: {ex}")
        self.active = False

    def poll(self) -> list[str]:
        """All keys pressed since the last poll (never blocks)."""

        if not self.active or self._fd is None:
            return []

        import select

        keys: list[str] = []
        while True:
            ready, _, _ = select.select([self._fd], [], [], 0)
            if not ready:
                break
            data = os.read(self._fd, 64)
            if not data:
                break
            keys.extend(parse_key_bytes(data))
        return keys
