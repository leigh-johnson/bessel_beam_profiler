"""
Tests for the raw-terminal key reader used by the free alignment mode:
the pure bytes -> key-name parser, plus the reader's graceful
degradation on a non-tty stdin.
"""

import io

from term_keys import TerminalKeyReader, parse_key_bytes


def test_parse_arrows_letters_and_punctuation():
    data = b"\x1b[Aq,.-=eEa\x1b[D"
    assert parse_key_bytes(data) == [
        "up", "q", ",", ".", "-", "=", "e", "E", "a", "left",
    ]


def test_parse_application_cursor_variants():
    assert parse_key_bytes(b"\x1bOA\x1bOD") == ["up", "left"]


def test_parse_swallows_unknown_escapes_and_control_bytes():
    # Home (ESC[H) and End (ESC[4~) are unmapped: the WHOLE sequence is
    # swallowed — no spurious "[", "H", "4", "~" keys. Control bytes
    # (newline, Ctrl-C) are dropped; printables survive.
    assert parse_key_bytes(b"\x1b[Hq\x1b[4~\n\x03r") == ["q", "r"]


def test_parse_lone_escape_is_dropped():
    assert parse_key_bytes(b"\x1bq") == ["q"]


def test_reader_degrades_gracefully_on_non_tty():
    with TerminalKeyReader(stream=io.StringIO()) as reader:
        assert reader.active is False
        assert reader.poll() == []
