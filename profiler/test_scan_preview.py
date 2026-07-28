"""Tests for the live scan viewer (file-tailing, headless rendering)."""

import os
import time

import numpy as np

import scan_preview


def _write_frame(path, value, mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.full((6, 8), value, dtype=np.uint8))
    saved = path if path.suffix == ".npy" else path.with_suffix(".npy")
    if mtime is not None:
        os.utime(saved, (mtime, mtime))
    return saved


def test_find_latest_frame_picks_newest_by_mtime(tmp_path):
    now = time.time()
    _write_frame(tmp_path / "y0100.00cm" / "old.npy", 10, mtime=now - 60)
    newest = _write_frame(tmp_path / "y0099.00cm" / "new.npy", 20, mtime=now)

    assert scan_preview.find_latest_frame(tmp_path) == newest


def test_find_latest_frame_empty_dir(tmp_path):
    assert scan_preview.find_latest_frame(tmp_path) is None


def test_newest_run_dir(tmp_path):
    old = tmp_path / "auto_scan-2026-07-27_10-00-00"
    new = tmp_path / "auto_scan-2026-07-28_10-00-00"
    old.mkdir()
    new.mkdir()
    now = time.time()
    os.utime(old, (now - 60, now - 60))
    os.utime(new, (now, now))
    (tmp_path / "not_a_run.txt").write_text("x")

    assert scan_preview.newest_run_dir(tmp_path) == new
    assert scan_preview.newest_run_dir(tmp_path / "missing_children") is None


def test_watch_headless_displays_frames_and_updates(tmp_path):
    now = time.time()
    _write_frame(tmp_path / "y0100.00cm" / "a.npy", 50, mtime=now - 10)
    _write_frame(tmp_path / "y0100.00cm" / "b.npy", 200, mtime=now)

    shown = scan_preview.watch(
        tmp_path, interval_s=0.01, display=False, max_ticks=3
    )

    # Newest frame shown once; unchanged directory on later ticks adds
    # nothing.
    assert shown == 1


def test_watch_tolerates_unreadable_midwrite_file(tmp_path):
    now = time.time()
    good = _write_frame(tmp_path / "good.npy", 50, mtime=now - 10)

    # A "frame" mid-write: newest by mtime but not loadable as .npy.
    bad = tmp_path / "partial.npy"
    bad.write_bytes(b"\x00\x01 not a real npy header")
    os.utime(bad, (now, now))

    shown = scan_preview.watch(
        tmp_path, interval_s=0.01, display=False, max_ticks=2
    )

    # Must not crash; the unreadable file is retried, never counted.
    assert shown == 0
    assert good.exists()
