"""
Live viewer for auto-scan runs — `dataset watch [RUN_DIR]`.

Tails a run directory and displays the newest saved frame (.npy) in a
matplotlib window, refreshing as the scan writes files. It runs as its
OWN process and only ever READS the filesystem: it never touches the
camera or the gantry, so it cannot block, slow, or crash the capture
loop. `dataset auto --preview` spawns it automatically per placement.

Robustness notes (the scan is writing while we read):
- a .npy mid-write fails to load -> logged at DEBUG, retried next tick;
- dark-frame relabeling renames files under us -> the directory is
  re-scanned every tick, so renames just show up as a "new" file;
- closing the window stops the watcher, never the scan.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def find_latest_frame(run_dir: Path) -> Optional[Path]:
    """Newest .npy under the run dir (frames, backgrounds, darks alike)."""

    newest = None
    newest_mtime = -1.0

    for path in run_dir.rglob("*.npy"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            # Renamed/removed between rglob and stat (dark relabeling).
            continue
        if mtime > newest_mtime:
            newest, newest_mtime = path, mtime

    return newest


def newest_run_dir(dataset_root: Path) -> Optional[Path]:
    """Most recently modified auto_scan-* run directory under the root."""

    candidates = [
        p for p in dataset_root.glob("auto_scan-*") if p.is_dir()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


class ScanPreviewWindow:
    """One imshow panel, updated in place; Agg-backed when display=False."""

    def __init__(self, display: bool = True):
        import matplotlib

        if not display:
            matplotlib.use("Agg", force=True)

        import matplotlib.pyplot as plt

        self._plt = plt
        self.display = display

        if display:
            plt.ion()

        self.figure, self.ax = plt.subplots(figsize=(7.5, 6))
        manager = getattr(self.figure.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title("Auto-scan live preview")
        self._image = None

    @property
    def closed(self) -> bool:
        return not self._plt.fignum_exists(self.figure.number)

    def show_frame(self, path: Path, run_dir: Path) -> bool:
        """Display one saved frame; False if it could not be read."""

        # Stat ONCE, inside the guard: the scan renames no-signal frames
        # to '-dark' after each raster, so the path can vanish between
        # any two filesystem calls (hardware-observed 2026-07-28 — a
        # second bare stat() for the title crashed the viewer mid-scan).
        try:
            mtime = path.stat().st_mtime
            frame = np.load(path)
        except Exception as ex:  # noqa: BLE001 - mid-write/rename races
            logger.debug(
                f"Frame {path.name} not readable yet ({ex}); will retry "
                "on the next tick."
            )
            return False

        if frame.ndim != 2:
            logger.warning(
                f"Frame {path.name} has shape {frame.shape}; the preview "
                "only renders 2-D frames — skipping it."
            )
            return True  # counted as shown: no point retrying

        lit = frame[frame > 0]
        vmax = float(np.percentile(lit, 99.5)) if lit.size else 1.0
        vmax = max(vmax, 1.0)

        if self._image is None or self._image.get_array().shape != frame.shape:
            self.ax.clear()
            self._image = self.ax.imshow(
                frame, cmap="inferno", vmin=0.0, vmax=vmax
            )
            self.ax.set_xlabel("sensor px")
            self.ax.set_ylabel("sensor px")
        else:
            self._image.set_data(frame)
            self._image.set_clim(0.0, vmax)

        try:
            label = path.relative_to(run_dir)
        except ValueError:
            label = path.name

        self.ax.set_title(
            f"{label}\npeak {frame.max():g}   "
            f"{time.strftime('%H:%M:%S', time.localtime(mtime))}"
        )
        self.figure.canvas.draw_idle()
        return True

    def pump(self, seconds: float) -> None:
        if self.display:
            self._plt.pause(seconds)
        else:
            time.sleep(seconds)

    def close(self) -> None:
        self._plt.close(self.figure)


def watch(
    run_dir: Path,
    interval_s: float = 1.0,
    display: bool = True,
    max_ticks: Optional[int] = None,
) -> int:
    """
    Watch loop: show the newest frame whenever it changes. Returns the
    number of frames displayed. Stops when the window is closed (or
    after max_ticks, for tests).
    """

    window = ScanPreviewWindow(display=display)
    shown: Optional[Path] = None
    shown_mtime = -1.0
    displayed = 0
    ticks = 0

    logger.info(f"Watching {run_dir} (close the window to stop).")

    try:
        while max_ticks is None or ticks < max_ticks:
            ticks += 1

            latest = find_latest_frame(run_dir)
            if latest is not None:
                try:
                    mtime = latest.stat().st_mtime
                except OSError:
                    mtime = -1.0
                if latest != shown or mtime > shown_mtime:
                    if window.show_frame(latest, run_dir):
                        shown, shown_mtime = latest, mtime
                        displayed += 1

            window.pump(interval_s)

            if display and window.closed:
                break
    finally:
        if not display:
            window.close()

    return displayed
