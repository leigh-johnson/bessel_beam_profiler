"""
Interactive dataset mode for a manually operated translation stage.

Workflow (no CNC gantry required):

    1. You are prompted (by the CLI) for the camera sensor's current
       z-position, e.g. 6.5 cm in front of axicon #1.
    2. A matplotlib window shows a continuously refreshing live preview of
       the beam.
    3. Press SPACE (or 's') to save the currently displayed frame, then move
       the stage by hand, wait for the image to settle, and save again.
       Repeat as many times as you like.
    4. Press 'q' (or ESC / ENTER, or just close the window) to finish.

Every saved frame goes through FLIRDatasetWriter.save_frame_array, so the
run directory and frames.jsonl manifest have exactly the same schema as
gantry scans — the only differences are ScanKind="ManualStage" and a
MoveIndex counter in the Extra metadata (per your note, no coordinates are
tracked between moves).

After the session, the frames can be stitched into a composite image with
stitcher.stitch_run_dir (the CLI does this automatically unless --no-stitch
is passed).

This module deliberately avoids importing PySpin so that it can be unit
tested against a fake writer/camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional
import logging

import numpy as np

from coordinates import ScanPoint, Vec3D

if TYPE_CHECKING:
    from dataset_writer import FLIRDatasetWriter, FrameRecord

logger = logging.getLogger(__name__)


SAVE_KEYS = (" ", "space", "s")
QUIT_KEYS = ("q", "escape", "enter")


@dataclass
class ManualStageConfig:
    # Where the sensor plane sits along the beamline, as entered at the
    # prompt. Stored in TablePosition_mm.z_mm and in the manifest metadata.
    SensorZ_mm: float

    # Free-text description of what SensorZ is measured from,
    # e.g. "front face of axicon #1".
    SensorZReference: str = ""

    PlacementID: str = "manual-stage"

    # Seconds between live preview refreshes.
    PreviewInterval_s: float = 0.05

    # Timeout for each preview frame grab, in milliseconds.
    AcquisitionTimeout_ms: int = 2000

    # Rendering of the live preview.
    Colormap: str = "inferno"
    NormalizationPercentiles: tuple[float, float] = (0.5, 99.9)

    # Extra metadata recorded on every saved frame.
    Metadata: dict[str, Any] = field(default_factory=dict)


class ManualStageSession:
    """
    Live-preview save/move/save loop for a hand-cranked translation stage.

    Drives an already-prepared FLIRDatasetWriter: the writer owns the camera
    and the run directory; this class owns the matplotlib UI loop.
    """

    def __init__(self, writer: "FLIRDatasetWriter", config: ManualStageConfig):
        self.writer = writer
        self.config = config

        self.move_index = 0
        self.saved_records: list["FrameRecord"] = []

        self._done = False
        self._last_frame: Optional[np.ndarray] = None
        self._status_message = ""

    # ------------------------------------------------------------------
    # Camera interaction
    # ------------------------------------------------------------------

    def grab_frame(self) -> Optional[np.ndarray]:
        """
        Software-trigger one frame and return it as a detached NumPy array.
        Returns None for incomplete frames (preview just skips them).
        """

        self.writer._execute_software_trigger()

        image_result = self.writer.cam.GetNextImage(
            self.config.AcquisitionTimeout_ms
        )

        try:
            if image_result.IsIncomplete():
                return None
            return np.array(image_result.GetNDArray(), copy=True)

        finally:
            image_result.Release()
            # A PySpin ImagePtr keeps the camera referenced even after
            # Release(); if this local survives in the traceback of a
            # propagating exception, the camera cannot be released at
            # cleanup (Spinnaker error -1004).
            image_result = None

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _current_scan_point(self) -> ScanPoint:
        return ScanPoint(
            PlacementID=self.config.PlacementID,
            # No stage coordinates are tracked in manual mode.
            GantryPosition_mm=Vec3D(0.0, 0.0, 0.0),
            # Coordinate convention: Y = distance along the beam from the
            # reference optic (X horizontal transverse, Z vertical). The
            # SensorZ_mm field name is legacy; its value is the beam-path
            # distance and therefore lives in TablePosition y.
            TablePosition_mm=Vec3D(0.0, self.config.SensorZ_mm, 0.0),
            NShots=1,
            Metadata={
                "ScanKind": "ManualStage",
                "MoveIndex": self.move_index,
                "SensorZ_mm": self.config.SensorZ_mm,
                "SensorZReference": self.config.SensorZReference,
                **self.config.Metadata,
            },
        )

    def save_current_frame(self) -> Optional["FrameRecord"]:
        if self._last_frame is None:
            self._status_message = "No frame to save yet."
            return None

        record = self.writer.save_frame_array(
            self._last_frame,
            self._current_scan_point(),
            # ShotIndex doubles as the move counter so filenames stay unique.
            shot_idx=self.move_index,
        )

        self.saved_records.append(record)
        self.move_index += 1

        self._status_message = (
            f"Saved frame {self.move_index} -> {record.Path}\n"
            "Move the stage, then press SPACE to save the next one."
        )

        return record

    # ------------------------------------------------------------------
    # UI loop
    # ------------------------------------------------------------------

    def _on_key(self, event) -> None:
        if event.key in SAVE_KEYS:
            self.save_current_frame()
        elif event.key in QUIT_KEYS:
            self._done = True

    def _preview_limits(self, arr: np.ndarray) -> tuple[float, float]:
        lo, hi = np.percentile(arr, self.config.NormalizationPercentiles)

        if hi <= lo:
            hi = lo + 1.0

        return float(lo), float(hi)

    def _saturation_suffix(self, arr: np.ndarray) -> str:
        if not np.issubdtype(arr.dtype, np.integer):
            return ""

        dtype_max = np.iinfo(arr.dtype).max
        saturated = int(np.sum(arr == dtype_max))

        if saturated == 0:
            return ""

        return f"  |  WARNING: {saturated} saturated px"

    def run(self) -> list["FrameRecord"]:
        """
        Run the interactive preview loop until the user quits.

        Returns the FrameRecords of every saved frame, in order.
        """

        import matplotlib.pyplot as plt

        self.writer._begin_acquisition()

        fig = None

        try:
            fig, ax = plt.subplots(figsize=(9, 7))

            try:
                fig.canvas.manager.set_window_title("Manual stage dataset mode")
            except AttributeError as ex:
                # headless / unusual backends have no window manager
                logger.warning(f"Could not set preview window title: {ex}")

            fig.canvas.mpl_connect("key_press_event", self._on_key)

            # Grab one frame up front to size the axes image.
            first = None

            while first is None:
                first = self.grab_frame()

            self._last_frame = first

            lo, hi = self._preview_limits(first)
            image_artist = ax.imshow(
                first,
                cmap=self.config.Colormap,
                vmin=lo,
                vmax=hi,
                interpolation="nearest",
            )
            fig.colorbar(image_artist, ax=ax, fraction=0.046)

            status_text = fig.text(
                0.02,
                0.02,
                "",
                fontsize=9,
                family="monospace",
                va="bottom",
            )

            plt.show(block=False)

            while not self._done and plt.fignum_exists(fig.number):
                frame = self.grab_frame()

                if frame is not None:
                    self._last_frame = frame

                    lo, hi = self._preview_limits(frame)
                    image_artist.set_data(frame)
                    image_artist.set_clim(lo, hi)

                    ax.set_title(
                        f"z = {self.config.SensorZ_mm / 10.0:g} cm "
                        f"({self.config.SensorZReference})  |  "
                        f"saved: {self.move_index}  |  "
                        f"max: {frame.max()}"
                        f"{self._saturation_suffix(frame)}"
                    )

                status_text.set_text(
                    "SPACE: save frame    q / ESC / close window: finish\n"
                    + self._status_message
                )

                fig.canvas.draw_idle()

                # plt.pause runs the GUI event loop, so key presses are
                # delivered here.
                plt.pause(self.config.PreviewInterval_s)

        finally:
            self.writer._end_acquisition()

            if fig is not None and plt.fignum_exists(fig.number):
                plt.close(fig)

        return self.saved_records