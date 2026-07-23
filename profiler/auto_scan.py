"""
Automated beam-stack acquisition on the FluidNC gantry.

COORDINATE CONVENTION — one frame, used everywhere:

    X = horizontal transverse (perpendicular to the beam)
    Y = beam propagation direction down the table
    Z = vertical (perpendicular to the optics table)

The gantry's machine axes coincide with this frame after homing, so
ScanPoint.GantryPosition_mm IS the machine coordinate. TablePosition_mm
differs only in Y, where the per-placement measurement anchors machine Y
to the beamline:

    TableY = MeasuredSensorY + BeamDirectionSign * (machineY - YStart)

(X and Z table coordinates are left equal to machine coordinates; only the
along-beam position is measured against the optic.)

A cross-section of the beam is therefore an X-Z raster at fixed machine Y
(an "XZ slice"), and the scan steps down the beam along Y:

    1. You manually measure the optic -> camera-sensor distance with the
       camera at machine Y = YStart, and enter it when prompted.
    2. At each Y step: exposure is auto-calibrated (headless, seeded from
       the previous slice), an off-axis ambient background is captured
       when needed (see BackgroundMode), then the X-Z raster runs
       (adaptive by default).
    3. Frames land in per-slice subfolders named by the distance from the
       optic, e.g. run_dir/y0100.00cm/, each with its own frames.jsonl,
       calibration, raster metadata, and background reference.

BeamDirectionSign was verified on hardware 2026-07-22 (preflight
beam-direction check): on this rig machine +Y moves the camera TOWARD the
optic, so the default is -1. Re-verify after any gantry re-orientation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dataclass_replace
from pathlib import Path
from typing import Any, Callable, Optional
import json
import logging
import time

import numpy as np

from adaptive_raster import AdaptiveRasterConfig, AdaptiveRasterRunner
from coordinates import AxisRange, ScanPoint, Vec3D
from headless_calibration import (
    HeadlessCalibrationConfig,
    HeadlessCalibrationResult,
    calibrate_exposure_headless,
)

logger = logging.getLogger(__name__)

BACKGROUND_SUBFOLDER = "background"

# Filename suffix (before the extension) for the adaptive raster's
# no-signal frames — the proof-of-darkness perimeter. They stay in the
# dataset (they document why growth stopped) but are visibly labeled and
# excluded from composites by default.
DARK_FRAME_SUFFIX = "-dark"


def default_background_ladder_us(
    min_us: float = 25.0,
    max_us: float = 100_000.0,
    count: int = 10,
) -> tuple[float, ...]:
    """Log-spaced exposure ladder covering the calibration range."""
    return tuple(float(v) for v in np.geomspace(min_us, max_us, count))


@dataclass(frozen=True)
class AutoScanConfig:
    PlacementID: str

    # Manually measured optic -> camera-sensor distance along the beam (mm)
    # with the camera at machine Y = YStart_machine_mm, plus which optic it
    # was measured from.
    MeasuredSensorY_mm: float
    MeasuredFrom: str = "axicon3"

    # Beam-propagation stack: machine Y positions stepped along the beam.
    YStart_machine_mm: float = 10.0
    YStop_machine_mm: float = 150.0
    YStep_mm: float = 10.0

    # +1 if machine +Y points downstream (away from the optic), -1 if
    # machine +Y points toward it. Default -1: VERIFIED on hardware
    # 2026-07-22 (preflight beam-direction check) — machine +Y moves the
    # camera TOWARD the optic on this rig.
    BeamDirectionSign: int = -1

    # X-Z cross-section raster (machine coordinates) at every Y. In
    # adaptive raster mode these are the CAPS (maximum extent), not the
    # fixed grid. X = horizontal transverse, Z = vertical.
    X: AxisRange = AxisRange(start_mm=45.0, stop_mm=75.0, step_mm=5.0)
    Z: AxisRange = AxisRange(start_mm=-75.0, stop_mm=-45.0, step_mm=4.0)

    # "adaptive": start from one seed frame at the calibration point and
    #   grow the rectangle only while its edge frames still contain signal
    #   (background-referenced). If the beam fits in one camera frame, the
    #   slice completes after a single position.
    # "fixed": raster the full X x Z grid at every slice.
    RasterMode: str = "adaptive"

    # Adaptive-raster signal test: a border strip "has signal" when at
    # least MinSignalPixels exceed (background p99 + SignalMargin_counts).
    SignalMargin_counts: float = 8.0
    MinSignalPixels: int = 50
    BorderStripFraction: float = 0.15

    # "any" = orientation-independent border test (default, safe;
    # overshoots the beam extent by <= ~1 frame per side). "directional"
    # tests only the outward-facing strip and needs the image->machine axis
    # mapping below verified once on hardware. In the raster lattice,
    # lattice-x = machine X and lattice-y = machine Z.
    BorderTest: str = "any"
    ImageTranspose: bool = False
    ImageFlipX: bool = False
    ImageFlipY: bool = False

    # Signal threshold fallback when no per-slice background exists
    # (background mode "ladder" or "none").
    FallbackBackgroundP99_counts: float = 5.0

    # Where the camera sits during exposure calibration (beam core), in
    # machine X/Z. None = center of the X/Z raster caps.
    CalibrationX_mm: Optional[float] = None
    CalibrationZ_mm: Optional[float] = None

    # Follow the beam along the stack: after each slice, move the
    # calibration point (and adaptive-raster seed) to the brightest
    # measured cell of that slice. Essential for ring beams whose radius
    # changes with Y — a fixed point can drift into the dark interior,
    # breaking both exposure calibration and the raster seed.
    FollowBeam: bool = True

    # Find the beam automatically when no explicit calibration point is
    # given (and re-find after a slice reports BeamFound=false): sweep a
    # column of frames along Z at the calibration X, starting from the
    # FAR end of the Z caps (Z.start_mm — the extremum away from the Z
    # home switch, where the beam sits on this rig), looking for
    # CONTRAST (frame max - median >= FindBeamContrast_counts). Contrast
    # rejects what raw brightness cannot: exposure calibration on a dark
    # spot happily "converges" by amplifying flat ambient light, but
    # ambient has no contrast at any exposure. If a whole sweep is flat,
    # exposure is scaled x8 and the sweep repeats (FindBeamExposureScalings
    # times) before giving up with a warning.
    FindBeam: bool = True
    FindBeamStepZ_mm: float = 5.0  # ~sensor height: gap-free column
    FindBeamContrast_counts: float = 30.0
    FindBeamStartExposure_us: float = 10_000.0
    FindBeamExposureScalings: int = 2

    NShots: int = 1

    # How beam-off/ambient backgrounds are captured:
    #
    #   "offaxis" (default): at each Y, right after exposure calibration,
    #       the camera moves to an X/Z position outside the beam and
    #       captures backgrounds at that slice's calibrated exposure —
    #       exact exposure match, drift tracking, no manual beam blocking.
    #   "ladder": once per placement, you block the beam when prompted and
    #       a log-spaced exposure ladder is captured.
    #   "none": no backgrounds (quick alignment runs).
    BackgroundMode: str = "offaxis"

    # Off-axis background position (machine X/Z). None = automatically use
    # the machine-limit X/Z corner farthest from the calibration point.
    BackgroundX_mm: Optional[float] = None
    BackgroundZ_mm: Optional[float] = None

    # Off-axis mode: recapture the background only when the calibrated
    # exposure has changed by at least this fraction since the background
    # was last CAPTURED (cumulative, so slow drift still triggers a
    # refresh). Slices that reuse an earlier background record it in their
    # background_reference.json. 0.0 = capture at every slice.
    BackgroundExposureChangeFraction: float = 0.10

    # Ladder mode only: exposures for the beam-blocked ladder.
    BackgroundExposures_us: tuple[float, ...] = ()

    BackgroundShots: int = 3

    Metadata: dict[str, Any] = field(default_factory=dict)

    def y_values_machine_mm(self) -> list[float]:
        return AxisRange(
            start_mm=self.YStart_machine_mm,
            stop_mm=self.YStop_machine_mm,
            step_mm=self.YStep_mm,
        ).values()

    def calibration_xz(self) -> tuple[float, float]:
        x = (
            self.CalibrationX_mm
            if self.CalibrationX_mm is not None
            else (self.X.start_mm + self.X.stop_mm) / 2.0
        )
        z = (
            self.CalibrationZ_mm
            if self.CalibrationZ_mm is not None
            else (self.Z.start_mm + self.Z.stop_mm) / 2.0
        )
        return x, z

    def beam_y_mm(self, machine_y_mm: float) -> float:
        """Distance from the reference optic along the beam (table Y)."""

        return self.MeasuredSensorY_mm + self.BeamDirectionSign * (
            machine_y_mm - self.YStart_machine_mm
        )

    def placement_notes(self) -> str:
        return (
            f"MeasuredSensorY={self.MeasuredSensorY_mm:g}mm after "
            f"{self.MeasuredFrom} at machine Y={self.YStart_machine_mm:g}mm; "
            f"BeamDirectionSign={self.BeamDirectionSign:+d} (machine +Y "
            f"{'away from' if self.BeamDirectionSign > 0 else 'toward'} "
            "the optic)"
        )


def y_subfolder_name(beam_y_mm: float) -> str:
    """
    Zero-padded, lexically sortable slice folder name from the distance
    along the beam, e.g. 'y0100.00cm'.
    """
    return f"y{beam_y_mm / 10.0:07.2f}cm"


# ---------------------------------------------------------------------------
# Camera helpers (software-trigger grab + exposure set on the writer's cam)
# ---------------------------------------------------------------------------


def _grab_frame(writer, timeout_ms: int) -> Optional[np.ndarray]:
    writer._execute_software_trigger()

    # Long exposures (dim slices) can exceed the base acquisition timeout;
    # extend it by the current exposure time so the grab cannot time out
    # while the sensor is legitimately still integrating.
    try:
        timeout_ms = int(timeout_ms + writer.cam.ExposureTime.GetValue() / 1000.0)
    except Exception as ex:  # noqa: BLE001 - fall back to the base timeout
        logger.warning(
            f"Could not read ExposureTime to extend the grab timeout ({ex}); "
            f"using the base timeout of {timeout_ms} ms. Long-exposure "
            "frames may spuriously time out."
        )

    try:
        image_result = writer.cam.GetNextImage(timeout_ms)
    except Exception as ex:  # noqa: BLE001 - narrow to Spinnaker timeouts
        # A software trigger fired while the camera was still arming can be
        # silently dropped (GenTL -1011 timeout). Treat it like an
        # incomplete frame: the calibration loop re-triggers on the next
        # grab_frame() call and gives up cleanly after its retry limit.
        if "Spinnaker" in type(ex).__name__ or "-1011" in str(ex):
            logger.warning(
                f"Calibration frame grab timed out ({ex}); re-triggering. "
                "If this repeats, check that SpinView is fully closed."
            )
            return None
        raise

    try:
        if image_result.IsIncomplete():
            return None
        return np.array(image_result.GetNDArray(), copy=True)
    finally:
        image_result.Release()
        # Drop the PySpin ImagePtr local so a propagating exception cannot
        # pin the camera reference (Spinnaker error -1004).
        image_result = None


def set_exposure_us(cam, exposure_us: float) -> float:
    """Clamp to the camera's own exposure limits, apply, return the value set."""

    lo = float(cam.ExposureTime.GetMin())
    hi = float(cam.ExposureTime.GetMax())
    value = float(max(lo, min(hi, exposure_us)))
    cam.ExposureTime.SetValue(value)
    return value


# ---------------------------------------------------------------------------
# Auto-scan session
# ---------------------------------------------------------------------------


class AutoScanSession:
    """
    Drives one placement's beam stack (XZ slices stepped along Y) through
    an already-prepared FLIRDatasetWriter whose stage_controller is a
    FluidNCStageController.
    """

    def __init__(
        self,
        writer,  # FLIRDatasetWriter
        config: AutoScanConfig,
        calibration_config: HeadlessCalibrationConfig = HeadlessCalibrationConfig(),
        pause_fn: Callable[[str], None] = None,
    ):
        self.writer = writer
        self.config = config
        self.calibration_config = calibration_config
        self.pause_fn = pause_fn or (lambda message: input(f"{message}\nPress ENTER... "))

        self.records = []
        self.calibrations: dict[str, HeadlessCalibrationResult] = {}

        # Off-axis background reuse state: set whenever a background is
        # actually captured, consulted before capturing the next one.
        self._last_background: Optional[dict[str, Any]] = None

        # Where the next slice calibrates and seeds its raster. Starts at
        # the configured point; with FollowBeam it tracks the brightest
        # measured cell slice-to-slice.
        self._calibration_xz: tuple[float, float] = config.calibration_xz()

        # Run the find-beam sweep before the first slice when the user did
        # not point us at the beam explicitly; re-armed whenever a slice
        # comes up empty.
        self._need_find_beam: bool = config.FindBeam and (
            config.CalibrationX_mm is None or config.CalibrationZ_mm is None
        )

    # -- geometry -------------------------------------------------------

    def table_position(self, x_mm: float, machine_y_mm: float, z_mm: float) -> Vec3D:
        """
        Table coordinates: identical to machine coordinates except Y, which
        is anchored to the distance from the reference optic.
        """

        return Vec3D(
            x_mm=x_mm,
            y_mm=self.config.beam_y_mm(machine_y_mm),
            z_mm=z_mm,
        )

    def _make_point(
        self,
        x_mm: float,
        machine_y_mm: float,
        z_mm: float,
        nshots: int = 1,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ScanPoint:
        return ScanPoint(
            PlacementID=self.config.PlacementID,
            GantryPosition_mm=Vec3D(x_mm=x_mm, y_mm=machine_y_mm, z_mm=z_mm),
            TablePosition_mm=self.table_position(x_mm, machine_y_mm, z_mm),
            NShots=nshots,
            Metadata=metadata or {},
        )

    def _move_to(self, x_mm: float, machine_y_mm: float, z_mm: float) -> None:
        self.writer._move_and_wait(self._make_point(x_mm, machine_y_mm, z_mm))

    # -- backgrounds ----------------------------------------------------

    def background_xz(self, machine_limits) -> tuple[float, float]:
        """
        Off-axis background position in the X-Z cross-section plane:
        explicit config values if given, otherwise the machine-limit X/Z
        corner farthest from the calibration point (as far from the beam
        as this placement allows).
        """

        if (
            self.config.BackgroundX_mm is not None
            and self.config.BackgroundZ_mm is not None
        ):
            return self.config.BackgroundX_mm, self.config.BackgroundZ_mm

        calib_x, calib_z = self.config.calibration_xz()

        corners = [
            (machine_limits.x_min_mm, machine_limits.z_min_mm),
            (machine_limits.x_min_mm, machine_limits.z_max_mm),
            (machine_limits.x_max_mm, machine_limits.z_min_mm),
            (machine_limits.x_max_mm, machine_limits.z_max_mm),
        ]

        return max(
            corners,
            key=lambda c: (c[0] - calib_x) ** 2 + (c[1] - calib_z) ** 2,
        )

    def capture_background_offaxis(
        self,
        machine_y_mm: float,
        exposure_us: float,
        machine_limits,
        y_name: str,
    ) -> list:
        """
        Move the camera out of the beam (in X/Z) and capture ambient
        backgrounds at this slice's already-applied calibrated exposure.
        Frames land in the slice's own subfolder, tagged as backgrounds.
        """

        bg_x, bg_z = self.background_xz(machine_limits)
        self._move_to(bg_x, machine_y_mm, bg_z)

        point = self._make_point(
            bg_x,
            machine_y_mm,
            bg_z,
            nshots=self.config.BackgroundShots,
            metadata={
                "ScanKind": "Background",
                "BackgroundMode": "OffAxisAmbient",
                "Subfolder": y_name,
                "FileTag": "background",
                "Exposure_us": exposure_us,
                "MachineY_mm": machine_y_mm,
                "BeamY_mm": self.config.beam_y_mm(machine_y_mm),
                "MeasuredFrom": self.config.MeasuredFrom,
                **self.config.Metadata,
            },
        )

        records = self.writer.acquire_at_current_position(point)
        self.records.extend(records)
        return records

    def background_for_slice(
        self,
        machine_y_mm: float,
        exposure_us: float,
        machine_limits,
        y_name: str,
    ) -> list:
        """
        Off-axis background with change-based cadence: capture a fresh
        background only when the calibrated exposure moved by at least
        BackgroundExposureChangeFraction since the last captured one;
        otherwise reuse it. Either way, background_reference.json in the
        slice folder records which background frames apply to this slice.
        """

        last = self._last_background
        change = None

        if last is not None and last["Exposure_us"] > 0:
            change = (
                abs(exposure_us - last["Exposure_us"]) / last["Exposure_us"]
            )

        must_capture = (
            change is None
            or change >= self.config.BackgroundExposureChangeFraction
        )

        if must_capture:
            records = self.capture_background_offaxis(
                machine_y_mm, exposure_us, machine_limits, y_name
            )

            self._last_background = {
                "Exposure_us": exposure_us,
                "Records": records,
                "SliceName": y_name,
                "MachineY_mm": machine_y_mm,
            }

            bg_x, bg_z = self.background_xz(machine_limits)
            reason = (
                "first slice"
                if change is None
                else f"exposure changed {change * 100.0:.1f}%"
            )
            logger.info(
                f"{len(records)} off-axis background frame(s) at "
                f"X{bg_x:g} Z{bg_z:g} ({reason})"
            )
        else:
            records = last["Records"]
            logger.info(
                f"reusing background from {last['SliceName']}/ "
                f"(exposure changed {change * 100.0:.1f}% < "
                f"{self.config.BackgroundExposureChangeFraction * 100.0:g}%)"
            )

        self._write_background_reference(
            y_name, exposure_us, change, captured=must_capture
        )

        return records

    def _write_background_reference(
        self,
        y_name: str,
        exposure_us: float,
        change: Optional[float],
        captured: bool,
    ) -> None:
        last = self._last_background

        slice_dir = self.writer.run_dir / y_name
        slice_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "Reused": not captured,
            "SliceExposure_us": exposure_us,
            "BackgroundExposure_us": last["Exposure_us"],
            "BackgroundSlice": last["SliceName"],
            "BackgroundMachineY_mm": last["MachineY_mm"],
            "ExposureChangeFraction": change,
            "MaxExposureChangeFraction": (
                self.config.BackgroundExposureChangeFraction
            ),
            "BackgroundPaths": [
                getattr(record, "Path", None) for record in last["Records"]
            ],
        }

        (slice_dir / "background_reference.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )

    # -- ladder mode: beam-blocked backgrounds, once per placement -----

    def capture_background_ladder(self) -> list:
        exposures = self.config.BackgroundExposures_us or default_background_ladder_us()

        calib_x, calib_z = self.config.calibration_xz()
        self._move_to(calib_x, self.config.YStart_machine_mm, calib_z)

        self.pause_fn(
            "BLOCK THE BEAM now (backgrounds for this placement are about to "
            f"be captured at {len(exposures)} exposures)."
        )

        records = []

        for rung_idx, exposure_us in enumerate(exposures):
            actual_us = set_exposure_us(self.writer.cam, exposure_us)

            point = self._make_point(
                calib_x,
                self.config.YStart_machine_mm,
                calib_z,
                nshots=self.config.BackgroundShots,
                metadata={
                    "ScanKind": "Background",
                    "Subfolder": BACKGROUND_SUBFOLDER,
                    "FileTag": f"exp{actual_us:09.1f}us",
                    "Exposure_us": actual_us,
                    "LadderIndex": rung_idx,
                    "MeasuredFrom": self.config.MeasuredFrom,
                    **self.config.Metadata,
                },
            )

            records.extend(self.writer.acquire_at_current_position(point))
            logger.info(
                f"background ladder {rung_idx + 1}/{len(exposures)}: "
                f"{actual_us:.1f} us x {self.config.BackgroundShots} shots"
            )

        self.pause_fn("UNBLOCK THE BEAM now (backgrounds done).")

        self.records.extend(records)
        return records

    # -- beam finding ---------------------------------------------------

    def find_beam(self, machine_y_mm: float) -> bool:
        """
        Sweep a column of frames along Z at the calibration X, far end
        first, hunting for structured light (contrast = max - median).
        On a hit, the calibration point / raster seed moves there.
        """

        x = self._calibration_xz[0]

        z_values: list[float] = []
        z = self.config.Z.start_mm  # far-from-switch end (Z homes at top)
        while z <= self.config.Z.stop_mm + 1e-9:
            z_values.append(round(z, 3))
            z += self.config.FindBeamStepZ_mm

        exposure_us = float(self.config.FindBeamStartExposure_us)
        max_exposure_us = self.calibration_config.Base.MaxExposure_us
        timeout_ms = self.writer.config.AcquisitionTimeout_ms

        for attempt in range(1 + self.config.FindBeamExposureScalings):
            actual_us = set_exposure_us(
                self.writer.cam, min(exposure_us, max_exposure_us)
            )
            logger.info(
                f"find-beam: sweeping Z {z_values[0]:g}..{z_values[-1]:g} at "
                f"X{x:g}, exposure {actual_us:g} us "
                f"(attempt {attempt + 1}/{1 + self.config.FindBeamExposureScalings})"
            )

            self.writer._begin_acquisition()

            if self.writer.config.TriggerArmDelay_s > 0:
                time.sleep(self.writer.config.TriggerArmDelay_s)

            try:
                for z_mm in z_values:
                    self._move_to(x, machine_y_mm, z_mm)
                    arr = _grab_frame(self.writer, timeout_ms)

                    if arr is None:
                        continue

                    contrast = float(arr.max()) - float(np.median(arr))

                    if contrast >= self.config.FindBeamContrast_counts:
                        self._calibration_xz = (x, z_mm)
                        logger.info(
                            f"find-beam: structured light at X{x:g} "
                            f"Z{z_mm:g} (contrast {contrast:.0f} counts) — "
                            "calibrating and seeding there."
                        )
                        return True
            finally:
                self.writer._end_acquisition()

            exposure_us *= 8.0

        logger.warning(
            f"find-beam: no structured light anywhere along Z at X{x:g} "
            f"after {1 + self.config.FindBeamExposureScalings} exposure "
            "attempts. Keeping the seed at "
            f"X{self._calibration_xz[0]:g} Z{self._calibration_xz[1]:g} — "
            "check the beam is on and the X position crosses it."
        )
        return False

    # -- per-slice calibration -----------------------------------------

    def calibrate_at(
        self,
        machine_y_mm: float,
        start_exposure_us: float,
    ) -> HeadlessCalibrationResult:
        calib_x, calib_z = self._calibration_xz
        self._move_to(calib_x, machine_y_mm, calib_z)

        timeout_ms = self.writer.config.AcquisitionTimeout_ms

        # Cap calibration exposures at the configured maximum (default
        # 1 s), NOT the camera's own limit (30 s on the BFS-PGE-31S4M): a
        # dark calibration point must fail fast with Converged=False, not
        # ramp into half-minute frames.
        max_exposure_us = self.calibration_config.Base.MaxExposure_us

        self.writer._begin_acquisition()

        # Let the camera finish arming before the first software trigger
        # (a trigger fired too early is silently dropped -> -1011 timeout).
        if self.writer.config.TriggerArmDelay_s > 0:
            time.sleep(self.writer.config.TriggerArmDelay_s)

        try:
            result = calibrate_exposure_headless(
                grab_frame=lambda: _grab_frame(self.writer, timeout_ms),
                set_exposure_us=lambda us: set_exposure_us(
                    self.writer.cam, min(us, max_exposure_us)
                ),
                start_exposure_us=start_exposure_us,
                config=self.calibration_config,
            )
        finally:
            self.writer._end_acquisition()

        self._save_calibration(machine_y_mm, result)
        return result

    def _save_calibration(
        self, machine_y_mm: float, result: HeadlessCalibrationResult
    ) -> None:
        beam_y_mm = self.config.beam_y_mm(machine_y_mm)
        slice_dir = self.writer.run_dir / y_subfolder_name(beam_y_mm)
        slice_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "MachineY_mm": machine_y_mm,
            "BeamY_mm": beam_y_mm,
            "MeasuredFrom": self.config.MeasuredFrom,
            "CalibrationX_mm": self._calibration_xz[0],
            "CalibrationZ_mm": self._calibration_xz[1],
            "FinalExposure_us": result.FinalExposure_us,
            "LastMax": result.LastMax,
            "LastSaturatedPixels": result.LastSaturatedPixels,
            "Iterations": result.Iterations,
            "Converged": result.Converged,
            "Note": result.Note,
        }

        (slice_dir / "calibration_result.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )

        # Full camera-settings JSON, same schema as the calibrations/ dir,
        # when the writer holds a real FLIRCameraSettings dataclass.
        settings = self.writer.camera_settings
        if hasattr(settings, "__dataclass_fields__"):
            calibrated = dataclass_replace(
                settings, ExposureTime=result.FinalExposure_us
            )
            calibrated.to_json_file(slice_dir / "calibrated_camera_settings.json")

        self.calibrations[y_subfolder_name(beam_y_mm)] = result

    # -- the beam stack -------------------------------------------------

    def run_y_stack(self, machine_limits) -> list:
        """
        For each machine Y: headless exposure calibration, background (if
        offaxis mode), then the X-Z cross-section raster. machine_limits is
        the Bounds3D the stage controller enforces (pass
        FluidNCStageController.config.MachineLimits_mm).
        """

        y_values = self.config.y_values_machine_mm()

        logger.info(
            f"Calibration/seed point: X{self._calibration_xz[0]:g} "
            f"Z{self._calibration_xz[1]:g} "
            f"(follow-beam={'on' if self.config.FollowBeam else 'off'}); "
            f"X caps {self.config.X.start_mm:g}..{self.config.X.stop_mm:g}, "
            f"Z caps {self.config.Z.start_mm:g}..{self.config.Z.stop_mm:g}"
        )

        exposure_seed_us = getattr(
            self.writer.camera_settings, "ExposureTime", None
        ) or 1000.0

        for y_idx, machine_y in enumerate(y_values):
            beam_y_mm = self.config.beam_y_mm(machine_y)
            y_name = y_subfolder_name(beam_y_mm)

            logger.info(
                f"[{y_idx + 1}/{len(y_values)}] machine Y {machine_y:g} mm "
                f"(beam y = {beam_y_mm / 10.0:g} cm after "
                f"{self.config.MeasuredFrom}) -> {y_name}/"
            )

            if self._need_find_beam:
                if self.find_beam(machine_y):
                    self._need_find_beam = False

            calibration = self.calibrate_at(machine_y, exposure_seed_us)
            exposure_seed_us = calibration.FinalExposure_us

            logger.info(
                f"exposure {calibration.FinalExposure_us:.1f} us "
                f"(max {calibration.LastMax}, "
                f"saturated {calibration.LastSaturatedPixels}, "
                f"converged={calibration.Converged})"
            )

            background_records = []

            if self.config.BackgroundMode == "offaxis":
                background_records = self.background_for_slice(
                    machine_y,
                    calibration.FinalExposure_us,
                    machine_limits,
                    y_name,
                )

            point_metadata = {
                "ScanKind": "AutoBeamStack",
                "Subfolder": y_name,
                "MachineY_mm": machine_y,
                "BeamY_mm": beam_y_mm,
                "MeasuredFrom": self.config.MeasuredFrom,
                "Exposure_us": calibration.FinalExposure_us,
                **self.config.Metadata,
            }

            if self.config.RasterMode == "adaptive":
                records, raster_metadata = self._run_adaptive_raster(
                    machine_y,
                    machine_limits,
                    background_records,
                    point_metadata,
                )

                records, raster_metadata = self._label_dark_frames(
                    y_name, records, raster_metadata
                )

                if raster_metadata.get("BeamFound") is False:
                    # Re-arm the sweep so the next slice hunts again.
                    self._need_find_beam = self.config.FindBeam
                    logger.warning(
                        f"Slice {y_name}: NO beam found around the seed at "
                        f"X{self._calibration_xz[0]:g} "
                        f"Z{self._calibration_xz[1]:g}. "
                        + (
                            "The find-beam sweep will run again next slice."
                            if self.config.FindBeam
                            else "Point --calibration-x/--calibration-z at "
                            "the beam and check the X/Z caps contain it."
                        )
                    )
            else:
                points = self._fixed_grid_points(
                    machine_y, machine_limits, point_metadata
                )
                records = self.writer.acquire_scan(points)

                raster_metadata = {
                    "RasterMode": "fixed",
                    "Step_mm": [self.config.X.step_mm, self.config.Z.step_mm],
                    "GridShape": [
                        len(self.config.X.values()),
                        len(self.config.Z.values()),
                    ],
                    "FinalRect_mm": {
                        "XMin": self.config.X.start_mm,
                        "XMax": self.config.X.stop_mm,
                        "ZMin": self.config.Z.start_mm,
                        "ZMax": self.config.Z.stop_mm,
                    },
                    "CellsCaptured": len(points),
                }

            self.records.extend(records)
            self._write_raster_metadata(y_name, machine_y, raster_metadata)

            logger.info(f"{len(records)} frames -> {y_name}/")

            self._follow_beam(records, background_records)

        return self.records

    def _follow_beam(self, records, background_records) -> None:
        """
        Move the next slice's calibration point / raster seed to this
        slice's brightest measured cell (ring beams shift and change
        radius along Y; a static point can drift into the dark interior).
        """

        if not self.config.FollowBeam or not records:
            return

        threshold, _ = self.signal_threshold(background_records)
        brightest = max(records, key=lambda record: record.Max)

        if brightest.Max <= threshold:
            logger.warning(
                "follow-beam: no frame in this slice exceeded the signal "
                f"threshold ({threshold:g} counts); keeping the calibration "
                f"point at X{self._calibration_xz[0]:g} "
                f"Z{self._calibration_xz[1]:g}."
            )
            return

        new_xz = (
            brightest.GantryPosition_mm.x_mm,
            brightest.GantryPosition_mm.z_mm,
        )

        if new_xz != self._calibration_xz:
            logger.info(
                f"follow-beam: brightest cell (max {brightest.Max}) at "
                f"X{new_xz[0]:g} Z{new_xz[1]:g} — next slice calibrates and "
                "seeds there."
            )

        self._calibration_xz = new_xz

    def _fixed_grid_points(
        self,
        machine_y_mm: float,
        machine_limits,
        point_metadata: dict,
    ) -> list[ScanPoint]:
        points = []

        for z_mm in self.config.Z.values():
            for x_mm in self.config.X.values():
                position = Vec3D(x_mm=x_mm, y_mm=machine_y_mm, z_mm=z_mm)

                if not machine_limits.contains(position):
                    raise ValueError(
                        f"Raster point outside machine limits: {position}"
                    )

                points.append(
                    self._make_point(
                        x_mm,
                        machine_y_mm,
                        z_mm,
                        nshots=self.config.NShots,
                        metadata=point_metadata,
                    )
                )

        return points

    # -- adaptive raster ------------------------------------------------

    def signal_threshold(self, background_records) -> tuple[float, str]:
        """
        Counts threshold for "this frame contains beam signal": the p99 of
        this slice's own off-axis background frames plus a margin, falling
        back to a configured constant when no background is available.
        """

        if background_records:
            arrays = [np.load(record.Path) for record in background_records]
            p99 = float(np.percentile(np.stack(arrays), 99))
            threshold = p99 + self.config.SignalMargin_counts
            source = (
                f"offaxis-background-p99({p99:.2f})"
                f"+margin({self.config.SignalMargin_counts:g})"
            )
            return threshold, source

        threshold = (
            self.config.FallbackBackgroundP99_counts
            + self.config.SignalMargin_counts
        )
        source = (
            f"fallback({self.config.FallbackBackgroundP99_counts:g})"
            f"+margin({self.config.SignalMargin_counts:g}) "
            "— no per-slice background available"
        )
        return threshold, source

    def _run_adaptive_raster(
        self,
        machine_y_mm: float,
        machine_limits,
        background_records,
        point_metadata: dict,
    ) -> tuple[list, dict]:
        threshold, threshold_source = self.signal_threshold(background_records)
        calib_x, calib_z = self._calibration_xz

        # Caps: the configured X/Z ranges, further clamped to machine
        # limits. In the raster lattice, lattice-x = machine X and
        # lattice-y = machine Z (the cross-section plane).
        x_min = max(self.config.X.start_mm, machine_limits.x_min_mm)
        x_max = min(self.config.X.stop_mm, machine_limits.x_max_mm)
        z_min = max(self.config.Z.start_mm, machine_limits.z_min_mm)
        z_max = min(self.config.Z.stop_mm, machine_limits.z_max_mm)

        raster_config = AdaptiveRasterConfig(
            CenterX_mm=calib_x,
            CenterY_mm=calib_z,
            StepX_mm=self.config.X.step_mm,
            StepY_mm=self.config.Z.step_mm,
            XMin_mm=x_min,
            XMax_mm=x_max,
            YMin_mm=z_min,
            YMax_mm=z_max,
            SignalThreshold_counts=threshold,
            MinSignalPixels=self.config.MinSignalPixels,
            BorderStripFraction=self.config.BorderStripFraction,
            BorderTest=self.config.BorderTest,
            ImageTranspose=self.config.ImageTranspose,
            ImageFlipX=self.config.ImageFlipX,
            ImageFlipY=self.config.ImageFlipY,
        )

        def capture(x_mm: float, z_mm: float, i: int, j: int):
            point = self._make_point(
                x_mm,
                machine_y_mm,
                z_mm,
                nshots=self.config.NShots,
                metadata={**point_metadata, "GridI": i, "GridJ": j},
            )

            records = self.writer.acquire_scan([point])
            arr = np.load(records[0].Path)
            return records, arr

        runner = AdaptiveRasterRunner(raster_config, capture)
        result = runner.run()

        metadata = result.Metadata
        metadata["SignalThresholdSource"] = threshold_source
        # The runner's lattice is axis-agnostic; record what its axes mean
        # here so BorderSignal/FinalRect keys are unambiguous.
        metadata["LatticeAxes"] = {"x": "machine X", "y": "machine Z"}

        logger.info(
            f"adaptive raster: {metadata['CellsCaptured']} cells "
            f"(grid {metadata['GridShape'][0]}x{metadata['GridShape'][1]}) "
            f"vs {metadata['FixedGridCells']} for the full fixed grid"
        )

        return result.Records, metadata

    def _label_dark_frames(
        self, y_name: str, records: list, metadata: dict
    ) -> tuple[list, dict]:
        """
        Rename the raster's no-signal frames (npy + companion jpg) with a
        '-dark' suffix so they are identifiable in a directory listing,
        and update the slice + run manifests and the raster metadata to
        the new paths.
        """

        cells = metadata.get("Cells", [])
        dark_paths = {
            path
            for cell in cells
            if not cell.get("AnySignal")
            for path in cell.get("Paths", [])
            if path
        }

        if not dark_paths:
            return records, metadata

        slice_dir = self.writer.run_dir / y_name
        mapping: dict[str, str] = {}

        for old_str in dark_paths:
            old = Path(old_str)

            if old.stem.endswith(DARK_FRAME_SUFFIX):
                continue  # already labeled (should not happen, but harmless)

            if not old.exists():
                fallback = slice_dir / old.name
                if fallback.exists():
                    old = fallback
                else:
                    logger.warning(
                        f"Cannot label dark frame (file not found): {old_str}"
                    )
                    continue

            new = old.with_name(f"{old.stem}{DARK_FRAME_SUFFIX}{old.suffix}")

            try:
                old.rename(new)

                jpg_old = old.with_suffix(".jpg")
                if jpg_old.exists():
                    jpg_old.rename(new.with_suffix(".jpg"))
            except OSError as ex:
                logger.warning(f"Could not rename dark frame {old}: {ex}")
                continue

            # Manifest entries store the ORIGINAL path string.
            mapping[old_str] = str(Path(old_str).with_name(new.name))

        if not mapping:
            return records, metadata

        for manifest in (self.writer.manifest_path, slice_dir / "frames.jsonl"):
            self._rewrite_manifest_paths(manifest, mapping)

        records = [
            dataclass_replace(record, Path=mapping.get(record.Path, record.Path))
            for record in records
        ]

        for cell in cells:
            cell["Paths"] = [
                mapping.get(path, path) if path else path
                for path in cell.get("Paths", [])
            ]

        logger.info(
            f"Labeled {len(mapping)} proof-of-darkness frame(s) with the "
            f"'{DARK_FRAME_SUFFIX}' filename suffix (kept in the dataset)."
        )

        return records, metadata

    @staticmethod
    def _rewrite_manifest_paths(manifest_path: Path, mapping: dict[str, str]) -> None:
        if not manifest_path.exists():
            logger.warning(
                f"Manifest {manifest_path} not found while relabeling dark "
                "frames; its entries keep the old paths."
            )
            return

        lines = []

        for line in manifest_path.read_text().splitlines():
            record = json.loads(line)
            if record.get("Path") in mapping:
                record["Path"] = mapping[record["Path"]]
            lines.append(json.dumps(record))

        manifest_path.write_text("\n".join(lines) + "\n")

    def _write_raster_metadata(
        self, y_name: str, machine_y_mm: float, metadata: dict
    ) -> None:
        slice_dir = self.writer.run_dir / y_name
        slice_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "MachineY_mm": machine_y_mm,
            "BeamY_mm": self.config.beam_y_mm(machine_y_mm),
            **metadata,
        }

        (slice_dir / "raster_metadata.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )

    # -- everything ----------------------------------------------------

    def run(self, machine_limits) -> list:
        setup = {
            "PlacementID": self.config.PlacementID,
            "CoordinateConvention": (
                "X horizontal transverse, Y beam propagation, Z vertical; "
                "machine == table except TableY, anchored to the optic."
            ),
            "MeasuredSensorY_mm": self.config.MeasuredSensorY_mm,
            "MeasuredFrom": self.config.MeasuredFrom,
            "YStart_machine_mm": self.config.YStart_machine_mm,
            "YStop_machine_mm": self.config.YStop_machine_mm,
            "YStep_mm": self.config.YStep_mm,
            "BeamDirectionSign": self.config.BeamDirectionSign,
            "X": self.config.X,
            "Z": self.config.Z,
            "RasterMode": self.config.RasterMode,
            "NShots": self.config.NShots,
            "BackgroundMode": self.config.BackgroundMode,
            "BackgroundShots": self.config.BackgroundShots,
            "Metadata": self.config.Metadata,
            "PlacementNotes": self.config.placement_notes(),
        }

        if self.config.BackgroundMode == "offaxis":
            bg_x, bg_z = self.background_xz(machine_limits)
            setup["BackgroundXZ_mm"] = [bg_x, bg_z]
            setup["BackgroundExposureChangeFraction"] = (
                self.config.BackgroundExposureChangeFraction
            )

        if self.config.BackgroundMode == "ladder":
            setup["BackgroundExposures_us"] = list(
                self.config.BackgroundExposures_us
                or default_background_ladder_us()
            )

        self.writer.write_json_artifact("auto_scan_setup.json", setup)

        if self.config.BackgroundMode == "ladder":
            self.capture_background_ladder()

        return self.run_y_stack(machine_limits)
