"""
Automated Z-stack acquisition on the FluidNC gantry.

One "placement" = one physical position of the CNC gantry on the optics
table, and runs as:

    1. You manually measure the optic -> camera-sensor distance with the
       camera at the scan's starting machine Z, and enter it when prompted.
    2. Background frames (beam blocked) are captured across an exposure
       LADDER, so any per-z calibrated exposure later has a near-matching
       background for subtraction.
    3. At each machine Z step: exposure is auto-calibrated (headless, seeded
       from the previous z), then an XY raster of frames is captured.
    4. Frames land in per-z subfolders of the run directory, named by the
       table z-position, e.g. run_dir/z0100.00cm/, each with its own
       frames.jsonl and calibration JSON. Background frames land in
       run_dir/background/.

Machine-coordinate conventions (see fluidnc_stage.py): GantryPosition_mm is
the absolute machine coordinate; +Z is up = along the beam = away from the
axicon, so TableZ_mm = MeasuredSensorZ_mm + (machineZ - ZStart_machine).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dataclass_replace
from pathlib import Path
from typing import Any, Callable, Optional
import json
import logging

import numpy as np

from adaptive_raster import AdaptiveRasterConfig, AdaptiveRasterRunner
from coordinates import (
    AxisRange,
    Bounds2D,
    GantryPlacement,
    ScanPoint,
    Vec3D,
    XYCrossSectionPlan,
)
from headless_calibration import (
    HeadlessCalibrationConfig,
    HeadlessCalibrationResult,
    calibrate_exposure_headless,
)

logger = logging.getLogger(__name__)

BACKGROUND_SUBFOLDER = "background"


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

    # Manually measured optic -> camera-sensor distance (mm) with the camera
    # at machine Z = ZStart_machine_mm, plus what it was measured from.
    MeasuredSensorZ_mm: float
    SensorZReference: str = "axicon3"

    # Machine-coordinate Z stack (negative values; larger = farther from
    # the optic). The scan starts at ZStart and steps by ZStep_mm.
    ZStart_machine_mm: float = -120.0
    ZStop_machine_mm: float = -5.0
    ZStep_mm: float = 10.0

    # XY raster (machine coordinates) captured at every Z. In adaptive
    # raster mode these are the CAPS (maximum extent), not the fixed grid.
    X: AxisRange = AxisRange(start_mm=45.0, stop_mm=75.0, step_mm=5.0)
    Y: AxisRange = AxisRange(start_mm=65.0, stop_mm=95.0, step_mm=4.0)

    # "adaptive": start from one seed frame at the calibration point and
    #   grow the rectangle only while its edge frames still contain signal
    #   (background-referenced). If the beam fits in one camera frame, the
    #   slice completes after a single position.
    # "fixed": raster the full X x Y grid at every slice.
    RasterMode: str = "adaptive"

    # Adaptive-raster signal test: a border strip "has signal" when at
    # least MinSignalPixels exceed (background p99 + SignalMargin_counts).
    SignalMargin_counts: float = 8.0
    MinSignalPixels: int = 50
    BorderStripFraction: float = 0.15

    # "any" = orientation-independent border test (default, safe;
    # overshoots the beam extent by <= ~1 frame per side). "directional"
    # tests only the outward-facing strip and needs the image->machine axis
    # mapping below verified once on hardware.
    BorderTest: str = "any"
    ImageTranspose: bool = False
    ImageFlipX: bool = False
    ImageFlipY: bool = False

    # Signal threshold fallback when no per-slice background exists
    # (background mode "ladder" or "none").
    FallbackBackgroundP99_counts: float = 5.0

    # Off-axis mode: recapture the background only when the calibrated
    # exposure has changed by at least this fraction since the background
    # was last CAPTURED (cumulative, so slow drift still triggers a
    # refresh). Slices that reuse an earlier background record it in their
    # background_reference.json. 0.0 = capture at every slice.
    BackgroundExposureChangeFraction: float = 0.10

    # Where the camera sits during exposure calibration (beam core).
    # None = center of the XY raster.
    CalibrationX_mm: Optional[float] = None
    CalibrationY_mm: Optional[float] = None

    NShots: int = 1

    # How beam-off/ambient backgrounds are captured:
    #
    #   "offaxis" (default): at each Z, right after exposure calibration,
    #       the camera moves to an XY position outside the beam and captures
    #       backgrounds at that slice's calibrated exposure — exact exposure
    #       match, per-slice drift tracking, no manual beam blocking.
    #       Captures ambient light only (plus any stray scatter reaching the
    #       off-axis position); validate once on hardware that the position
    #       is beam-free at the highest exposure in the stack.
    #   "ladder": once per placement, you block the beam when prompted and a
    #       log-spaced exposure ladder is captured (subtract the nearest /
    #       interpolated rung at analysis time).
    #   "none": no backgrounds (quick alignment runs).
    BackgroundMode: str = "offaxis"

    # Off-axis background position (machine coords). None = automatically
    # use the machine-limit corner farthest from the calibration point.
    BackgroundX_mm: Optional[float] = None
    BackgroundY_mm: Optional[float] = None

    # Ladder mode only: exposures for the beam-blocked ladder.
    BackgroundExposures_us: tuple[float, ...] = ()

    BackgroundShots: int = 3

    Metadata: dict[str, Any] = field(default_factory=dict)

    def z_values_machine_mm(self) -> list[float]:
        return AxisRange(
            start_mm=self.ZStart_machine_mm,
            stop_mm=self.ZStop_machine_mm,
            step_mm=self.ZStep_mm,
        ).values()

    def calibration_xy(self) -> tuple[float, float]:
        x = (
            self.CalibrationX_mm
            if self.CalibrationX_mm is not None
            else (self.X.start_mm + self.X.stop_mm) / 2.0
        )
        y = (
            self.CalibrationY_mm
            if self.CalibrationY_mm is not None
            else (self.Y.start_mm + self.Y.stop_mm) / 2.0
        )
        return x, y

    def placement(self) -> GantryPlacement:
        """
        TableOrigin such that gantry_to_table(z=ZStart) lands on the
        measured optic->sensor distance. Table X/Y are left in gantry-local
        coordinates (origin 0), since only z is measured against the optic.
        """

        return GantryPlacement(
            PlacementID=self.PlacementID,
            TableOrigin_mm=Vec3D(
                x_mm=0.0,
                y_mm=0.0,
                z_mm=self.MeasuredSensorZ_mm - self.ZStart_machine_mm,
            ),
            Notes=(
                f"MeasuredSensorZ={self.MeasuredSensorZ_mm:g}mm after "
                f"{self.SensorZReference} at machine Z={self.ZStart_machine_mm:g}mm"
            ),
        )

    def table_z_mm(self, machine_z_mm: float) -> float:
        return self.MeasuredSensorZ_mm + (machine_z_mm - self.ZStart_machine_mm)


def z_subfolder_name(table_z_mm: float) -> str:
    """
    Zero-padded, lexically sortable z folder name, e.g. 'z0100.00cm'.
    """
    return f"z{table_z_mm / 10.0:07.2f}cm"


# ---------------------------------------------------------------------------
# Camera helpers (software-trigger grab + exposure set on the writer's cam)
# ---------------------------------------------------------------------------


def _grab_frame(writer, timeout_ms: int) -> Optional[np.ndarray]:
    writer._execute_software_trigger()

    image_result = writer.cam.GetNextImage(timeout_ms)

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
    Drives one placement's Z stack through an already-prepared
    FLIRDatasetWriter whose stage_controller is a FluidNCStageController.
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

        self.placement = config.placement()
        self.records = []
        self.calibrations: dict[str, HeadlessCalibrationResult] = {}

        # Off-axis background reuse state: set whenever a background is
        # actually captured, consulted before capturing the next one.
        self._last_background: Optional[dict[str, Any]] = None

    # -- motion --------------------------------------------------------

    def _move_to(self, x_mm: float, y_mm: float, z_mm: float) -> None:
        point = ScanPoint(
            PlacementID=self.placement.PlacementID,
            GantryPosition_mm=Vec3D(x_mm=x_mm, y_mm=y_mm, z_mm=z_mm),
            TablePosition_mm=self.placement.gantry_to_table(
                Vec3D(x_mm=x_mm, y_mm=y_mm, z_mm=z_mm)
            ),
        )
        self.writer._move_and_wait(point)

    # -- backgrounds ----------------------------------------------------

    def background_xy(self, machine_limits) -> tuple[float, float]:
        """
        Off-axis background position: explicit config values if given,
        otherwise the machine-limit corner farthest from the calibration
        point (i.e. as far from the beam as this placement allows).
        """

        if (
            self.config.BackgroundX_mm is not None
            and self.config.BackgroundY_mm is not None
        ):
            return self.config.BackgroundX_mm, self.config.BackgroundY_mm

        calib_x, calib_y = self.config.calibration_xy()

        corners = [
            (machine_limits.x_min_mm, machine_limits.y_min_mm),
            (machine_limits.x_min_mm, machine_limits.y_max_mm),
            (machine_limits.x_max_mm, machine_limits.y_min_mm),
            (machine_limits.x_max_mm, machine_limits.y_max_mm),
        ]

        return max(
            corners,
            key=lambda c: (c[0] - calib_x) ** 2 + (c[1] - calib_y) ** 2,
        )

    def capture_background_offaxis(
        self,
        machine_z_mm: float,
        exposure_us: float,
        machine_limits,
        z_name: str,
    ) -> list:
        """
        Move the camera out of the beam and capture ambient backgrounds at
        this slice's already-applied calibrated exposure. Frames land in the
        slice's own z subfolder, tagged as backgrounds.
        """

        bg_x, bg_y = self.background_xy(machine_limits)
        self._move_to(bg_x, bg_y, machine_z_mm)

        point = ScanPoint(
            PlacementID=self.placement.PlacementID,
            GantryPosition_mm=Vec3D(bg_x, bg_y, machine_z_mm),
            TablePosition_mm=self.placement.gantry_to_table(
                Vec3D(bg_x, bg_y, machine_z_mm)
            ),
            NShots=self.config.BackgroundShots,
            Metadata={
                "ScanKind": "Background",
                "BackgroundMode": "OffAxisAmbient",
                "Subfolder": z_name,
                "FileTag": "background",
                "Exposure_us": exposure_us,
                "MachineZ_mm": machine_z_mm,
                "TableZ_mm": self.config.table_z_mm(machine_z_mm),
                "SensorZReference": self.config.SensorZReference,
                **self.config.Metadata,
            },
        )

        records = self.writer.acquire_at_current_position(point)
        self.records.extend(records)
        return records

    def background_for_slice(
        self,
        machine_z_mm: float,
        exposure_us: float,
        machine_limits,
        z_name: str,
    ) -> list:
        """
        Off-axis background with change-based cadence: capture a fresh
        background only when the calibrated exposure moved by at least
        BackgroundExposureChangeFraction since the last captured one;
        otherwise reuse it. Either way, background_reference.json in the
        z folder records which background frames apply to this slice.
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
                machine_z_mm, exposure_us, machine_limits, z_name
            )

            self._last_background = {
                "Exposure_us": exposure_us,
                "Records": records,
                "ZName": z_name,
                "MachineZ_mm": machine_z_mm,
            }

            bg_x, bg_y = self.background_xy(machine_limits)
            reason = (
                "first slice"
                if change is None
                else f"exposure changed {change * 100.0:.1f}%"
            )
            logger.info(
                f"{len(records)} off-axis background frame(s) at "
                f"X{bg_x:g} Y{bg_y:g} ({reason})"
            )
        else:
            records = last["Records"]
            logger.info(
                f"reusing background from {last['ZName']}/ "
                f"(exposure changed {change * 100.0:.1f}% < "
                f"{self.config.BackgroundExposureChangeFraction * 100.0:g}%)"
            )

        self._write_background_reference(
            z_name, exposure_us, change, captured=must_capture
        )

        return records

    def _write_background_reference(
        self,
        z_name: str,
        exposure_us: float,
        change: Optional[float],
        captured: bool,
    ) -> None:
        last = self._last_background

        z_dir = self.writer.run_dir / z_name
        z_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "Reused": not captured,
            "SliceExposure_us": exposure_us,
            "BackgroundExposure_us": last["Exposure_us"],
            "BackgroundZ": last["ZName"],
            "BackgroundMachineZ_mm": last["MachineZ_mm"],
            "ExposureChangeFraction": change,
            "MaxExposureChangeFraction": (
                self.config.BackgroundExposureChangeFraction
            ),
            "BackgroundPaths": [
                getattr(record, "Path", None) for record in last["Records"]
            ],
        }

        (z_dir / "background_reference.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )

    # -- ladder mode: beam-blocked backgrounds, once per placement -----

    def capture_background_ladder(self) -> list:
        exposures = self.config.BackgroundExposures_us or default_background_ladder_us()

        calib_x, calib_y = self.config.calibration_xy()
        self._move_to(calib_x, calib_y, self.config.ZStart_machine_mm)

        self.pause_fn(
            "BLOCK THE BEAM now (backgrounds for this placement are about to "
            f"be captured at {len(exposures)} exposures)."
        )

        records = []

        for rung_idx, exposure_us in enumerate(exposures):
            actual_us = set_exposure_us(self.writer.cam, exposure_us)

            point = ScanPoint(
                PlacementID=self.placement.PlacementID,
                GantryPosition_mm=Vec3D(calib_x, calib_y, self.config.ZStart_machine_mm),
                TablePosition_mm=self.placement.gantry_to_table(
                    Vec3D(calib_x, calib_y, self.config.ZStart_machine_mm)
                ),
                NShots=self.config.BackgroundShots,
                Metadata={
                    "ScanKind": "Background",
                    "Subfolder": BACKGROUND_SUBFOLDER,
                    "FileTag": f"exp{actual_us:09.1f}us",
                    "Exposure_us": actual_us,
                    "LadderIndex": rung_idx,
                    "SensorZReference": self.config.SensorZReference,
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

    # -- step 3: per-z calibration -------------------------------------

    def calibrate_at(
        self,
        machine_z_mm: float,
        start_exposure_us: float,
    ) -> HeadlessCalibrationResult:
        calib_x, calib_y = self.config.calibration_xy()
        self._move_to(calib_x, calib_y, machine_z_mm)

        timeout_ms = self.writer.config.AcquisitionTimeout_ms

        self.writer._begin_acquisition()

        try:
            result = calibrate_exposure_headless(
                grab_frame=lambda: _grab_frame(self.writer, timeout_ms),
                set_exposure_us=lambda us: set_exposure_us(self.writer.cam, us),
                start_exposure_us=start_exposure_us,
                config=self.calibration_config,
            )
        finally:
            self.writer._end_acquisition()

        self._save_calibration(machine_z_mm, result)
        return result

    def _save_calibration(
        self, machine_z_mm: float, result: HeadlessCalibrationResult
    ) -> None:
        table_z_mm = self.config.table_z_mm(machine_z_mm)
        z_dir = self.writer.run_dir / z_subfolder_name(table_z_mm)
        z_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "MachineZ_mm": machine_z_mm,
            "TableZ_mm": table_z_mm,
            "SensorZReference": self.config.SensorZReference,
            "FinalExposure_us": result.FinalExposure_us,
            "LastMax": result.LastMax,
            "LastSaturatedPixels": result.LastSaturatedPixels,
            "Iterations": result.Iterations,
            "Converged": result.Converged,
            "Note": result.Note,
        }

        (z_dir / "calibration_result.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )

        # Full camera-settings JSON, same schema as the calibrations/ dir,
        # when the writer holds a real FLIRCameraSettings dataclass.
        settings = self.writer.camera_settings
        if hasattr(settings, "__dataclass_fields__"):
            calibrated = dataclass_replace(
                settings, ExposureTime=result.FinalExposure_us
            )
            calibrated.to_json_file(z_dir / "calibrated_camera_settings.json")

        self.calibrations[z_subfolder_name(table_z_mm)] = result

    # -- steps 3-5: the Z stack ----------------------------------------

    def run_z_stack(self, machine_limits) -> list:
        """
        For each machine Z: headless exposure calibration, then the XY
        raster. machine_limits is the Bounds3D the stage controller
        enforces (pass FluidNCStageController.config.MachineLimits_mm).
        """

        z_values = self.config.z_values_machine_mm()
        roi = Bounds2D(
            x_min_mm=self.config.X.start_mm,
            x_max_mm=self.config.X.stop_mm,
            y_min_mm=self.config.Y.start_mm,
            y_max_mm=self.config.Y.stop_mm,
        )

        exposure_seed_us = getattr(
            self.writer.camera_settings, "ExposureTime", None
        ) or 1000.0

        for z_idx, machine_z in enumerate(z_values):
            table_z_mm = self.config.table_z_mm(machine_z)
            z_name = z_subfolder_name(table_z_mm)

            logger.info(
                f"[{z_idx + 1}/{len(z_values)}] machine Z {machine_z:g} mm "
                f"(table z = {table_z_mm / 10.0:g} cm after "
                f"{self.config.SensorZReference}) -> {z_name}/"
            )

            calibration = self.calibrate_at(machine_z, exposure_seed_us)
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
                    machine_z,
                    calibration.FinalExposure_us,
                    machine_limits,
                    z_name,
                )

            point_metadata = {
                "ScanKind": "AutoZStack",
                "Subfolder": z_name,
                "MachineZ_mm": machine_z,
                "TableZ_mm": table_z_mm,
                "SensorZReference": self.config.SensorZReference,
                "Exposure_us": calibration.FinalExposure_us,
                **self.config.Metadata,
            }

            if self.config.RasterMode == "adaptive":
                records, raster_metadata = self._run_adaptive_raster(
                    machine_z,
                    machine_limits,
                    background_records,
                    point_metadata,
                )
            else:
                plan = XYCrossSectionPlan(
                    Placement=self.placement,
                    MachineLimits=machine_limits,
                    ROI=roi,
                    GantryZ_mm=machine_z,
                    X=self.config.X,
                    Y=self.config.Y,
                    NShots=self.config.NShots,
                    Metadata=point_metadata,
                )

                points = plan.generate_points()
                records = self.writer.acquire_scan(points)

                raster_metadata = {
                    "RasterMode": "fixed",
                    "Step_mm": [self.config.X.step_mm, self.config.Y.step_mm],
                    "GridShape": [
                        len(self.config.X.values()),
                        len(self.config.Y.values()),
                    ],
                    "FinalRect_mm": {
                        "XMin": self.config.X.start_mm,
                        "XMax": self.config.X.stop_mm,
                        "YMin": self.config.Y.start_mm,
                        "YMax": self.config.Y.stop_mm,
                    },
                    "CellsCaptured": len(points),
                }

            self.records.extend(records)
            self._write_raster_metadata(z_name, machine_z, raster_metadata)

            logger.info(f"{len(records)} frames -> {z_name}/")

        return self.records

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
        machine_z_mm: float,
        machine_limits,
        background_records,
        point_metadata: dict,
    ) -> tuple[list, dict]:
        threshold, threshold_source = self.signal_threshold(background_records)
        calib_x, calib_y = self.config.calibration_xy()

        # Caps: the configured X/Y ranges, further clamped to machine limits.
        x_min = max(self.config.X.start_mm, machine_limits.x_min_mm)
        x_max = min(self.config.X.stop_mm, machine_limits.x_max_mm)
        y_min = max(self.config.Y.start_mm, machine_limits.y_min_mm)
        y_max = min(self.config.Y.stop_mm, machine_limits.y_max_mm)

        raster_config = AdaptiveRasterConfig(
            CenterX_mm=calib_x,
            CenterY_mm=calib_y,
            StepX_mm=self.config.X.step_mm,
            StepY_mm=self.config.Y.step_mm,
            XMin_mm=x_min,
            XMax_mm=x_max,
            YMin_mm=y_min,
            YMax_mm=y_max,
            SignalThreshold_counts=threshold,
            MinSignalPixels=self.config.MinSignalPixels,
            BorderStripFraction=self.config.BorderStripFraction,
            BorderTest=self.config.BorderTest,
            ImageTranspose=self.config.ImageTranspose,
            ImageFlipX=self.config.ImageFlipX,
            ImageFlipY=self.config.ImageFlipY,
        )

        def capture(x_mm: float, y_mm: float, i: int, j: int):
            point = ScanPoint(
                PlacementID=self.placement.PlacementID,
                GantryPosition_mm=Vec3D(x_mm, y_mm, machine_z_mm),
                TablePosition_mm=self.placement.gantry_to_table(
                    Vec3D(x_mm, y_mm, machine_z_mm)
                ),
                NShots=self.config.NShots,
                Metadata={**point_metadata, "GridI": i, "GridJ": j},
            )

            records = self.writer.acquire_scan([point])
            arr = np.load(records[0].Path)
            return records, arr

        runner = AdaptiveRasterRunner(raster_config, capture)
        result = runner.run()

        metadata = result.Metadata
        metadata["SignalThresholdSource"] = threshold_source

        logger.info(
            f"adaptive raster: {metadata['CellsCaptured']} cells "
            f"(grid {metadata['GridShape'][0]}x{metadata['GridShape'][1]}) "
            f"vs {metadata['FixedGridCells']} for the full fixed grid"
        )

        return result.Records, metadata

    def _write_raster_metadata(
        self, z_name: str, machine_z_mm: float, metadata: dict
    ) -> None:
        z_dir = self.writer.run_dir / z_name
        z_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "MachineZ_mm": machine_z_mm,
            "TableZ_mm": self.config.table_z_mm(machine_z_mm),
            **metadata,
        }

        (z_dir / "raster_metadata.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )

    # -- everything ----------------------------------------------------

    def run(self, machine_limits) -> list:
        setup = {
            "PlacementID": self.config.PlacementID,
            "MeasuredSensorZ_mm": self.config.MeasuredSensorZ_mm,
            "SensorZReference": self.config.SensorZReference,
            "ZStart_machine_mm": self.config.ZStart_machine_mm,
            "ZStop_machine_mm": self.config.ZStop_machine_mm,
            "ZStep_mm": self.config.ZStep_mm,
            "X": self.config.X,
            "Y": self.config.Y,
            "NShots": self.config.NShots,
            "BackgroundMode": self.config.BackgroundMode,
            "BackgroundShots": self.config.BackgroundShots,
            "Metadata": self.config.Metadata,
            "PlacementNotes": self.placement.Notes,
        }

        if self.config.BackgroundMode == "offaxis":
            bg_x, bg_y = self.background_xy(machine_limits)
            setup["BackgroundXY_mm"] = [bg_x, bg_y]

        if self.config.BackgroundMode == "ladder":
            setup["BackgroundExposures_us"] = list(
                self.config.BackgroundExposures_us
                or default_background_ladder_us()
            )

        self.writer.write_json_artifact("auto_scan_setup.json", setup)

        if self.config.BackgroundMode == "ladder":
            self.capture_background_ladder()

        return self.run_z_stack(machine_limits)
