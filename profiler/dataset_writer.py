from __future__ import annotations  # convert type hints to strings at runtime

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional
import datetime as dt
import json
import logging
import re
import threading
import time
import uuid

import numpy as np
import PySpin

from camera_settings import FLIRCameraSettings
from coordinates import ScanPoint, Vec3D

from camera_base import FLIRCameraControllerBase

logger = logging.getLogger(__name__)


class DatasetWriterError(RuntimeError):
    pass

class DatasetWriterJobType:
    """Enum for the type of dataset writer job."""
    AUTO_SCAN = "auto_scan"
    MANUAL_SCAN = "manual_scan"
    STATIC = "static"

@dataclass(frozen=True)
class DatasetWriterConfig:
    JobType: str
    DatasetRoot: Path
    AcquisitionTimeout_ms: int = 2000
    StageTimeout_s: float = 30.0
    SettleTime_s: float = 0.0

    # Pause between BeginAcquisition and the first software trigger. A
    # trigger fired while the camera is still arming can be silently
    # dropped (GenTL -1011 timeout on GetNextImage); this avoids paying
    # the timeout-and-retry penalty at every raster point.
    TriggerArmDelay_s: float = 0.1

    # Pause before restarting acquisition after in-place re-triggers fail
    # (persistent -1011). Gives a congested/stalled camera link a moment
    # to drain before the stream restart; observed 2026-08-03 alongside
    # FluidNC WiFi drops and Slack post timeouts (flaky lab network).
    StallRecoveryPause_s: float = 3.0

    # Save uint8 / uint16 arrays in binary numpy format.
    # We can encode to BMP or PNG later, but want to avoid any lossy compression at this stage.
    ImageExtension: str = ".npy"

    # Group frames into per-slice subfolders of the run directory (named
    # from TablePosition_mm.y_mm, the distance along the beam) when no
    # explicit Metadata["Subfolder"] is set. Coordinate convention:
    # X horizontal transverse, Y beam propagation, Z vertical.
    GroupByYSubfolder: bool = False

    def make_run_dir(self) -> Path:
        now = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        return self.DatasetRoot / f"{self.JobType}-{now}"


@dataclass
class AcquisitionSignals:
    """
    threading.Event objects used to coordinate the acquisition order between
    the dataset writer and a stepper-motor controller.

    The important safety gate is:

        MovementComplete must be set before the writer calls BeginAcquisition().

    Later, the stepper-motor controller can own MovementStarted/MovementComplete signals.
    """

    MovementStarted: threading.Event = field(default_factory=threading.Event)
    MovementComplete: threading.Event = field(default_factory=threading.Event)

    AcquisitionStarted: threading.Event = field(default_factory=threading.Event)
    FrameBuffered: threading.Event = field(default_factory=threading.Event)
    FrameWritten: threading.Event = field(default_factory=threading.Event)

    StopRequested: threading.Event = field(default_factory=threading.Event)

    # TODO: potential speed-up
    # Add a "FrameAcquired" signal that is set after the writer has acquired
    # the frame but before it writes to disk. This would allow a future
    # stepper-motor controller to start moving while the writer is still
    # writing the frame to disk.

    def reset_for_position(self) -> None:
        self.MovementStarted.clear()
        self.MovementComplete.clear()
        self.AcquisitionStarted.clear()
        self.FrameBuffered.clear()
        self.FrameWritten.clear()


class StageController:
    """
    Stub for future stepper-motor integration.

    The dataset writer assumes:

        move_to_scan_point(...) starts or performs the motion
        wait_until_motion_complete(...) blocks until the stage is actually still
    """

    def move_to_scan_point(
        self,
        point: ScanPoint,
        signals: AcquisitionSignals,
    ) -> None:
        # TODO move to XYZ:
        #   point.GantryPosition_mm.x_mm
        #   point.GantryPosition_mm.y_mm
        #   point.GantryPosition_mm.z_mm
        signals.MovementStarted.set()
        time.sleep(1)

    def wait_until_motion_complete(
        self,
        point: ScanPoint,
        timeout_s: float,
        signals: AcquisitionSignals,
    ) -> None:
        # TODO check if XYZ motion is complete.
        time.sleep(1)
        signals.MovementComplete.set()


# The FluidNC implementation of this interface lives in
# fluidnc_stage.FluidNCStageController (kept in its own module so this one
# stays importable without any networking / gantry concerns).


@dataclass(frozen=True)
class FrameRecord:
    """
    Metadata for one saved camera frame.

    GantryPosition_mm is the motor-control coordinate.
    TablePosition_mm is the optics-table / beamline reconstruction coordinate.
    """

    Path: str

    PlacementID: str
    GantryPosition_mm: Vec3D
    TablePosition_mm: Vec3D

    ShotIndex: int
    TimestampUTC: str

    Shape: tuple[int, ...]
    DType: str
    Min: int
    Max: int
    SaturatedPixelCount: Optional[int]

    Extra: dict[str, Any] = field(default_factory=dict)


class FLIRDatasetWriter(FLIRCameraControllerBase):
    def __init__(
        self,
        camera_index: int,
        camera_settings: FLIRCameraSettings,
        config: DatasetWriterConfig,
        stage_controller: Optional[StageController] = None,
        signals: Optional[AcquisitionSignals] = None,
        cam: Optional[Any] = None,
        camera_serial: Optional[str] = None,
    ):

        self.camera_index = camera_index
        self.camera_settings = camera_settings
        self.config = config
        self.stage_controller = stage_controller or StageController()
        self.signals = signals or AcquisitionSignals()

        self.run_dir = self.config.make_run_dir()
        self.manifest_path = self.run_dir / "frames.jsonl"

        # Optional callable invoked after a camera reconnect, for state
        # that lives on the camera but outside camera_settings (e.g. the
        # per-slice calibrated exposure set directly on the node).
        self.RestoreState = None

        # `cam` allows dependency injection of a fake camera for unit tests.
        super().__init__(
            camera_index, camera_settings, cam=cam, camera_serial=camera_serial
        )


    def prepare_run(self) -> Path:
        """
        Create run directory, apply camera settings, and save camera_settings.json.
        """

        self.run_dir.mkdir(parents=True, exist_ok=False)

        # Apply known settings at the start of the run.
        self.camera_settings.apply(self.cam, strict=True)

        # Save the intended calibration/acquisition settings.
        self.camera_settings.to_json_file(self.run_dir / "camera_settings.json")

        # Also useful for reconstructing the run later.
        run_metadata = {
            "CreatedUTC": _utc_now(),
            "RunDir": str(self.run_dir),
            "Config": _dataclass_to_jsonable(self.config),
            "CoordinateConvention": {
                "GantryPosition_mm": "Local CNC/gantry coordinates used for motion commands.",
                "TablePosition_mm": "Optics-table / beamline coordinates used for reconstruction.",
                "PlacementID": "Physical placement of the CNC gantry on the optics table.",
            },
        }

        (self.run_dir / "run_metadata.json").write_text(
            json.dumps(run_metadata, indent=2) + "\n"
        )

        return self.run_dir

    def write_json_artifact(self, filename: str, payload: Any) -> Path:
        """
        Save scan plans, placements, machine limits, notes, etc. beside the data.

        Example:
            writer.write_json_artifact("scan_plan.json", plan)
        """

        if not filename.endswith(".json"):
            raise ValueError("filename must end with '.json'")

        path = self.run_dir / filename
        path.write_text(json.dumps(_dataclass_to_jsonable(payload), indent=2) + "\n")
        return path

    def acquire_static(
        self,
        *,
        nshots: int = 1,
        placement_id: str = "static-camera",
        gantry_position_mm: Optional[Vec3D] = None,
        table_position_mm: Optional[Vec3D] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[FrameRecord]:
        """
        Acquire one or more images with the camera at its current physical position.

        This is the no-gantry path to use while the camera is static. It creates a
        synthetic ScanPoint at the current position so the existing filename and
        manifest format stays compatible with future gantry scans.
        """

        if nshots < 1:
            raise ValueError("nshots must be at least 1.")

        point = ScanPoint(
            PlacementID=placement_id,
            GantryPosition_mm=gantry_position_mm or Vec3D(0.0, 0.0, 0.0),
            TablePosition_mm=table_position_mm or Vec3D(0.0, 0.0, 0.0),
            NShots=nshots,
            Metadata={
                "ScanKind": "Static",
                **(metadata or {}),
            },
        )

        return self.acquire_at_current_position(point)

    def acquire_at_current_position(self, point: ScanPoint) -> list[FrameRecord]:
        """
        Acquire frames for an existing ScanPoint without commanding gantry motion.

        This is useful if you manually place the camera and still want to record a
        real table coordinate, placement ID, or note in the normal manifest schema.
        """

        self.signals.reset_for_position()
        self.signals.MovementComplete.set()

        return self._acquire_point_frames(point)


    def acquire_scan(self, points: Iterable[ScanPoint]) -> list[FrameRecord]:
        """
        Acquire a scan from points generated by coordinates.XYCrossSectionPlan
        or coordinates.ZStackPlan.

        For each ScanPoint:

            1. command/wait for stage motion using point.GantryPosition_mm
            2. require MovementComplete signal
            3. wait optional settling time
            4. software-trigger image acquisition
            5. write arrays to disk
            6. record both gantry and table coordinates
        """

        records: list[FrameRecord] = []

        for point in points:
            if self.signals.StopRequested.is_set():
                break

            self._move_and_wait(point)
            records.extend(self._acquire_point_frames(point))


        return records

    def _move_and_wait(self, point: ScanPoint) -> None:
        self.signals.reset_for_position()

        # Hook for future stepper implementation.
        # Motor motion should use point.GantryPosition_mm, not point.TablePosition_mm.
        self.stage_controller.move_to_scan_point(point, self.signals)

        self.stage_controller.wait_until_motion_complete(
            point=point,
            timeout_s=self.config.StageTimeout_s,
            signals=self.signals,
        )

        ok = self.signals.MovementComplete.wait(timeout=self.config.StageTimeout_s)

        if not ok:
            raise DatasetWriterError(
                "Timed out waiting for stage motion complete at "
                f"PlacementID={point.PlacementID!r}, "
                f"GantryPosition_mm={point.GantryPosition_mm}."
            )

    def _acquire_point_frames(self, point: ScanPoint) -> list[FrameRecord]:
        if point.NShots < 1:
            raise ValueError("ScanPoint.NShots must be at least 1.")

        records: list[FrameRecord] = []

        if self.config.SettleTime_s > 0:
            time.sleep(self.config.SettleTime_s)

        self._begin_acquisition()

        if self.config.TriggerArmDelay_s > 0:
            time.sleep(self.config.TriggerArmDelay_s)

        try:
            for shot_idx in range(point.NShots):
                if self.signals.StopRequested.is_set():
                    break

                try:
                    record = self._acquire_one_frame(point, shot_idx)
                except DatasetWriterError as ex:
                    record = self._recover_shot(point, shot_idx, ex)

                self._append_manifest(record)
                records.append(record)

        finally:
            try:
                self._end_acquisition()
            except Exception as ex:  # noqa: BLE001 - dead handle on the way out
                logger.warning(f"EndAcquisition during cleanup failed: {ex}")

        return records

    def _acquire_one_frame(
        self,
        point: ScanPoint,
        shot_idx: int
    ) -> FrameRecord:
        """
        Acquire exactly one software-triggered frame after MovementComplete has fired.
        """

        if not self.signals.MovementComplete.is_set():
            raise DatasetWriterError(
                "Refusing to acquire: MovementComplete signal is not set."
            )

        self.signals.AcquisitionStarted.set()

        image_result = None

        try:
            image_result = self._trigger_and_get_image()

            if image_result.IsIncomplete():
                status = image_result.GetImageStatus()
                raise DatasetWriterError(f"Image incomplete; image status = {status}")

            image_data = image_result.GetNDArray()
            self.signals.FrameBuffered.set()

            # Force a detached NumPy copy before releasing the camera image pointer.
            arr = np.array(image_data, copy=True)

            extra = dict(point.Metadata)

            if hasattr(image_result, "GetFrameID"):
                extra["FrameID"] = int(image_result.GetFrameID())

            frame_path = self._frame_path(point, shot_idx)
            self._write_array(frame_path, arr)
            frame_path_jpg = str(frame_path.with_suffix(".jpg"))
            image_result.Save(frame_path_jpg)  # Save a JPG copy for quick viewing
            self.signals.FrameWritten.set()

            return self._build_frame_record(arr, frame_path, point, shot_idx, extra)

        except PySpin.SpinnakerException as ex:
            raise DatasetWriterError(f"Spinnaker acquisition failed: {ex}") from ex

        finally:
            if image_result is not None:
                image_result.Release()
                # A PySpin ImagePtr keeps the camera referenced even after
                # Release(); if this local survives in the traceback of a
                # propagating exception, the camera cannot be released at
                # cleanup (Spinnaker error -1004).
                image_result = None

    def _restart_acquisition(
        self,
        *,
        reconnect: bool,
        pause_s: float = 0.0,
    ) -> None:
        """
        End the current stream and start a fresh one, optionally doing a
        full camera reconnect (reopen by serial + re-apply settings) in
        between. Used by the mid-shot recovery ladder.
        """

        try:
            self._end_acquisition()
        except Exception as end_ex:  # noqa: BLE001 - dead handle
            logger.warning(
                f"EndAcquisition during acquisition restart failed: {end_ex}"
            )

        if pause_s > 0:
            time.sleep(pause_s)

        if reconnect:
            self.reconnect()

        self._begin_acquisition()

        if self.config.TriggerArmDelay_s > 0:
            time.sleep(self.config.TriggerArmDelay_s)

    def _recover_shot(
        self,
        point: ScanPoint,
        shot_idx: int,
        ex: DatasetWriterError,
    ) -> FrameRecord:
        """
        Mid-shot recovery ladder. Position is untouched in every branch,
        so retrying the shot is always safe.

        -1024 (bus removal, e.g. an Ethernet blip on the moving cable
        run): full reconnect, retry the shot once.

        -1011 (GetNextImage timeout AFTER the in-place re-triggers in
        _trigger_and_get_image were exhausted — a stalled stream or a
        network blackout, not a single dropped trigger): restart the
        stream after a pause and retry; if the stall persists, escalate
        to a full reconnect before giving up. Observed 2026-08-03
        killing auto scans in step with FluidNC WiFi drops.
        """

        if self._is_camera_removed_error(ex):
            logger.warning(
                f"Camera removal during shot {shot_idx} at "
                f"{point.GantryPosition_mm}: {ex}"
            )
            self._restart_acquisition(reconnect=True)
            return self._acquire_one_frame(point, shot_idx)

        if self._is_timeout_error(ex):
            logger.warning(
                f"Persistent GetNextImage timeout during shot {shot_idx} at "
                f"{point.GantryPosition_mm}: {ex} — restarting acquisition "
                f"after a {self.config.StallRecoveryPause_s:g} s pause."
            )
            self._restart_acquisition(
                reconnect=False,
                pause_s=self.config.StallRecoveryPause_s,
            )

            try:
                return self._acquire_one_frame(point, shot_idx)
            except DatasetWriterError as retry_ex:
                if not (
                    self._is_timeout_error(retry_ex)
                    or self._is_camera_removed_error(retry_ex)
                ):
                    raise
                logger.warning(
                    f"Acquisition restart did not recover the stream "
                    f"({retry_ex}); escalating to a full camera reconnect."
                )
                self._restart_acquisition(reconnect=True)
                return self._acquire_one_frame(point, shot_idx)

        raise ex

    def reconnect(self) -> None:
        """
        Recover the camera after a bus removal (-1024): reopen (by serial),
        re-apply settings, then restore any runtime camera state via the
        RestoreState hook (e.g. the slice's calibrated exposure).
        """

        logger.warning("Camera removed from the bus; attempting reconnect...")
        self.reopen()

        if self.RestoreState is not None:
            try:
                self.RestoreState()
            except Exception as ex:  # noqa: BLE001 - report, keep scanning
                logger.warning(
                    f"RestoreState after camera reconnect failed: {ex} — "
                    "camera is using the base settings' exposure until the "
                    "next calibration."
                )

    @staticmethod
    def _is_camera_removed_error(ex: Exception) -> bool:
        message = str(ex)
        return "-1024" in message or "no longer valid" in message

    @staticmethod
    def _is_timeout_error(ex: Exception) -> bool:
        return "-1011" in str(ex)

    def _trigger_and_get_image(self, retries: int = 2):
        """
        Software-trigger the camera and fetch the resulting image.

        A trigger executed while the camera is still (re-)arming after
        BeginAcquisition can be silently dropped, which surfaces as a
        GenTL -1011 timeout on GetNextImage even though the camera is
        healthy. Re-triggering recovers it, so retry a couple of times
        before giving up.
        """

        attempt = 0

        while True:
            self._execute_software_trigger()

            try:
                return self.cam.GetNextImage(self.config.AcquisitionTimeout_ms)
            except PySpin.SpinnakerException as ex:
                if attempt < retries and "-1011" in str(ex):
                    attempt += 1
                    logger.warning(
                        "GetNextImage timed out (dropped software trigger?); "
                        f"re-triggering (attempt {attempt}/{retries}). "
                        "If this repeats every frame, check that SpinView "
                        "is fully closed."
                    )
                    continue
                raise

    def save_frame_array(
        self,
        arr: np.ndarray,
        point: ScanPoint,
        shot_idx: int = 0,
        extra: Optional[dict[str, Any]] = None,
    ) -> FrameRecord:
        """
        Save an already-acquired NumPy frame to disk and append it to the
        manifest, without touching the camera.

        This is used by the manual translation-stage mode, where frames are
        grabbed continuously for a live preview and the user chooses which
        one to keep. The manifest record is identical in schema to frames
        acquired by acquire_scan / acquire_static.
        """

        merged_extra = {**dict(point.Metadata), **(extra or {})}

        frame_path = self._frame_path(point, shot_idx)
        self._write_array(frame_path, np.asarray(arr))

        self._write_img(np.asarray(arr), frame_path.with_suffix(".jpg"))

        record = self._build_frame_record(
            np.asarray(arr), frame_path, point, shot_idx, merged_extra
        )
        self._append_manifest(record)

        return record

    def _write_img(self, arr: np.ndarray, path: Path) -> None:
        """
        Save a JPG copy of the array for quick viewing.

        This is used by the manual translation-stage mode, where frames are
        grabbed continuously for a live preview and the user chooses which
        one to keep. The manifest record is identical in schema to frames
        acquired by acquire_scan / acquire_static.
        """

        import cv2

        if path.suffix != ".jpg":
            raise ValueError("path must end with '.jpg'")

        # Convert Mono16 to Mono8 for JPG saving.
        if arr.dtype == np.uint16:
            arr = (arr >> 8).astype(np.uint8)

        # Convert Mono8 to BGR for JPG saving.
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(str(path), arr)

    def _build_frame_record(
        self,
        arr: np.ndarray,
        frame_path: Path,
        point: ScanPoint,
        shot_idx: int,
        extra: dict[str, Any],
    ) -> FrameRecord:
        return FrameRecord(
            Path=str(frame_path),
            PlacementID=point.PlacementID,
            GantryPosition_mm=point.GantryPosition_mm,
            TablePosition_mm=point.TablePosition_mm,
            ShotIndex=shot_idx,
            TimestampUTC=_utc_now(),
            Shape=tuple(arr.shape),
            DType=str(arr.dtype),
            Min=int(np.min(arr)),
            Max=int(np.max(arr)),
            SaturatedPixelCount=_estimate_saturated_pixel_count(arr),
            Extra=extra,
        )

    def _frame_path(self, point: ScanPoint, shot_idx: int) -> Path:
        # Optional filename tag to keep otherwise-identical points unique,
        # e.g. the rungs of a background exposure ladder taken at one
        # position (Metadata["FileTag"] = "exp000123.0us").
        file_tag = (point.Metadata or {}).get("FileTag")
        tag_part = f"{_format_placement_id(str(file_tag))}-" if file_tag else ""

        filename = (
            f"{_format_placement_id(point.PlacementID)}-"
            f"{tag_part}"
            # TablePosition y = distance along the beam from the reference
            # optic (X horizontal transverse, Y propagation, Z vertical).
            f"tabley{_format_mm(point.TablePosition_mm.y_mm)}-"
            f"gantry{_format_vec3(point.GantryPosition_mm)}-"
            f"shot{shot_idx:04d}"
            f"{self.config.ImageExtension}"
        )

        return self._point_dir(point) / filename

    def _point_dir(self, point: ScanPoint) -> Path:
        """
        Directory a point's frames are saved in.

        Frames are grouped into a per-position subfolder of the run
        directory when either:

            * point.Metadata["Subfolder"] names one explicitly (used by the
              auto scan for beam-position folders and the background
              ladder), or
            * config.GroupByYSubfolder is set, in which case the table
              y-position (distance along the beam) is used,
              e.g. "y_p001000.000mm/".
        """

        name = (point.Metadata or {}).get("Subfolder")

        if not name and self.config.GroupByYSubfolder:
            name = f"y_{_format_mm(point.TablePosition_mm.y_mm)}"

        if not name:
            return self.run_dir

        subdir = self.run_dir / _format_placement_id(str(name))
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir

    def _write_array(self, path: Path, arr: np.ndarray) -> None:
        if path.suffix == ".npy":
            np.save(path, arr)
            return

        if path.suffix == ".raw":
            arr.tofile(path)
            return

        raise DatasetWriterError(
            f"Unsupported image extension {path.suffix!r}. "
            "Use '.npy' for now."
        )

    def _append_manifest(self, record: FrameRecord) -> None:
        line = json.dumps(_dataclass_to_jsonable(record)) + "\n"

        with self.manifest_path.open("a") as f:
            f.write(line)

        # Frames grouped into a subfolder also get a local frames.jsonl so
        # each z-position folder is self-contained (e.g. for per-z
        # stitching with stitcher.stitch_run_dir on the subfolder).
        parent = Path(record.Path).parent

        if parent != self.run_dir:
            with (parent / "frames.jsonl").open("a") as f:
                f.write(line)



def _format_placement_id(placement_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", placement_id).strip("_")
    return safe or "placement"


def _format_mm(value_mm: float) -> str:
    """
    Filename-safe millimeter formatter.

    Examples:
        +12.5  -> p000012.500mm
        -12.5  -> m000012.500mm
    """
    sign = "m" if value_mm < 0 else "p"
    return f"{sign}{abs(value_mm):010.3f}mm"


def _format_vec3(v: Vec3D) -> str:
    return (
        f"x{_format_mm(v.x_mm)}-"
        f"y{_format_mm(v.y_mm)}-"
        f"z{_format_mm(v.z_mm)}"
    )


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _estimate_saturated_pixel_count(arr: np.ndarray) -> Optional[int]:
    """
    Conservative saturation estimate.

    For uint16 Mono16 containing 12-bit data, the saturation value might be 4095
    or it might be left-shifted depending on camera/output settings. This just
    counts pixels at the array max representable value when obvious.

    You can replace this later with a camera-specific threshold like 4095.
    """

    if not np.issubdtype(arr.dtype, np.integer):
        return None

    dtype_max = np.iinfo(arr.dtype).max

    if int(arr.max()) == dtype_max:
        return int(np.sum(arr == dtype_max))

    return 0


def _dataclass_to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if hasattr(obj, "__dataclass_fields__"):
        return {
            key: _dataclass_to_jsonable(value)
            for key, value in asdict(obj).items()
        }

    if isinstance(obj, dict):
        return {key: _dataclass_to_jsonable(value) for key, value in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_jsonable(value) for value in obj]

    return obj