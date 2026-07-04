from __future__ import annotations  # convert type hints to strings at runtime

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional
import datetime as dt
import json
import re
import threading
import time
import uuid

import numpy as np
import PySpin

from camera_settings import FLIRCameraSettings
from coordinates import ScanPoint, Vec3D

from camera_base import FLIRCameraControllerBase

system = PySpin.System.GetInstance()

class DatasetWriterError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetWriterConfig:
    DatasetRoot: Path
    AcquisitionTimeout_ms: int = 2000
    StageTimeout_s: float = 30.0
    SettleTime_s: float = 0.0

    # Save uint8 / uint16 arrays in binary numpy format.
    # We can encode to BMP or PNG later, but want to avoid any lossy compression at this stage.
    ImageExtension: str = ".npy"

    # Give each run a unique ID so that multiple runs on the same day don't collide.
    RunUUID: str = field(default_factory=lambda: uuid.uuid4().hex)

    def make_run_dir(self) -> Path:
        today = dt.date.today().isoformat()
        return self.DatasetRoot / f"{today}-{self.RunUUID}"


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


# TODO: wrap calls to FluidNC, which will be running on an ESP32 microcontroller
# to drive stepper motors.
# http://wiki.fluidnc.com/en/home
# class FluidNCStageController(StageController):
#     def move_to_scan_point(self, point, signals):
#         xyz = point.GantryPosition_mm
#         # TODO send G-code / FluidNC command using xyz.x_mm, xyz.y_mm, xyz.z_mm
#         signals.MovementStarted.set()

#     def wait_until_motion_complete(self, point, timeout_s, signals):
#         # TODO poll FluidNC until idle
#         signals.MovementComplete.set()


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
    ):
        self.camera_index = camera_index
        self.config = config
        self.stage_controller = stage_controller or StageController()
        self.signals = signals or AcquisitionSignals()

        self.run_dir = self.config.make_run_dir()
        self.manifest_path = self.run_dir / "frames.jsonl"

        super().__init__(camera_index, camera_settings)


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

        try:
            for shot_idx in range(point.NShots):
                if self.signals.StopRequested.is_set():
                    break

                record = self._acquire_one_frame(
                    point,
                    shot_idx,
                )
                self._append_manifest(record)
                records.append(record)

        finally:
            self._end_acquisition()

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
            self._execute_software_trigger()

            image_result = self.cam.GetNextImage(self.config.AcquisitionTimeout_ms)

            if image_result.IsIncomplete():
                status = image_result.GetImageStatus()
                raise DatasetWriterError(f"Image incomplete; image status = {status}")

            image_data = image_result.GetNDArray()
            self.signals.FrameBuffered.set()

            # Force a detached NumPy copy before releasing the camera image pointer.
            arr = np.array(image_data, copy=True)

            frame_path = self._frame_path(point, shot_idx)
            self._write_array(frame_path, arr)
            frame_path_jpg = str(frame_path.with_suffix(".jpg"))
            image_result.Save(frame_path_jpg)  # Save a PNG copy for quick viewing
            self.signals.FrameWritten.set()

            saturated_count = _estimate_saturated_pixel_count(arr)

            extra = dict(point.Metadata)

            if hasattr(image_result, "GetFrameID"):
                extra["FrameID"] = int(image_result.GetFrameID())

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
                SaturatedPixelCount=saturated_count,
                Extra=extra,
            )

        except PySpin.SpinnakerException as ex:
            raise DatasetWriterError(f"Spinnaker acquisition failed: {ex}") from ex

        finally:
            if image_result is not None:
                image_result.Release()

    def _frame_path(self, point: ScanPoint, shot_idx: int) -> Path:
        filename = (
            f"{_format_placement_id(point.PlacementID)}-"
            f"tablez{_format_mm(point.TablePosition_mm.z_mm)}-"
            f"gantry{_format_vec3(point.GantryPosition_mm)}-"
            f"shot{shot_idx:04d}"
            f"{self.config.ImageExtension}"
        )

        return self.run_dir / filename

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
        with self.manifest_path.open("a") as f:
            f.write(json.dumps(_dataclass_to_jsonable(record)) + "\n")



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