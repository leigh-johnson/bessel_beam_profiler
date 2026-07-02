from __future__ import annotations # convert type hints to strings at runtime

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol
import datetime as dt
import json
import threading
import time
import uuid

import numpy as np
import PySpin

from camera_settings import FLIRCameraSettings
from coordinates import XY, ScanPoint, AcquisitionSignals

class DatasetWriterError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetWriterConfig:
    DatasetRoot: Path
    AcquisitionTimeout_ms: int = 2000
    StageTimeout_s: float = 30.0
    SettleTime_s: float = 0.0

    # This is the Spinnaker GenAPI AcquisitionMode enum name. 
    # The FLIR examples use "Continuous" for live display and "SingleFrame" for single-shot acquisition.
    AcquisitionMode: str = "SingleFrame"

    # Save uint8 / uint16 arrays in binary numpy format. We can encode to BMP or PNG, but want to avoid any lossy compression at this stage.
    ImageExtension: str = ".npy"

    # Give each run a unique ID so that multiple runs on the same day don't collide.
    RunUUID: str = field(default_factory=lambda: uuid.uuid4().hex)

    def make_run_dir(self) -> Path:
        today = dt.date.today().isoformat()
        return self.DatasetRoot / f"{today}-{self.RunUUID}"


@dataclass
class AcquisitionSignals:
    """
    threading.Event objects used to coordinate the acquisition order between the dataset writer and a stepper-motor controller.

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
    # add a "FrameAcquired" signal that is set after the writer has acquired the frame but before it writes to disk. 
    # This would allow a future stepper-motor controller to start moving while the writer is still writing the frame to disk.

    def reset_for_position(self) -> None:
        self.MovementStarted.clear()
        self.MovementComplete.clear()
        self.AcquisitionStarted.clear()
        self.FrameBuffered.clear()
        self.FrameWritten.clear()


class StageController(Protocol):
    """
    Interface for future stepper-motor integration.

    The dataset writer assumes:

        move_to_scan_point(...) starts or performs the motion
        wait_until_motion_complete(...) blocks until the stage is actually still

    TODO: wrap calls to FluidNC, which will be running on an ESP32 microcontroller to drive stepper motors.
    http://wiki.fluidnc.com/en/home
    """

    def move_to_scan_point(
        self,
        point: ScanPoint,
        signals: AcquisitionSignals,
    ) -> None:
        pass

    def wait_until_motion_complete(
        self,
        point: ScanPoint,
        timeout_s: float,
        signals: AcquisitionSignals,
    ) -> None:
        pass


@dataclass(frozen=True)
class FrameRecord:
    Path: str
    ZPosition_mm: float
    TopLeftXY: XY
    BotRightXY: XY
    ShotIndex: int
    TimestampUTC: str
    Shape: tuple[int, int]
    DType: str
    Min: int
    Max: int
    SaturatedPixelCount: Optional[int]
    Extra: dict[str, Any] = field(default_factory=dict)


class FLIRDatasetWriter:
    def __init__(
        self,
        cam: PySpin.Camera,
        camera_settings: FLIRCameraSettings,
        config: DatasetWriterConfig,
        stage_controller: Optional[StageController] = None,
        signals: Optional[AcquisitionSignals] = None,
    ):
        self.cam = cam
        self.camera_settings = camera_settings
        self.config = config
        self.stage_controller = stage_controller or StageController()
        self.signals = signals or AcquisitionSignals()

        self.run_dir = self.config.make_run_dir()
        self.manifest_path = self.run_dir / "frames.jsonl"

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
            "Config": {
                **asdict(self.config),
                "DatasetRoot": str(self.config.DatasetRoot),
            },
        }

        (self.run_dir / "run_metadata.json").write_text(
            json.dumps(run_metadata, indent=2) + "\n"
        )

        return self.run_dir

    def acquire_scan(self, points: Iterable[ScanPoint]) -> list[FrameRecord]:
        """
        Acquire the full z-scan.

        For each ScanPoint:

            1. command/wait for stage motion
            2. require MovementComplete signal
            3. wait optional settling time
            4. acquire images
            5. write arrays to disk
        """

        records: list[FrameRecord] = []

        self._set_acquisition_mode(self.config.AcquisitionMode)

        for point in points:
            if self.signals.StopRequested.is_set():
                break

            self._move_and_wait(point)

            if self.config.SettleTime_s > 0:
                time.sleep(self.config.SettleTime_s)

            for shot_idx in range(point.NShots):
                if self.signals.StopRequested.is_set():
                    break

                record = self._acquire_one_frame(point, shot_idx)
                self._append_manifest(record)
                records.append(record)

        return records

    def _move_and_wait(self, point: ScanPoint) -> None:
        self.signals.reset_for_position()

        # Hook for future stepper implementation.
        self.stage_controller.move_to_scan_point(point, self.signals)

        self.stage_controller.wait_until_motion_complete(
            point=point,
            timeout_s=self.config.StageTimeout_s,
            signals=self.signals,
        )

        ok = self.signals.MovementComplete.wait(timeout=self.config.StageTimeout_s)

        if not ok:
            raise DatasetWriterError(
                f"Timed out waiting for stage motion complete at "
                f"ZPosition_mm={point.ZPosition_mm}."
            )

    def _acquire_one_frame(self, point: ScanPoint, shot_idx: int) -> FrameRecord:
        """
        Acquire exactly one frame after MovementComplete has fired.
        """

        if not self.signals.MovementComplete.is_set():
            raise DatasetWriterError(
                "Refusing to acquire: MovementComplete signal is not set."
            )

        self.signals.AcquisitionStarted.set()

        image_result = None

        try:
            # This is intentionally after MovementComplete.
            self.cam.BeginAcquisition()

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
            self.signals.FrameWritten.set()

            saturated_count = _estimate_saturated_pixel_count(arr)

            return FrameRecord(
                Path=str(frame_path),
                ZPosition_mm=point.ZPosition_mm,
                TopLeftXY=point.TopLeftXY,
                BotRightXY=point.BotRightXY,
                ShotIndex=shot_idx,
                TimestampUTC=_utc_now(),
                Shape=tuple(arr.shape),
                DType=str(arr.dtype),
                Min=int(np.min(arr)),
                Max=int(np.max(arr)),
                SaturatedPixelCount=saturated_count,
                Extra=point.Metadata,
            )

        except PySpin.SpinnakerException as ex:
            raise DatasetWriterError(f"Spinnaker acquisition failed: {ex}") from ex

        finally:
            if image_result is not None:
                image_result.Release()

            # End acquisition after every frame to avoid free-running capture
            # during future stage movement.
            try:
                self.cam.EndAcquisition()
            except PySpin.SpinnakerException:
                pass

    def _frame_path(self, point: ScanPoint, shot_idx: int) -> Path:
        filename = (
            f"{_format_z(point.ZPosition_mm)}-"
            f"topleft{_format_xy(point.TopLeftXY)}-"
            f"botright{_format_xy(point.BotRightXY)}-"
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

    def _set_acquisition_mode(self, mode: str) -> None:
        """
        Set AcquisitionMode using GenAPI.

        The FLIR examples set AcquisitionMode by getting the enum node and then
        selecting an entry like Continuous. We use the same pattern here.
        """

        nodemap = self.cam.GetNodeMap()
        node_acquisition_mode = PySpin.CEnumerationPtr(
            nodemap.GetNode("AcquisitionMode")
        )

        if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(
            node_acquisition_mode
        ):
            raise DatasetWriterError("Unable to access AcquisitionMode.")

        entry = node_acquisition_mode.GetEntryByName(mode)

        if not PySpin.IsReadable(entry):
            raise DatasetWriterError(f"AcquisitionMode entry {mode!r} is not readable.")

        node_acquisition_mode.SetIntValue(entry.GetValue())


def _format_z(z_mm: float) -> str:
    return f"z{z_mm:010.3f}mm"


def _format_xy(xy: XY) -> str:
    return f"{xy.x:05d}_{xy.y:05d}"


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