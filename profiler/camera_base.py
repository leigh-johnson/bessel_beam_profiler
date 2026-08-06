from __future__ import annotations  # avoid evaluating PySpin type hints at import time

import PySpin
import gc
import logging
import sys
import traceback
import time

from camera_settings import FLIRCameraSettings

logger = logging.getLogger(__name__)

class FLIRCameraError(RuntimeError):
    pass

def _open_camera(
    camera_index: int,
    camera_serial: str | None = None,
) -> PySpin.Camera:
    """
    Open a camera by serial number (preferred: robust to bus enumeration
    order) or by enumeration index when no serial is given.
    """

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cam = None

    num_cameras = cam_list.GetSize()

    if num_cameras == 0:
        raise Exception("No FLIR cameras detected.")

    if camera_serial is not None:
        detected: list[str] = []
        for index in range(num_cameras):
            candidate = cam_list.GetByIndex(index)
            try:
                found = candidate.TLDevice.DeviceSerialNumber.GetValue()
            except Exception as ex:  # noqa: BLE001 - keep scanning the bus
                logger.warning(
                    f"Could not read serial of camera {index} while "
                    f"searching for {camera_serial!r}: {ex}"
                )
                detected.append("<unreadable>")
                continue
            detected.append(found)
            if found == camera_serial:
                cam = candidate
                camera_index = index
                break

        if cam is None:
            raise Exception(
                f"No camera with serial {camera_serial!r} found. "
                f"Detected {num_cameras} camera(s) with serial(s): "
                f"{', '.join(detected)}."
            )

        cam.Init()
        print(f"Opened camera serial {camera_serial} (index {camera_index})")
        return system, cam_list, cam

    if camera_index >= num_cameras:
        raise Exception(
            f"Requested camera index {camera_index}, "
            f"but only {num_cameras} camera(s) were detected."
        )

    cam = cam_list.GetByIndex(camera_index)
    cam.Init()
    print(f"Opened camera {camera_index}")

    return system, cam_list, cam


def _warn_cleanup_failure(step: str, ex: Exception) -> None:
    print(f"Warning: {step} during camera cleanup failed: {ex}")

class FLIRCameraControllerBase:
    """
    Base class for FLIR cameras. This class is not meant to be instantiated directly.
    It provides common functionality for all FLIR camera types.
    """

    def __init__(
        self,
        camera_index: int,
        camera_settings: FLIRCameraSettings,
        cam=None,
        camera_serial: str | None = None,
    ) -> None:
        if cam is not None:
            # Dependency injection for unit tests: use the provided camera
            # object and skip PySpin system/camera discovery entirely.
            self.system, self.cam_list, self.cam = None, None, cam
        else:
            # camera_serial (when given) takes precedence over camera_index.
            self.system, self.cam_list, self.cam = _open_camera(
                camera_index, camera_serial
            )

        self.camera_index = camera_index
        self.camera_settings = camera_settings

        # Remembered so reopen() can find the SAME camera again even if
        # enumeration order changes after a bus removal (-1024).
        self.camera_serial = camera_serial
        try:
            self.camera_serial = self.cam.TLDevice.DeviceSerialNumber.GetValue()
        except Exception as ex:  # noqa: BLE001 - serial is best-effort
            if self.camera_serial is not None:
                logger.warning(
                    f"Could not read camera serial number ({ex}); keeping "
                    f"the requested serial {self.camera_serial!r} for reopen()."
                )
            else:
                logger.warning(
                    f"Could not read camera serial number ({ex}); reopen() "
                    "will fall back to the camera index."
                )

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            logger.error(f"Exception during camera cleanup in __del__: {e}")

    def apply_settings(self) -> None:
        """
        Apply the camera settings to the camera.
        """
        self.camera_settings.apply(self.cam, strict=True)

    def _begin_acquisition(self) -> None:
        """
        Begin acquisition on the camera.
        """
        return self.cam.BeginAcquisition()
    
    def _end_acquisition(self) -> None:
        return self.cam.EndAcquisition()
    
    def _execute_software_trigger(self, delay_s: float = 2.0) -> None:
        nodemap = self.cam.GetNodeMap()
        command = PySpin.CCommandPtr(nodemap.GetNode("TriggerSoftware"))

        if not PySpin.IsWritable(command):
            raise FLIRCameraError("Unable to execute TriggerSoftware.")

        command.Execute()

    def reopen(self, retries: int = 5, delay_s: float = 3.0) -> None:
        """
        Recover from a bus removal (Spinnaker -1024 "Camera has been
        removed from the list"): release everything, re-enumerate, find
        the same camera (by serial, falling back to index), Init, and
        re-apply the camera settings. Raises FLIRCameraError if the
        camera does not come back within `retries` attempts.
        """

        import time

        if self.system is None and self.cam is not None:
            raise FLIRCameraError(
                "reopen() is not available for injected test cameras."
            )

        serial = getattr(self, "camera_serial", None)
        self.close()

        last_error: Exception | None = None

        for attempt in range(1, retries + 1):
            system = PySpin.System.GetInstance()
            cam_list = system.GetCameras()
            cam = None

            try:
                if serial:
                    for index in range(cam_list.GetSize()):
                        candidate = cam_list.GetByIndex(index)
                        try:
                            found = (
                                candidate.TLDevice.DeviceSerialNumber.GetValue()
                            )
                        except Exception as ex:  # noqa: BLE001
                            logger.warning(
                                f"Could not read serial of camera {index} "
                                f"while reconnecting: {ex}"
                            )
                            continue
                        if found == serial:
                            cam = candidate
                            break

                if cam is None and cam_list.GetSize() > self.camera_index:
                    cam = cam_list.GetByIndex(self.camera_index)

                if cam is None:
                    raise FLIRCameraError(
                        f"camera (serial {serial!r}) not present on the bus"
                    )

                cam.Init()

                self.system, self.cam_list, self.cam = system, cam_list, cam
                self.apply_settings()

                logger.warning(
                    f"Camera reconnected on attempt {attempt}/{retries} "
                    f"(serial {serial!r})."
                )
                return

            except Exception as ex:  # noqa: BLE001 - retry with cleanup
                last_error = ex
                logger.warning(
                    f"Camera reconnect attempt {attempt}/{retries} failed: {ex}"
                )

                # Release this attempt's handles before retrying.
                del cam
                try:
                    cam_list.Clear()
                except Exception as cleanup_ex:  # noqa: BLE001
                    logger.warning(f"CameraList.Clear during retry: {cleanup_ex}")
                try:
                    system.ReleaseInstance()
                except Exception as cleanup_ex:  # noqa: BLE001
                    logger.warning(f"ReleaseInstance during retry: {cleanup_ex}")

                time.sleep(delay_s)

        raise FLIRCameraError(
            f"Camera did not return after {retries} reconnect attempts "
            f"({delay_s:g}s apart): {last_error}"
        )

    def close(self) -> None:
        """
        End acquisition, de-initialize, and release the camera and system.

        Deliberately never raises: this runs in finally blocks and __del__,
        where a propagating Spinnaker exception can escape into C++ object
        destructors and abort the whole process (error -1004, "Can't clear a
        camera because something still holds a reference to the camera").
        """
        print("Cleaning up PySpin Camera and System instances...")

        cam = getattr(self, "cam", None)
        cam_list = getattr(self, "cam_list", None)
        system = getattr(self, "system", None)

        self.cam = None
        self.cam_list = None
        self.system = None

        if system is None:
            # Injected fake camera (unit tests): nothing to release.
            return

        # When close() runs inside a finally block while an exception is
        # propagating, the exception's traceback frames can pin camera
        # references — notably the PySpin GetNextImage wrapper frame, whose
        # `self` IS the camera. A pinned camera makes Clear/ReleaseInstance
        # below raise -1004. Clearing the locals of those (non-executing)
        # frames drops the pins; the traceback's file/line info is preserved,
        # and still-executing frames are skipped automatically.
        exc = sys.exc_info()[1]
        seen_exceptions = set()
        while exc is not None and id(exc) not in seen_exceptions:
            seen_exceptions.add(id(exc))
            if exc.__traceback__ is not None:
                traceback.clear_frames(exc.__traceback__)
            exc = exc.__cause__ or exc.__context__

        if cam is not None:
            try:
                if cam.IsStreaming():
                    cam.EndAcquisition()
            except Exception as ex:
                _warn_cleanup_failure("EndAcquisition", ex)

            try:
                cam.DeInit()
            except Exception as ex:
                _warn_cleanup_failure("DeInit", ex)

        # CameraList.Clear() raises -1004 if any Python object still
        # references a camera, so drop every reference we hold first.
        del cam
        gc.collect()

        try:
            if cam_list is not None:
                cam_list.Clear()
        except Exception as ex:
            _warn_cleanup_failure("CameraList.Clear", ex)

        del cam_list
        gc.collect()

        try:
            # Balance the GetInstance() from _open_camera so the system is
            # not torn down inside a C++ destructor at interpreter shutdown,
            # where a Spinnaker exception would abort the process.
            system.ReleaseInstance()
        except Exception as ex:
            _warn_cleanup_failure("System.ReleaseInstance", ex)
            try:
                # The failed release left the SWIG wrapper owning the system;
                # its C++ destructor would retry the release, and a Spinnaker
                # exception thrown inside a destructor aborts the whole
                # process (libc++abi terminate). Disown the wrapper and let
                # the OS reclaim the camera at process exit instead.
                system.thisown = False
            except Exception as e:
                logger.error(f"Disowning the PySpin system wrapper failed: {e}")