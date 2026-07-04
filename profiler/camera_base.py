import PySpin
import gc

from camera_settings import FLIRCameraSettings

class FLIRCameraError(RuntimeError):
    pass

def _open_camera(camera_index: int) -> PySpin.Camera:

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cam = None

    num_cameras = cam_list.GetSize()

    if num_cameras == 0:
        raise Exception("No FLIR cameras detected.")

    if camera_index >= num_cameras:
        raise Exception(
            f"Requested camera index {camera_index}, "
            f"but only {num_cameras} camera(s) were detected."
        )

    cam = cam_list.GetByIndex(camera_index)
    cam.Init()
    print(f"Opened camera {camera_index}")

    return system, cam_list, cam


def _close_camera(system: PySpin.System, cam_list: PySpin.CameraList, cam: PySpin.Camera) -> None:
    """
    De-initialize and release a PySpin camera instance.

    This is a separate function so that we can call it in a finally block
    without having to check if cam is None.
    """
    cam_list = system.GetCameras()
    cam.DeInit()
    cam_list.Clear()

class FLIRCameraControllerBase:
    """
    Base class for FLIR cameras. This class is not meant to be instantiated directly.
    It provides common functionality for all FLIR camera types.
    """

    def __init__(self, camera_index: int, camera_settings: FLIRCameraSettings) -> None:
        self.system, self.cam_list, self.cam = _open_camera(camera_index)
        self.camera_settings = camera_settings

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

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
    
    def _execute_software_trigger(self) -> None:
        nodemap = self.cam.GetNodeMap()
        command = PySpin.CCommandPtr(nodemap.GetNode("TriggerSoftware"))

        if not PySpin.IsWritable(command):
            raise FLIRCameraError("Unable to execute TriggerSoftware.")

        command.Execute()

    def close(self) -> None:
        print("Cleaning up PySpin Camera and System instances...")

        cam = getattr(self, "cam", None)
        cam_list = getattr(self, "cam_list", None)
        system = getattr(self, "system", None)

        self.cam = None
        self.cam_list = None
        self.system = None

        _close_camera(system, cam_list, cam)
        gc.collect()