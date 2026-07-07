import importlib
import json
import sys
import types

import numpy as np
import pytest


MODULE_NAME = "calibration"


class FakeNode:
    def __init__(
        self,
        value=None,
        *,
        minimum=float("-inf"),
        maximum=float("inf"),
        available=True,
        readable=True,
        writable=True,
    ):
        self.value = value
        self.minimum = minimum
        self.maximum = maximum
        self.available = available
        self.readable = readable
        self.writable = writable
        self.set_calls = []

    def SetValue(self, value):
        self.value = value
        self.set_calls.append(value)

    def GetValue(self):
        return self.value

    def GetMin(self):
        return self.minimum

    def GetMax(self):
        return self.maximum


class FakeEnumEntry:
    def __init__(self, name, value=1, readable=True):
        self.name = name
        self.value = value
        self.readable = readable
        self.available = True

    def GetValue(self):
        return self.value

    def GetSymbolic(self):
        return self.name


class FakeEnumNode:
    def __init__(self, readable=True, writable=True):
        self.readable = readable
        self.writable = writable
        self.available = True
        self.entries_requested = []
        self.set_values = []

    def GetEntryByName(self, name):
        self.entries_requested.append(name)
        return FakeEnumEntry(name, value=len(self.entries_requested))

    def SetValue(self, value):
        self.set_values.append(value)

    def SetIntValue(self, value):
        self.set_values.append(value)

    def GetCurrentEntry(self):
        return FakeEnumEntry("Current")


class FakeNodeMap:
    def __init__(self, nodes):
        self.nodes = nodes

    def GetNode(self, name):
        return self.nodes.get(name)


class FakeImage:
    def __init__(self, arr, incomplete=False, status=0):
        self.arr = np.array(arr, copy=True)
        self.incomplete = incomplete
        self.status = status
        self.released = False

    def IsIncomplete(self):
        return self.incomplete

    def GetImageStatus(self):
        return self.status

    def GetNDArray(self):
        return self.arr

    def Release(self):
        self.released = True


class FakeCamera:
    def __init__(self, images):
        self.images = list(images)
        self.events = []

        # QuickSpin-style nodes used by real FLIRCameraSettings.apply(...)
        self.PixelFormat = FakeNode()
        self.ExposureAuto = FakeNode()
        self.ExposureMode = FakeNode()
        self.ExposureTime = FakeNode(minimum=25.0, maximum=1_000_000.0)

        self.AcquisitionFrameRatePersistence = FakeNode(True)
        self.AcquisitionMode = FakeEnumNode()
        self.AcquisitionFrameRateEnable = FakeNode(True)
        self.AcquisitionFrameRate = FakeNode(1.0, minimum=0.0, maximum=1_000_000.0)

        self.GainAuto = FakeNode()
        self.Gain = FakeNode(minimum=0.0, maximum=48.0)

        self.GammaEnable = FakeNode()
        self.Gamma = FakeNode(minimum=0.25, maximum=4.0)

        self.BlackLevelClampingEnable = FakeNode(False)

        self.BlackLevelSelector = FakeNode()
        self.BlackLevel = FakeNode(minimum=0.0, maximum=100.0)

        self.BalanceWhiteAuto = FakeNode()
        self.BalanceRatioSelector = FakeNode()
        self.BalanceRatio = FakeNode(minimum=0.0, maximum=10.0)
        self.GevSCPSPacketSize = FakeNode(1500, minimum=0, maximum=10_000_000)
        self.DeviceLinkThroughputLimit = FakeNode(
            10_000_000,
            minimum=0,
            maximum=100_000_000,
        )
        self.TriggerMode = FakeEnumNode()
        self.TriggerSource = FakeEnumNode()

        self.AcquisitionMode = FakeEnumNode()
        self.stream_buffer_mode = FakeEnumNode()
        self.stream_buffer_count_mode = FakeEnumNode()
        self.stream_buffer_count_manual = FakeNode(10, minimum=1, maximum=10_000)

        self.node_map = FakeNodeMap(
            {
                "AcquisitionMode": self.AcquisitionMode,
            }
        )
        self.stream_node_map = FakeNodeMap(
            {
                "StreamBufferHandlingMode": self.stream_buffer_mode,
                "StreamBufferCountMode": self.stream_buffer_count_mode,
                "StreamBufferCountManual": self.stream_buffer_count_manual,
            }
        )
        self.tl_device_node_map = FakeNodeMap(
            {
                "DeviceModelName": FakeNode("BFS-PGE-31S4M"),
            }
        )

    def GetNodeMap(self):
        return self.node_map

    def GetTLStreamNodeMap(self):
        return self.stream_node_map

    def GetTLDeviceNodeMap(self):
        return self.tl_device_node_map

    def BeginAcquisition(self):
        self.events.append("begin")

    def GetNextImage(self, timeout_ms):
        self.events.append(f"get:{timeout_ms}")
        if not self.images:
            raise RuntimeError("No fake images left")
        return self.images.pop(0)

    def EndAcquisition(self):
        self.events.append("end")


class FakeCameraController:
    def __init__(self, camera_index, camera_settings):
        self.cam = camera_index
        self.camera_settings = camera_settings

    def apply_settings(self):
        self.camera_settings.apply(self.cam, strict=True)

    def _begin_acquisition(self):
        self.cam.BeginAcquisition()

    def _end_acquisition(self):
        self.cam.EndAcquisition()

    def _execute_software_trigger(self):
        pass


class FakeImageArtist:
    def set_data(self, arr):
        pass


class FakeAxis:
    def __init__(self):
        self.titles = []
        self.imshow_calls = []
        self.artist = FakeImageArtist()

    def imshow(self, arr, **kwargs):
        self.imshow_calls.append((np.array(arr, copy=True), kwargs))
        return self.artist

    def set_title(self, title):
        self.titles.append(title)


class FakeCanvas:
    def __init__(self):
        self.callbacks = {}
        self.manager = types.SimpleNamespace(set_window_title=lambda title: None)

    def mpl_connect(self, event_name, callback):
        self.callbacks[event_name] = callback
        return 1

    def draw_idle(self):
        pass


class FakeFigure:
    def __init__(self):
        self.canvas = FakeCanvas()


def make_fake_pyspin():
    fake = types.SimpleNamespace()

    class SpinnakerException(Exception):
        pass

    class FakeCameraList:
        def __init__(self, cameras=None):
            self.cameras = list(cameras or [])

        def GetSize(self):
            return len(self.cameras)

        def GetByIndex(self, index):
            return self.cameras[index]

        def Clear(self):
            self.cameras.clear()

    class FakeSystem:
        def __init__(self, cameras=None):
            self.cameras = FakeCameraList(cameras)

        def GetCameras(self):
            return self.cameras

        def ReleaseInstance(self):
            pass

    fake.FakeCameraList = FakeCameraList
    fake.FakeSystem = FakeSystem
    fake.SpinnakerException = SpinnakerException
    fake.Camera = object
    fake.System = types.SimpleNamespace(GetInstance=lambda: FakeSystem())
    fake.CameraList = object

    fake.IsAvailable = lambda node: node is not None and getattr(node, "available", True)
    fake.IsReadable = lambda node: node is not None and getattr(node, "readable", True)
    fake.IsWritable = lambda node: node is not None and getattr(node, "writable", True)

    fake.CEnumerationPtr = lambda node: node
    fake.CStringPtr = lambda node: node
    fake.CFloatPtr = lambda node: node
    fake.CBooleanPtr = lambda node: node
    fake.CIntegerPtr = lambda node: node

    # QuickSpin enum constants used by FLIRCameraSettings.apply(...)
    fake.PixelFormat_Mono8 = "PixelFormat_Mono8"
    fake.PixelFormat_Mono12Packed = "PixelFormat_Mono12Packed"
    fake.PixelFormat_Mono16 = "PixelFormat_Mono16"

    fake.ExposureAuto_Off = "ExposureAuto_Off"
    fake.ExposureAuto_Once = "ExposureAuto_Once"
    fake.ExposureAuto_Continuous = "ExposureAuto_Continuous"

    fake.ExposureMode_Timed = "ExposureMode_Timed"
    fake.ExposureMode_TriggerWidth = "ExposureMode_TriggerWidth"

    fake.GainAuto_Off = "GainAuto_Off"
    fake.GainAuto_Once = "GainAuto_Once"
    fake.GainAuto_Continuous = "GainAuto_Continuous"

    fake.BlackLevelSelector_All = "BlackLevelSelector_All"

    fake.BalanceWhiteAuto_Off = "BalanceWhiteAuto_Off"
    fake.BalanceWhiteAuto_Once = "BalanceWhiteAuto_Once"
    fake.BalanceWhiteAuto_Continuous = "BalanceWhiteAuto_Continuous"

    fake.BalanceRatioSelector_Blue = "BalanceRatioSelector_Blue"
    fake.BalanceRatioSelector_Red = "BalanceRatioSelector_Red"

    fake.TriggerMode_Off = "TriggerMode_Off"
    fake.TriggerMode_On = "TriggerMode_On"
    fake.TriggerSource_Software = "TriggerSource_Software"

    return fake


def make_fake_matplotlib():
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_pyplot = types.ModuleType("matplotlib.pyplot")

    def fake_subplots():
        return FakeFigure(), FakeAxis()

    fake_pyplot.subplots = fake_subplots
    fake_pyplot.pause = lambda _dt: None
    fake_pyplot.close = lambda _fig: None
    fake_matplotlib.pyplot = fake_pyplot

    return fake_matplotlib, fake_pyplot


@pytest.fixture(autouse=True)
def fake_external_dependencies(monkeypatch):
    fake_matplotlib, fake_pyplot = make_fake_matplotlib()

    monkeypatch.setitem(sys.modules, "PySpin", make_fake_pyspin())
    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_pyplot)


@pytest.fixture()
def modules(monkeypatch):
    """
    Use real camera_settings.FLIRCameraSettings, but fake PySpin before import.
    """
    monkeypatch.setitem(sys.modules, "PySpin", make_fake_pyspin())

    sys.modules.pop("camera_settings", None)
    sys.modules.pop(MODULE_NAME, None)

    camera_settings = importlib.import_module("camera_settings")
    calibration = importlib.import_module(MODULE_NAME)
    monkeypatch.setattr(calibration, "FLIRCameraControllerBase", FakeCameraController)

    return calibration, camera_settings


def install_fake_matplotlib(monkeypatch, calibration_module, keys_by_pause):
    fig = FakeFigure()
    ax = FakeAxis()
    pause_count = {"n": 0}

    def fake_subplots():
        return fig, ax

    def fake_pause(_dt):
        pause_count["n"] += 1
        key = keys_by_pause.get(pause_count["n"])
        if key is not None:
            callback = fig.canvas.callbacks["key_press_event"]
            callback(types.SimpleNamespace(key=key))

    monkeypatch.setattr(calibration_module.plt, "subplots", fake_subplots)
    monkeypatch.setattr(calibration_module.plt, "pause", fake_pause)
    monkeypatch.setattr(calibration_module.plt, "close", lambda fig: None)

    return fig, ax


def make_base_settings(camera_settings_module, exposure_us=1000.0):
    return camera_settings_module.FLIRCameraSettings(
        CameraModel="BFS-PGE-31S4M",
        PixelFormat="Mono16",
        ExposureAuto="Off",
        ExposureMode="Timed",
        ExposureTime=exposure_us,
        GainAuto="Off",
        Gain=0.0,
        GammaEnable=False,
        Gamma=1.0,
        BlackLevelSelector="All",
        BlackLevel=None,
        BalanceWhiteAuto=None,
        BalanceRatioBlue=None,
        BalanceRatioRed=None,
    )


def test_calibration_uses_real_flir_camera_settings_and_writes_json(
    modules,
    monkeypatch,
    tmp_path,
):
    calibration, camera_settings = modules

    install_fake_matplotlib(
        monkeypatch,
        calibration,
        keys_by_pause={1: "q"},
    )

    cam = FakeCamera(
        images=[
            FakeImage(np.array([[1, 2], [3, 4]], dtype=np.uint16)),
        ]
    )

    output_path = tmp_path / "camera_settings.json"

    result = calibration.calibrate_exposure_interactive(
        cam,
        make_base_settings(camera_settings, exposure_us=1000.0),
        output_json_path=output_path,
        config=calibration.ExposureCalibrationConfig(
            AcquisitionTimeout_ms=123,
            DisplayPause_s=0,
        ),
    )

    assert isinstance(result.Settings, camera_settings.FLIRCameraSettings)

    assert result.Settings.PixelFormat == "Mono8"
    assert result.Settings.ExposureAuto == "Off"
    assert result.Settings.ExposureMode == "Timed"
    assert result.Settings.ExposureTime == 1000.0
    assert result.Settings.GainAuto == "Off"
    assert result.Settings.Gain == 0.0
    assert result.Settings.GammaEnable is False

    # Verify real FLIRCameraSettings.apply(...) touched the fake QuickSpin nodes.
    assert cam.PixelFormat.value == "PixelFormat_Mono8"
    assert cam.ExposureAuto.value == "ExposureAuto_Off"
    assert cam.ExposureMode.value == "ExposureMode_Timed"
    assert cam.ExposureTime.set_calls == [1000.0]
    assert cam.GainAuto.value == "GainAuto_Off"
    assert cam.Gain.value == 0.0
    assert cam.GammaEnable.value is False

    assert output_path.exists()
    saved = json.loads(output_path.read_text())
    assert saved["CameraModel"] == "BFS-PGE-31S4M"
    assert saved["PixelFormat"] == "Mono8"
    assert saved["ExposureAuto"] == "Off"
    assert saved["ExposureMode"] == "Timed"
    assert saved["ExposureTime"] == 1000.0
    assert saved["GainAuto"] == "Off"
    assert saved["Gain"] == 0.0
    assert saved["GammaEnable"] is False

    assert result.FinalExposure_us == 1000.0
    assert result.LastMax == 4
    assert result.LastSaturatedPixels == 0

    assert cam.AcquisitionMode.entries_requested == ["Continuous"]
    assert cam.stream_buffer_mode.entries_requested == ["NewestOnly"]
    assert cam.events == ["begin", "get:123", "end"]


def test_auto_reduce_updates_real_flir_camera_settings_result(
    modules,
    monkeypatch,
):
    calibration, camera_settings = modules

    install_fake_matplotlib(
        monkeypatch,
        calibration,
        keys_by_pause={
            1: "a",
            3: "q",
        },
    )

    images = [
        FakeImage(np.array([[4095]], dtype=np.uint16)),
        FakeImage(np.array([[4095]], dtype=np.uint16)),
        FakeImage(np.array([[100]], dtype=np.uint16)),
    ]

    cam = FakeCamera(images=images)

    result = calibration.calibrate_exposure_interactive(
        cam,
        make_base_settings(camera_settings, exposure_us=1000.0),
        config=calibration.ExposureCalibrationConfig(
            ReductionFactor=0.5,
            AcquisitionTimeout_ms=50,
            DisplayPause_s=0,
        ),
    )

    assert isinstance(result.Settings, camera_settings.FLIRCameraSettings)
    assert result.FinalExposure_us == 500.0
    assert result.Settings.ExposureTime == 500.0
    assert result.LastMax == 100
    assert result.LastSaturatedPixels == 0

    # First call is from FLIRCameraSettings.apply(...), second is auto-reduction.
    assert cam.ExposureTime.set_calls == [1000.0, 500.0]

    assert all(img.released for img in images)
    assert cam.events == ["begin", "get:50", "get:50", "get:50", "end"]


def test_plus_and_minus_keys_update_real_settings_result(
    modules,
    monkeypatch,
):
    calibration, camera_settings = modules

    install_fake_matplotlib(
        monkeypatch,
        calibration,
        keys_by_pause={
            1: "+",
            2: "-",
            3: "q",
        },
    )

    cam = FakeCamera(
        images=[
            FakeImage(np.array([[10]], dtype=np.uint16)),
            FakeImage(np.array([[10]], dtype=np.uint16)),
            FakeImage(np.array([[10]], dtype=np.uint16)),
        ]
    )

    result = calibration.calibrate_exposure_interactive(
        cam,
        make_base_settings(camera_settings, exposure_us=1000.0),
        config=calibration.ExposureCalibrationConfig(
            IncreaseFactor=2.0,
            ReductionFactor=0.25,
            AcquisitionTimeout_ms=10,
            DisplayPause_s=0,
        ),
    )

    assert isinstance(result.Settings, camera_settings.FLIRCameraSettings)
    assert result.FinalExposure_us == 500.0
    assert result.Settings.ExposureTime == 500.0

    # Initial apply, then '+', then '-'
    assert cam.ExposureTime.set_calls == [1000.0, 2000.0, 500.0]

def test_exposure_calibration_config_saturation_threshold():
    from calibration import ExposureCalibrationConfig

    config = ExposureCalibrationConfig(PixelFormat="Mono8")
    assert config.SaturationThreshold == 255

    config = ExposureCalibrationConfig(PixelFormat="Mono10")
    assert config.SaturationThreshold == 1023

    config = ExposureCalibrationConfig(PixelFormat="Mono12")
    assert config.SaturationThreshold == 4095

    config = ExposureCalibrationConfig(PixelFormat="Mono16")
    assert config.SaturationThreshold == 65535

    with pytest.raises(ValueError):
        config = ExposureCalibrationConfig(PixelFormat="Mono12Packed")
        _ = config.SaturationThreshold

def test_image_is_overexposed(monkeypatch):
    from calibration import ExposureCalibrationConfig, image_is_overexposed

    config = ExposureCalibrationConfig(
        PixelFormat="Mono8",
        AllowedSaturatedPixels=0,
    )

    # Image with no saturated pixels
    arr1 = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    is_overexposed, max_value, saturated_pixels = image_is_overexposed(arr1, config)
    assert not is_overexposed
    assert max_value == 3
    assert saturated_pixels == 0

    # Image with some saturated pixels
    arr2 = np.array([[0, 255], [255, 3]], dtype=np.uint8)
    is_overexposed, max_value, saturated_pixels = image_is_overexposed(arr2, config)
    assert is_overexposed
    assert max_value == 255
    assert saturated_pixels == 2