import importlib
import sys
import types

import pytest


MODULE_NAME = "camera_settings"

# Fake PySpin classes for testing without a physical camera or Spinnaker installed.
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


class FakeNodeMap:
    def __init__(self, nodes):
        self.nodes = nodes

    def GetNode(self, name):
        return self.nodes.get(name)


class FakeCamera:
    def __init__(self, model_name="BFS-PGE-31S4M"):
        # QuickSpin-style nodes
        self.PixelFormat = FakeNode()
        self.ExposureAuto = FakeNode()
        self.ExposureMode = FakeNode()
        self.ExposureTime = FakeNode(minimum=1.0, maximum=30_000_000.0)

        self.GainAuto = FakeNode()
        self.Gain = FakeNode(minimum=0.0, maximum=48.0)

        self.GammaEnable = FakeNode()
        self.Gamma = FakeNode(minimum=0.25, maximum=4.0)

        self.BlackLevelSelector = FakeNode()
        self.BlackLevel = FakeNode(minimum=0.0, maximum=100.0)

        self.BalanceWhiteAuto = FakeNode()
        self.BalanceRatioSelector = FakeNode()
        self.BalanceRatio = FakeNode(minimum=0.0, maximum=10.0)

        self._tl_device_node_map = FakeNodeMap(
            {
                "DeviceModelName": FakeNode(model_name),
            }
        )

        self._node_map = FakeNodeMap({})

    def GetTLDeviceNodeMap(self):
        return self._tl_device_node_map

    def GetNodeMap(self):
        return self._node_map


def make_fake_pyspin():
    fake = types.SimpleNamespace()

    class SpinnakerException(Exception):
        pass

    fake.SpinnakerException = SpinnakerException
    fake.Camera = object

    # Availability/readability/writability checks
    fake.IsAvailable = lambda node: node is not None and getattr(node, "available", True)
    fake.IsReadable = lambda node: node is not None and getattr(node, "readable", True)
    fake.IsWritable = lambda node: node is not None and getattr(node, "writable", True)

    # Pointer wrappers. For these tests, identity is enough.
    fake.CStringPtr = lambda node: node
    fake.CEnumerationPtr = lambda node: node
    fake.CFloatPtr = lambda node: node
    fake.CBooleanPtr = lambda node: node

    # Enum constants used by the QuickSpin path:
    # getattr(PySpin, f"{feature}_{entry_name}")
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

    return fake


@pytest.fixture()
def settings_module(monkeypatch):
    """
    Import flir_camera_settings.py with a fake PySpin module.

    This lets the tests run on a laptop/CI machine without Spinnaker installed
    and without a physical camera connected.
    """
    fake_pyspin = make_fake_pyspin()

    monkeypatch.setitem(sys.modules, "PySpin", fake_pyspin)
    sys.modules.pop(MODULE_NAME, None)

    module = importlib.import_module(MODULE_NAME)

    # Extra safety in case the module was already imported elsewhere.
    monkeypatch.setattr(module, "PySpin", fake_pyspin)

    return module


def test_json_round_trip(settings_module, tmp_path):
    FLIRCameraSettings = settings_module.FLIRCameraSettings

    settings = FLIRCameraSettings(
        CameraModel="BFS-PGE-31S4M",
        PixelFormat="Mono16",
        ExposureAuto="Off",
        ExposureMode="Timed",
        ExposureTime=500.0,
        GainAuto="Off",
        Gain=0.0,
        GammaEnable=False,
        Gamma=1.0,
        BlackLevelSelector="All",
        BlackLevel=None,
    )

    path = tmp_path / "camera_settings.json"
    settings.to_json_file(path)

    loaded = FLIRCameraSettings.from_json_file(path)

    assert loaded == settings


def test_invalid_exposure_auto_is_rejected(settings_module):
    FLIRCameraSettings = settings_module.FLIRCameraSettings

    with pytest.raises(ValueError, match="ExposureAuto"):
        FLIRCameraSettings(
            CameraModel="BFS-PGE-31S4M",
            ExposureAuto="DefinitelyNotValid",  # type: ignore[arg-type]
            ExposureTime=500.0,
        )


def test_invalid_pixel_format_is_rejected(settings_module):
    FLIRCameraSettings = settings_module.FLIRCameraSettings

    with pytest.raises(ValueError, match="PixelFormat"):
        FLIRCameraSettings(
            CameraModel="BFS-PGE-31S4M",
            PixelFormat="RGB8",  # type: ignore[arg-type]
            ExposureTime=500.0,
        )


def test_apply_manual_measurement_settings(settings_module):
    FLIRCameraSettings = settings_module.FLIRCameraSettings

    cam = FakeCamera(model_name="BFS-PGE-31S4M")

    settings = FLIRCameraSettings(
        CameraModel="BFS-PGE-31S4M",
        PixelFormat="Mono16",
        ExposureAuto="Off",
        ExposureMode="Timed",
        ExposureTime=500.0,
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

    warnings = settings.apply(cam, strict=True)

    assert warnings == []

    assert cam.PixelFormat.value == "PixelFormat_Mono16"

    assert cam.ExposureAuto.value == "ExposureAuto_Off"
    assert cam.ExposureMode.value == "ExposureMode_Timed"
    assert cam.ExposureTime.value == 500.0

    assert cam.GainAuto.value == "GainAuto_Off"
    assert cam.Gain.value == 0.0

    assert cam.GammaEnable.value is False

    # Gamma value should not be applied when GammaEnable=False.
    assert cam.Gamma.value is None


def test_apply_rejects_model_mismatch_in_strict_mode(settings_module):
    FLIRCameraSettings = settings_module.FLIRCameraSettings
    CameraSettingError = settings_module.CameraSettingError

    cam = FakeCamera(model_name="SomeOtherCamera")

    settings = FLIRCameraSettings(
        CameraModel="BFS-PGE-31S4M",
        PixelFormat="Mono16",
        ExposureAuto="Off",
        ExposureMode="Timed",
        ExposureTime=500.0,
    )

    with pytest.raises(CameraSettingError, match="Camera model mismatch"):
        settings.apply(cam, strict=True)


def test_apply_model_mismatch_warns_in_non_strict_mode(settings_module):
    FLIRCameraSettings = settings_module.FLIRCameraSettings

    cam = FakeCamera(model_name="SomeOtherCamera")

    settings = FLIRCameraSettings(
        CameraModel="BFS-PGE-31S4M",
        PixelFormat="Mono16",
        ExposureAuto="Off",
        ExposureMode="Timed",
        ExposureTime=500.0,
    )

    warnings = settings.apply(cam, strict=False)

    assert any("Camera model mismatch" in warning for warning in warnings)

    # Even with the warning, settings are still applied in non-strict mode.
    assert cam.PixelFormat.value == "PixelFormat_Mono16"
    assert cam.ExposureAuto.value == "ExposureAuto_Off"
    assert cam.ExposureMode.value == "ExposureMode_Timed"
    assert cam.ExposureTime.value == 500.0


def test_manual_exposure_requires_exposure_time(settings_module):
    FLIRCameraSettings = settings_module.FLIRCameraSettings
    CameraSettingError = settings_module.CameraSettingError

    cam = FakeCamera(model_name="BFS-PGE-31S4M")

    settings = FLIRCameraSettings(
        CameraModel="BFS-PGE-31S4M",
        PixelFormat="Mono16",
        ExposureAuto="Off",
        ExposureMode="Timed",
        ExposureTime=None,
    )

    with pytest.raises(CameraSettingError, match="ExposureTime is None"):
        settings.apply(cam, strict=True)


def test_auto_exposure_does_not_apply_exposure_time(settings_module):
    FLIRCameraSettings = settings_module.FLIRCameraSettings

    cam = FakeCamera(model_name="BFS-PGE-31S4M")

    settings = FLIRCameraSettings(
        CameraModel="BFS-PGE-31S4M",
        PixelFormat="Mono16",
        ExposureAuto="Continuous",
        ExposureMode="Timed",
        ExposureTime=500.0,
        GainAuto="Off",
        Gain=0.0,
    )

    warnings = settings.apply(cam, strict=True)

    assert cam.ExposureAuto.value == "ExposureAuto_Continuous"

    # ExposureTime should not be written when ExposureAuto is not Off.
    assert cam.ExposureTime.value is None

    assert any("ExposureTime was not applied" in warning for warning in warnings)


def test_gain_outside_camera_range_raises(settings_module):
    FLIRCameraSettings = settings_module.FLIRCameraSettings
    CameraSettingError = settings_module.CameraSettingError

    cam = FakeCamera(model_name="BFS-PGE-31S4M")

    settings = FLIRCameraSettings(
        CameraModel="BFS-PGE-31S4M",
        PixelFormat="Mono16",
        ExposureAuto="Off",
        ExposureMode="Timed",
        ExposureTime=500.0,
        GainAuto="Off",
        Gain=100.0,
    )

    with pytest.raises(CameraSettingError, match="Gain=100.0"):
        settings.apply(cam, strict=True)


def test_gamma_value_is_applied_only_when_enabled(settings_module):
    FLIRCameraSettings = settings_module.FLIRCameraSettings

    cam = FakeCamera(model_name="BFS-PGE-31S4M")

    settings = FLIRCameraSettings(
        CameraModel="BFS-PGE-31S4M",
        PixelFormat="Mono16",
        ExposureAuto="Off",
        ExposureMode="Timed",
        ExposureTime=500.0,
        GainAuto="Off",
        Gain=0.0,
        GammaEnable=True,
        Gamma=1.5,
    )

    warnings = settings.apply(cam, strict=True)

    assert warnings == []
    assert cam.GammaEnable.value is True
    assert cam.Gamma.value == 1.5