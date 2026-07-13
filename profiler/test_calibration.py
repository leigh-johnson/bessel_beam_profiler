import importlib
import json
import sys
import types

import numpy as np
import pytest


from conftest import FakeCamera, FakeImage, make_fake_pyspin


MODULE_NAME = "calibration"


class FakeCameraController:
    def __init__(self, camera_index, camera_settings):
        self.cam = camera_index
        self.camera_settings = camera_settings
        self.closed = False

    def apply_settings(self):
        self.camera_settings.apply(self.cam, strict=True)

    def _begin_acquisition(self):
        self.cam.BeginAcquisition()

    def _end_acquisition(self):
        self.cam.EndAcquisition()

    def _execute_software_trigger(self):
        pass

    def close(self):
        self.closed = True


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
        self.manager = types.SimpleNamespace(
            set_window_title=lambda title: None,
            key_press_handler_id=1,
        )
        self.disconnected = []

    def mpl_connect(self, event_name, callback):
        self.callbacks[event_name] = callback
        return 1

    def mpl_disconnect(self, cid):
        self.disconnected.append(cid)

    def draw_idle(self):
        pass

    def flush_events(self):
        pass


class FakeFigure:
    def __init__(self):
        self.canvas = FakeCanvas()
        self.savefig_calls = []

    def savefig(self, path, **kwargs):
        self.savefig_calls.append(path)


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
    # Timeout is AcquisitionTimeout_ms plus the exposure time in ms: 123 + 1.
    assert cam.events == ["begin", "get:124", "end"]


def test_auto_reduce_updates_real_flir_camera_settings_result(
    modules,
    monkeypatch,
    tmp_path,
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
        output_json_path=tmp_path / "camera_settings.json",
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
    # Timeout is AcquisitionTimeout_ms plus the exposure time in ms:
    # 50 + 1 at 1000 us, then 50 + 0.5 (truncated) after the reduction to 500 us.
    assert cam.events == ["begin", "get:51", "get:51", "get:50", "end"]


def test_plus_and_minus_keys_update_real_settings_result(
    modules,
    monkeypatch,
    tmp_path,
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
        output_json_path=tmp_path / "camera_settings.json",
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

    # Threshold is SaturationThresholdPercent of the pixel format's full scale.
    config = ExposureCalibrationConfig(PixelFormat="Mono8", SaturationThresholdPercent=1.0)
    assert config.SaturationThreshold == 255

    config = ExposureCalibrationConfig(PixelFormat="Mono10", SaturationThresholdPercent=1.0)
    assert config.SaturationThreshold == 1023

    config = ExposureCalibrationConfig(PixelFormat="Mono12", SaturationThresholdPercent=1.0)
    assert config.SaturationThreshold == 4095

    # Default percent is 0.70.
    config = ExposureCalibrationConfig(PixelFormat="Mono8")
    assert config.SaturationThreshold == 178

    with pytest.raises(ValueError):
        config = ExposureCalibrationConfig(PixelFormat="Mono12Packed")
        _ = config.SaturationThreshold

    # Mono16 needs re-scaling and is not implemented yet.
    with pytest.raises(ValueError):
        config = ExposureCalibrationConfig(PixelFormat="Mono16")
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