import importlib

import pytest

from conftest import FakeCamera


MODULE_NAME = "camera_settings"


@pytest.fixture()
def settings_module(fake_pyspin):
    """
    Import camera_settings.py with a fake PySpin module.

    This lets the tests run on a laptop/CI machine without Spinnaker installed
    and without a physical camera connected.
    """
    return importlib.import_module(MODULE_NAME)


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