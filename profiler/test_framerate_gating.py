"""
Regression tests for AcquisitionFrameRate gating in FLIRCameraSettings.apply.

On the BFS the AcquisitionFrameRate node goes read-only the moment
AcquisitionFrameRateEnable is False, so apply(strict=True) must never try
to write the rate while the limiter is disabled (hardware failure
2026-07-22: "AcquisitionFrameRate is not available/writable").
"""

import importlib
from dataclasses import replace

import pytest

from conftest import FakeCamera


@pytest.fixture()
def camera_settings_module(fake_pyspin):
    return importlib.import_module("camera_settings")


def make_settings(camera_settings_module, **overrides):
    base = camera_settings_module.FLIRCameraSettings(
        TriggerMode="On", TriggerSource="Software"
    )
    return replace(base, **overrides)


def test_disabled_limiter_with_rate_none_applies_cleanly(camera_settings_module):
    cam = FakeCamera()

    settings = make_settings(
        camera_settings_module,
        AcquisitionFrameRateEnable=False,
        AcquisitionFrameRate=None,
    )
    settings.apply(cam, strict=True)

    assert cam.AcquisitionFrameRateEnable.set_calls == [False]
    assert cam.AcquisitionFrameRate.set_calls == []


def test_rate_is_skipped_when_limiter_disabled_even_if_set(camera_settings_module):
    # A settings file with Enable=False but a leftover rate value must not
    # attempt the (read-only) rate write.
    cam = FakeCamera()

    settings = make_settings(
        camera_settings_module, AcquisitionFrameRateEnable=False
    )
    assert settings.AcquisitionFrameRate is not None  # default rate present

    settings.apply(cam, strict=True)

    assert cam.AcquisitionFrameRate.set_calls == []


def test_enabled_limiter_still_applies_rate(camera_settings_module):
    cam = FakeCamera()

    settings = make_settings(camera_settings_module)
    settings.apply(cam, strict=True)

    assert cam.AcquisitionFrameRateEnable.set_calls == [True]
    assert cam.AcquisitionFrameRate.set_calls == [3]
