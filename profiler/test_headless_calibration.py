"""
Tests for the headless (auto-scan) exposure calibration loop.

These exercise only the pure control logic via injected callables, so they
need the fake PySpin fixture just to import calibration's config/helpers.
"""

import importlib

import numpy as np
import pytest


@pytest.fixture()
def headless(fake_pyspin):
    return importlib.import_module("headless_calibration")


class FakeExposure:
    def __init__(self, lo=25.0, hi=1_000_000.0):
        self.lo = lo
        self.hi = hi
        self.values = []

    def set(self, us):
        value = max(self.lo, min(self.hi, us))
        self.values.append(value)
        return value


def frame(value, shape=(4, 4)):
    return np.full(shape, value, dtype=np.uint8)


# Default Mono8 config: saturation threshold = 0.70 * 255 = 178,
# accept window = [0.60 * 178, no saturated px] = max >= 106.8.


def test_converges_immediately_when_in_window(headless):
    exposure = FakeExposure()

    result = headless.calibrate_exposure_headless(
        grab_frame=lambda: frame(150),
        set_exposure_us=exposure.set,
        start_exposure_us=1000.0,
    )

    assert result.Converged
    assert result.FinalExposure_us == 1000.0
    assert result.LastSaturatedPixels == 0
    assert result.Iterations == 1


def test_halves_exposure_when_saturated(headless):
    exposure = FakeExposure()
    frames = iter([frame(200), frame(200), frame(150)])

    result = headless.calibrate_exposure_headless(
        grab_frame=lambda: next(frames),
        set_exposure_us=exposure.set,
        start_exposure_us=1000.0,
    )

    assert result.Converged
    assert exposure.values == [1000.0, 500.0, 250.0]
    assert result.FinalExposure_us == 250.0


def test_increases_proportionally_when_dim_with_step_cap(headless):
    exposure = FakeExposure()
    frames = iter([frame(10), frame(120)])

    result = headless.calibrate_exposure_headless(
        grab_frame=lambda: next(frames),
        set_exposure_us=exposure.set,
        start_exposure_us=100.0,
    )

    assert result.Converged
    # max=10 is far below target 0.85*178=151.3 -> factor capped at 8x.
    assert exposure.values == [100.0, 800.0]
    assert result.FinalExposure_us == 800.0


def test_reports_not_converged_when_clamped_at_camera_limit(headless):
    exposure = FakeExposure(hi=200.0)

    result = headless.calibrate_exposure_headless(
        grab_frame=lambda: frame(5),
        set_exposure_us=exposure.set,
        start_exposure_us=200.0,
    )

    assert not result.Converged
    assert result.FinalExposure_us == 200.0
    assert "clamped" in result.Note


def test_skips_incomplete_frames_then_errors_after_too_many(headless):
    exposure = FakeExposure()

    with pytest.raises(headless.HeadlessCalibrationError, match="incomplete"):
        headless.calibrate_exposure_headless(
            grab_frame=lambda: None,
            set_exposure_us=exposure.set,
            start_exposure_us=1000.0,
        )


def test_gives_up_after_max_iterations(headless):
    exposure = FakeExposure()
    # Oscillate forever: alternate saturated / dim, never in window.
    state = {"n": 0}

    def grab():
        state["n"] += 1
        return frame(200) if state["n"] % 2 else frame(5)

    config = headless.HeadlessCalibrationConfig(MaxIterations=6)

    result = headless.calibrate_exposure_headless(
        grab_frame=grab,
        set_exposure_us=exposure.set,
        start_exposure_us=1000.0,
        config=config,
    )

    assert not result.Converged
    assert result.Iterations == 6
