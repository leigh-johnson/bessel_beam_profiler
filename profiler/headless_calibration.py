"""
Non-interactive exposure calibration for automated gantry scans.

At each Z step of an auto scan the beam brightness changes, so the scan
re-calibrates exposure before each XY cross-section: starting from the
previous Z's exposure, jump proportionally toward a target max-pixel value
just under the saturation threshold, then trim until zero saturated pixels.

This is the headless sibling of calibration.calibrate_exposure_interactive
and reuses its ExposureCalibrationConfig / image_is_overexposed logic. It
is written against two callables (grab a frame, set exposure) so it can be
unit tested without PySpin and reused with any acquisition path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import logging

import numpy as np

from calibration import ExposureCalibrationConfig, image_is_overexposed

logger = logging.getLogger(__name__)


class HeadlessCalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class HeadlessCalibrationConfig:
    Base: ExposureCalibrationConfig = ExposureCalibrationConfig()

    # Accept when: no saturated pixels AND max pixel value is at least
    # TargetLowFraction of the saturation threshold ("maximize the range").
    TargetLowFraction: float = 0.60

    # Proportional jumps assume signal ~ linear in exposure; cap the jump so
    # a nonlinear response cannot cause wild oscillation.
    TargetFraction: float = 0.85
    MaxStepFactor: float = 8.0

    MaxIterations: int = 60


@dataclass(frozen=True)
class HeadlessCalibrationResult:
    FinalExposure_us: float
    LastMax: int
    LastSaturatedPixels: int
    Iterations: int
    Converged: bool
    Note: str = ""


def calibrate_exposure_headless(
    grab_frame: Callable[[], Optional[np.ndarray]],
    set_exposure_us: Callable[[float], float],
    start_exposure_us: float,
    config: HeadlessCalibrationConfig = HeadlessCalibrationConfig(),
) -> HeadlessCalibrationResult:
    """
    Adjust exposure until the frame max lands in the target window.

    grab_frame:      returns the next frame (None = incomplete, retried).
    set_exposure_us: applies a (possibly clamped) exposure and returns the
                     value actually set — clamping is the caller's job since
                     the camera knows its own limits.
    """

    base = config.Base
    threshold = base.SaturationThreshold
    target_value = config.TargetFraction * threshold
    low_value = config.TargetLowFraction * threshold

    exposure_us = set_exposure_us(start_exposure_us)

    last_max = 0
    last_saturated = 0
    incomplete_frames = 0

    for iteration in range(1, config.MaxIterations + 1):
        arr = grab_frame()

        if arr is None:
            incomplete_frames += 1
            if incomplete_frames > 10:
                raise HeadlessCalibrationError(
                    "Too many incomplete frames during headless calibration."
                )
            continue

        _, last_max, last_saturated = image_is_overexposed(arr, base, strict=False)

        if last_saturated > base.AllowedSaturatedPixels:
            # Saturated: the true max is unknown (clipped), so halve.
            proposed = exposure_us * 0.5

        elif last_max < low_value:
            # Too dim: proportional jump toward the target, capped.
            factor = min(config.MaxStepFactor, target_value / max(last_max, 1))
            proposed = exposure_us * factor

        else:
            return HeadlessCalibrationResult(
                FinalExposure_us=exposure_us,
                LastMax=last_max,
                LastSaturatedPixels=last_saturated,
                Iterations=iteration,
                Converged=True,
            )

        new_exposure = set_exposure_us(proposed)

        if new_exposure == exposure_us:
            # Clamped at a camera limit; this is as good as it gets.
            note = (
                "Exposure clamped at camera limit "
                f"({exposure_us:g} us) before reaching the target window."
            )
            logger.warning(note)
            return HeadlessCalibrationResult(
                FinalExposure_us=exposure_us,
                LastMax=last_max,
                LastSaturatedPixels=last_saturated,
                Iterations=iteration,
                Converged=False,
                Note=note,
            )

        exposure_us = new_exposure

    note = f"Did not converge within {config.MaxIterations} iterations."
    logger.warning(note)
    return HeadlessCalibrationResult(
        FinalExposure_us=exposure_us,
        LastMax=last_max,
        LastSaturatedPixels=last_saturated,
        Iterations=config.MaxIterations,
        Converged=False,
        Note=note,
    )
