from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional
import logging

import matplotlib.pyplot as plt
import numpy as np
import PySpin

from camera_settings import FLIRCameraSettings, PixelFormatName
from camera_base import FLIRCameraControllerBase

logger = logging.getLogger(__name__)

class OverexposedError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExposureCalibrationConfig:
    InitialExposure_us: float = 1000.0
    MinExposure_us: float = 25.0
    MaxExposure_us: float = 1_000_000.0

    # For BFS-PGE-31S4M Mono8 (uint8), 255 is the maximum pixel value for greyscale images.
    # TODO Change to 65535 if saturating at uint16 max.
    PixelFormat: PixelFormatName = "Mono8"

    AllowedSaturatedPixels: int = 0
    SaturationThresholdPercent: float = 0.70

    ReductionFactor: float = 0.95
    IncreaseFactor: float = 1.05

    AcquisitionTimeout_ms: int = 10000
    DisplayPause_s: float = 0.001

    @property
    def SaturationThreshold(self) -> int:
        if self.PixelFormat == "Mono8":
            return int(255 * self.SaturationThresholdPercent)
        elif self.PixelFormat == "Mono10":
            return int(1023 * self.SaturationThresholdPercent)
        elif self.PixelFormat == "Mono12":
            return int(4095 * self.SaturationThresholdPercent)
        elif self.PixelFormat == "Mono12Packed":
             # TODO I'm not messing with Mono12Packed bit-packed image format for now, since it uses non-linear mapping. See the FLIR Spinnaker SDK documentation for details. You're welcome to implement it, but know that it will require a different non-linear way of evaluating thesaturation threshold and a different way to unpack the pixel values.
            raise ValueError(f"Unsupported PixelFormat: {self.PixelFormat}. Implement ExposureCalibrationConfig.saturation_threshold for this PixelFormat.")
        # https://www.flir.com/support-center/instruments2/clarification-of-the-flir-ax5-camera-pixel-formats2/
        # Teledyne Mono16 always returns 1 for bits 14 and 15, so the maximum pixel value is 16383
        elif self.PixelFormat == "Mono16":
            # Mono16 requires additional re-scaling. 
            raise ValueError(f"Unsupported PixelFormat: {self.PixelFormat}. Implement ExposureCalibrationConfig.saturation_threshold for this PixelFormat.")
        else:
            # TODO I'm not messing with Mono12Packed bit-packed image format for now, since it uses non-linear mapping. See the FLIR Spinnaker SDK documentation for details. You're welcome to implement it, but know that it will require a different non-linear way of evaluating thesaturation threshold and a different way to unpack the pixel values.
            raise ValueError(f"Unsupported PixelFormat: {self.PixelFormat}. Implement ExposureCalibrationConfig.saturation_threshold for this PixelFormat.")

@dataclass(frozen=True)
class ExposureCalibrationResult:
    Settings: FLIRCameraSettings
    FinalExposure_us: float
    LastMax: int
    LastSaturatedPixels: int


def calibrate_exposure_interactive(
   camera_index: int,
    base_settings: FLIRCameraSettings,
    output_json_path: Path,
    config: ExposureCalibrationConfig = ExposureCalibrationConfig(),
) -> ExposureCalibrationResult:
    """
    Interactive exposure calibration at one fixed camera/stage position.

    Controls:
        a       auto-reduce exposure until saturation disappears
        +/=     increase exposure
        -/_     decrease exposure
        q/esc   accept current settings and quit

    Assumes:
        cam.Init() has already been called.
    """

    exposure_us = base_settings.ExposureTime or config.InitialExposure_us
    exposure_us = _clamp(exposure_us, config.MinExposure_us, config.MaxExposure_us)

    settings = replace(
        base_settings,
        ExposureAuto="Off",
        ExposureMode="Timed",
        ExposureTime=exposure_us,
        GainAuto="Off",
        Gain=0.0,
        GammaEnable=False,
        PixelFormat=config.PixelFormat,
    )

    print("Initializing camera controller")
    flir_camera_controller = FLIRCameraControllerBase(camera_index, settings)
    print(f"Applying camera settings: {settings}")

    flir_camera_controller.apply_settings()

    state = {
        "running": True,
        "auto_reduce": False,
        "exposure_us": exposure_us,
        "last_max": 0,
        "last_saturated": 0,
    }

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("FLIR exposure calibration")
    # unbind default key press handler to avoid closing the window on key press
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    image_artist = None

    def on_key(event):
        key = event.key

        if key in ("q", "escape"):
            state["running"] = False

        elif key == "a":
            state["auto_reduce"] = True

        elif key in ("+", "="):
            state["auto_reduce"] = False
            new_exposure = state["exposure_us"] * config.IncreaseFactor
            state["exposure_us"] = _set_exposure_us(flir_camera_controller.cam, new_exposure, config)

        elif key in ("-", "_"):
            state["auto_reduce"] = False
            new_exposure = state["exposure_us"] * config.ReductionFactor
            state["exposure_us"] = _set_exposure_us(flir_camera_controller.cam, new_exposure, config)


    fig.canvas.mpl_connect("key_press_event", on_key)

    print(
        "\nExposure calibration controls:\n"
        "  a       auto-reduce exposure until unsaturated\n"
        "  +/=     increase exposure\n"
        "  -/_     decrease exposure\n"
        "  q/esc   accept and quit\n"
    )
    print("Beginning acquisition")
    flir_camera_controller._begin_acquisition()
    print(f"Starting exposure calibration with initial exposure = {state['exposure_us']:.3f} us")
    arr = None
    try:
        while state["running"]:
            timeout_ms = int(config.AcquisitionTimeout_ms + (state["exposure_us"] / 1000))
            image_result = flir_camera_controller.cam.GetNextImage(timeout_ms)

            try:
                if image_result.IsIncomplete():
                    print(
                        f"Image incomplete with status "
                        f"{image_result.GetImageStatus()}"
                    )
                    continue

                arr = np.array(image_result.GetNDArray(), copy=True)

            finally:
                image_result.Release()

            is_overexposed, max_value, saturated_pixels = image_is_overexposed(arr, config, strict=False)

            state["last_max"] = max_value
            state["last_saturated"] = saturated_pixels

            if (
                state["auto_reduce"]
                and saturated_pixels > config.AllowedSaturatedPixels
            ):
                new_exposure = state["exposure_us"] * config.ReductionFactor
                state["exposure_us"] = _set_exposure_us(flir_camera_controller.cam, new_exposure, config)

            elif (
                state["auto_reduce"]
                and saturated_pixels <= config.AllowedSaturatedPixels
            ):
                state["auto_reduce"] = False
                print(
                    f"Unsaturated at exposure = "
                    f"{state['exposure_us']:.3f} us, "
                    f"max = {max_value}, saturated pixels = {saturated_pixels}"
                )

            title = (
                f"Pixel format: {config.PixelFormat} | "
                f"Exposure: {state['exposure_us']:.3f} us \n"
                f"max pixel value: {max_value} | "
                f"sat pixel count: {saturated_pixels} |"
                f"sat pixel threshold: {config.SaturationThreshold:.1f}"
            )

            if image_artist is None:
                image_artist = ax.imshow(
                    arr,
                    cmap='inferno',
                    vmin=0,
                    vmax=config.SaturationThreshold,
                )
                ax.set_title(title)
            else:
                image_artist.set_data(arr)
                ax.set_title(title)

            fig.canvas.draw_idle()
            plt.pause(config.DisplayPause_s)

        image_artist.set_data(arr)
        ax.set_title(title) 
        fig.canvas.draw_idle() 
        fig.canvas.flush_events()
        fig.savefig(output_json_path.with_suffix('.png'), dpi=140, bbox_inches='tight')
        print(f"Saved: {output_json_path.with_suffix('.png')}")

        plt.close(fig)

        np.save(output_json_path.with_suffix('.npy'), arr)
        print(f"Saved: {output_json_path.with_suffix('.npy')}")
    finally:
        flir_camera_controller._end_acquisition()


    final_settings = replace(
        settings,
        ExposureTime=state["exposure_us"],
    )

    final_settings.to_json_file(output_json_path)

    return ExposureCalibrationResult(
        Settings=final_settings,
        FinalExposure_us=state["exposure_us"],
        LastMax=state["last_max"],
        LastSaturatedPixels=state["last_saturated"],
    )


def _set_exposure_us(
    cam: PySpin.Camera,
    exposure_us: float,
    config: ExposureCalibrationConfig,
) -> float:
    exposure_us = _clamp(
        exposure_us,
        config.MinExposure_us,
        config.MaxExposure_us,
    )
    cam.ExposureTime.SetValue(float(exposure_us))
    return float(exposure_us)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def image_is_overexposed(image: np.ndarray, config: ExposureCalibrationConfig, strict=False) -> tuple[bool, int, int]:
    """
    Check if the calibration image is saturated due to overexposure.
    strict: If True, raise an OverexposedError if the image is saturated. If False, log a debug message instead.

    Raises:
        OverexposedError: If the image is saturated.
    """
    max_value = int(np.max(image))
    saturated_pixels = int(np.sum(image >= config.SaturationThreshold))

    if saturated_pixels > config.AllowedSaturatedPixels:
        if strict:
            raise OverexposedError(
                f"Calibration image is saturated: "
                f"max = {max_value}, "
                f"saturated pixels = {saturated_pixels}, "
                f"allowed = {config.AllowedSaturatedPixels}"
            )
        else:
            logger.debug(
                f"Calibration image is saturated: "
                f"max = {max_value}, "
                f"saturated pixels = {saturated_pixels}, "
                f"allowed = {config.AllowedSaturatedPixels}"
            )
            return True, max_value, saturated_pixels
    return False, max_value, saturated_pixels