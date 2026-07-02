from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional
import time

import matplotlib.pyplot as plt
import numpy as np
import PySpin

from camera_settings import FLIRCameraSettings


@dataclass(frozen=True)
class ExposureCalibrationConfig:
    InitialExposure_us: float = 1000.0
    MinExposure_us: float = 25.0
    MaxExposure_us: float = 1_000_000.0

    # For BFS-PGE-31S4M native 12-bit data, 4095 is a reasonable first guess?
    # Change to 65535 ifsaturating at uint16 max.
    SaturationThreshold: int = 4095
    AllowedSaturatedPixels: int = 0

    ReductionFactor: float = 0.75
    IncreaseFactor: float = 1.25

    AcquisitionTimeout_ms: int = 1000
    DisplayPause_s: float = 0.001


@dataclass(frozen=True)
class ExposureCalibrationResult:
    Settings: FLIRCameraSettings
    FinalExposure_us: float
    LastMax: int
    LastSaturatedPixels: int


def calibrate_exposure_interactive(
    cam: PySpin.Camera,
    base_settings: FLIRCameraSettings,
    *,
    output_json_path: Optional[str | Path] = None,
    config: ExposureCalibrationConfig = ExposureCalibrationConfig(),
) -> ExposureCalibrationResult:
    """
    Interactive exposure calibration at one fixed camera/stage position.

    Controls:
        a       auto-reduce exposure until saturation disappears
        +/=     increase exposure
        -/_     decrease exposure
        s       save current settings to JSON
        q/esc   accept current settings and quit

    Assumes:
        cam.Init() has already been called.
    """

    exposure_us = base_settings.ExposureTime or config.InitialExposure_us
    exposure_us = _clamp(exposure_us, config.MinExposure_us, config.MaxExposure_us)

    settings = replace(
        base_settings,
        PixelFormat="Mono16",
        ExposureAuto="Off",
        ExposureMode="Timed",
        ExposureTime=exposure_us,
        GainAuto="Off",
        Gain=0.0,
        GammaEnable=False,
    )

    settings.apply(cam, strict=True)
    _set_acquisition_mode(cam, "Continuous")
    _set_stream_buffer_handling_mode(cam, "NewestOnly")

    state = {
        "running": True,
        "auto_reduce": False,
        "save_requested": False,
        "exposure_us": exposure_us,
        "last_max": 0,
        "last_saturated": 0,
    }

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("FLIR exposure calibration")

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
            state["exposure_us"] = _set_exposure_us(cam, new_exposure, config)

        elif key in ("-", "_"):
            state["auto_reduce"] = False
            new_exposure = state["exposure_us"] * config.ReductionFactor
            state["exposure_us"] = _set_exposure_us(cam, new_exposure, config)

        elif key == "s":
            state["save_requested"] = True

    fig.canvas.mpl_connect("key_press_event", on_key)

    print(
        "\nExposure calibration controls:\n"
        "  a       auto-reduce exposure until unsaturated\n"
        "  +/=     increase exposure\n"
        "  -/_     decrease exposure\n"
        "  s       save current settings JSON\n"
        "  q/esc   accept and quit\n"
    )

    cam.BeginAcquisition()

    try:
        while state["running"]:
            image_result = cam.GetNextImage(config.AcquisitionTimeout_ms)

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

            max_value = int(np.max(arr))
            saturated_pixels = int(np.sum(arr >= config.SaturationThreshold))

            state["last_max"] = max_value
            state["last_saturated"] = saturated_pixels

            if (
                state["auto_reduce"]
                and saturated_pixels > config.AllowedSaturatedPixels
            ):
                new_exposure = state["exposure_us"] * config.ReductionFactor
                state["exposure_us"] = _set_exposure_us(cam, new_exposure, config)

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
                f"Exposure: {state['exposure_us']:.3f} us | "
                f"max: {max_value} | "
                f"sat pixels: {saturated_pixels} | "
                f"auto: {state['auto_reduce']}"
            )

            if image_artist is None:
                image_artist = ax.imshow(
                    arr,
                    cmap="gray",
                    vmin=0,
                    vmax=config.SaturationThreshold,
                )
                ax.set_title(title)
            else:
                image_artist.set_data(arr)
                ax.set_title(title)

            fig.canvas.draw_idle()
            plt.pause(config.DisplayPause_s)

            if state["save_requested"]:
                current_settings = replace(
                    settings,
                    ExposureTime=state["exposure_us"],
                )
                if output_json_path is None:
                    path = Path("calibrated_camera_settings.json")
                else:
                    path = Path(output_json_path)

                current_settings.to_json_file(path)
                print(f"Saved: {path}")
                state["save_requested"] = False

    finally:
        try:
            cam.EndAcquisition()
        finally:
            plt.close(fig)

    final_settings = replace(
        settings,
        ExposureTime=state["exposure_us"],
    )

    if output_json_path is not None:
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


def _set_acquisition_mode(cam: PySpin.Camera, mode: str) -> None:
    nodemap = cam.GetNodeMap()
    node = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))

    if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
        raise RuntimeError("Unable to access AcquisitionMode.")

    entry = node.GetEntryByName(mode)

    if not PySpin.IsReadable(entry):
        raise RuntimeError(f"AcquisitionMode entry {mode!r} not readable.")

    node.SetIntValue(entry.GetValue())


def _set_stream_buffer_handling_mode(cam: PySpin.Camera, mode: str) -> None:
    stream_nodemap = cam.GetTLStreamNodeMap()
    node = PySpin.CEnumerationPtr(
        stream_nodemap.GetNode("StreamBufferHandlingMode")
    )

    if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
        print("StreamBufferHandlingMode not available; skipping.")
        return

    entry = node.GetEntryByName(mode)

    if not PySpin.IsReadable(entry):
        print(f"StreamBufferHandlingMode={mode!r} not readable; skipping.")
        return

    node.SetIntValue(entry.GetValue())