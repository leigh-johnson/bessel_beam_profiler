from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import click


@contextmanager
def _open_camera(camera_index: int) -> Iterator[object]:
    import PySpin

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cam = None

    try:
        num_cameras = cam_list.GetSize()

        if num_cameras == 0:
            raise click.ClickException("No FLIR cameras detected.")

        if camera_index >= num_cameras:
            raise click.ClickException(
                f"Requested camera index {camera_index}, "
                f"but only {num_cameras} camera(s) were detected."
            )

        cam = cam_list.GetByIndex(camera_index)
        cam.Init()

        yield cam

    finally:
        if cam is not None:
            try:
                cam.DeInit()
            except PySpin.SpinnakerException as ex:
                click.echo(f"Warning: camera de-init failed: {ex}", err=True)

            del cam

        cam_list.Clear()
        system.ReleaseInstance()


@click.group(name="calibration")
def calibration() -> None:
    """
    Interactive camera calibration routines.
    """


@calibration.command("exposure")
@click.option(
    "--camera-settings",
    "camera_settings_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a base FLIRCameraSettings JSON file.",
)
@click.option(
    "--output",
    "output_json_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to save the calibrated camera settings JSON.",
)
@click.option(
    "--camera-index",
    default=0,
    show_default=True,
    type=click.IntRange(min=0),
    help="Index of the camera to use from PySpin.System.GetCameras().",
)
@click.option(
    "--initial-exposure-us",
    default=1000.0,
    show_default=True,
    type=click.FloatRange(min=0.0, min_open=True),
)
@click.option(
    "--min-exposure-us",
    default=25.0,
    show_default=True,
    type=click.FloatRange(min=0.0, min_open=True),
)
@click.option(
    "--max-exposure-us",
    default=1_000_000.0,
    show_default=True,
    type=click.FloatRange(min=0.0, min_open=True),
)
@click.option(
    "--pixel-format",
    default="Mono16",
    show_default=True,
    type=click.Choice(["Mono8", "Mono16"]),
    help="Pixel format used to choose the saturation threshold.",
)
@click.option(
    "--allowed-saturated-pixels",
    default=0,
    show_default=True,
    type=click.IntRange(min=0),
)
@click.option(
    "--acquisition-timeout-ms",
    default=1000,
    show_default=True,
    type=click.IntRange(min=1),
)
def exposure(
    camera_settings_path: Path,
    output_json_path: Path,
    camera_index: int,
    initial_exposure_us: float,
    min_exposure_us: float,
    max_exposure_us: float,
    pixel_format: str,
    allowed_saturated_pixels: int,
    acquisition_timeout_ms: int,
) -> None:
    """
    Launch the interactive exposure calibration GUI.
    """

    from calibration import ExposureCalibrationConfig, calibrate_exposure_interactive
    from camera_settings import FLIRCameraSettings

    if min_exposure_us > max_exposure_us:
        raise click.ClickException("--min-exposure-us cannot exceed --max-exposure-us.")

    base_settings = FLIRCameraSettings.from_json_file(camera_settings_path)

    config = ExposureCalibrationConfig(
        InitialExposure_us=initial_exposure_us,
        MinExposure_us=min_exposure_us,
        MaxExposure_us=max_exposure_us,
        PixelFormat=pixel_format,
        AllowedSaturatedPixels=allowed_saturated_pixels,
        AcquisitionTimeout_ms=acquisition_timeout_ms,
    )

    with _open_camera(camera_index) as cam:
        result = calibrate_exposure_interactive(
            cam=cam,
            base_settings=base_settings,
            output_json_path=output_json_path,
            config=config,
        )

    click.echo(f"Final exposure: {result.FinalExposure_us:.3f} us")
    click.echo(f"Last max pixel value: {result.LastMax}")
    click.echo(f"Last saturated pixels: {result.LastSaturatedPixels}")
    click.echo(f"Saved camera settings: {output_json_path}")
