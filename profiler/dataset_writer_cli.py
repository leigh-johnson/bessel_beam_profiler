from __future__ import annotations

from pathlib import Path
import datetime as dt
import re

import click
from dataset_writer import DatasetWriterJobType
from log_utils import add_file_log

# Default optic configuration for the Bessel-beam line, prompted for at the
# start of every manual XY sweep.
DEFAULT_OPTIC_CONFIG = {
    "GaussianBeamWaist_mm": 4.59, # see beam_gaussian_fit_analysis_$DATE.pynb
    "Axicon1_deg": 5.0,
    "Axicon2_deg": 5.0,
    "Axicon3_deg": 0.5,
    "L12_mm": 190.0,
    "L23_mm": 50.0,
}

def _parse_metadata(metadata_items: tuple[str, ...]) -> dict[str, str]:
    metadata: dict[str, str] = {}

    for item in metadata_items:
        if "=" not in item:
            raise click.ClickException(
                f"Invalid metadata item {item!r}. Expected KEY=VALUE."
            )

        key, value = item.split("=", 1)
        key = key.strip()

        if not key:
            raise click.ClickException(
                f"Invalid metadata item {item!r}. Metadata key cannot be empty."
            )

        metadata[key] = value

    return metadata


def _load_base_camera_settings(camera_settings_path: Path | None):
    """
    Load FLIRCameraSettings from JSON, or defaults when no file is provided.
    """

    from camera_settings import FLIRCameraSettings

    if camera_settings_path is not None:
        return FLIRCameraSettings.from_json_file(camera_settings_path)

    return FLIRCameraSettings()


def _force_software_trigger(camera_settings):
    """
    Return a copy of the settings with software triggering forced on,
    which all dataset acquisition paths require.
    """

    from camera_settings import FLIRCameraSettings

    # set software trigger mode if not already set in the camera settings
    if camera_settings.TriggerMode != "On":
        click.echo(
            "Warning: TriggerMode is not set to 'On' in the camera settings. "
            "Setting TriggerMode to 'On' for this acquisition."
        )
        camera_settings_old = camera_settings.to_dict()
        camera_settings_old["TriggerMode"] = "On"
        camera_settings_old["TriggerSource"] = "Software"
        camera_settings = FLIRCameraSettings(**camera_settings_old)

    return camera_settings


def _load_camera_settings_for_software_trigger(camera_settings_path: Path | None):
    """
    Load FLIRCameraSettings from JSON (or defaults) and force software
    triggering, which all dataset acquisition paths require.
    """

    return _force_software_trigger(_load_base_camera_settings(camera_settings_path))


def _prompt_optic_configuration() -> dict[str, float]:
    """
    Prompt for the axicon angles and inter-optic spacings of the beamline.
    ENTER accepts the defaults for the current Bessel-beam setup.
    """

    click.echo("Optic configuration (ENTER accepts the default):")

    return {
        name: click.prompt(f"  {name}", default=default, type=float)
        for name, default in DEFAULT_OPTIC_CONFIG.items()
    }


def _position_slug(sensor_z_reference: str, sensor_z_cm: float) -> str:
    """
    Filename-safe tag for the sweep position, e.g. 'axicon3-z100.0cm'.
    """

    reference = re.sub(r"[^A-Za-z0-9_.-]+", "_", sensor_z_reference).strip("_")
    return f"{reference or 'optic'}-z{sensor_z_cm:g}cm"


@click.group(name="dataset")
def dataset() -> None:
    """
    Acquire and save beam-profile image datasets.
    """


@dataset.command("static")
@click.option(
    "--camera-settings",
    "camera_settings_path",
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a FLIRCameraSettings JSON file.",
)
@click.option(
    "--dataset-root",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Root directory where the timestamped run directory will be created.",
)
@click.option(
    "--nshots",
    default=1,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of software-triggered images to acquire without moving the camera.",
)
@click.option(
    "--camera-index",
    default=0,
    show_default=True,
    type=click.IntRange(min=0),
    help="Index of the camera to use from PySpin.System.GetCameras().",
)
@click.option(
    "--acquisition-timeout-ms",
    default=2000,
    show_default=True,
    type=click.IntRange(min=1),
    help="Timeout passed to cam.GetNextImage(...), in milliseconds.",
)
@click.option(
    "--settle-time-s",
    default=0.0,
    show_default=True,
    type=click.FloatRange(min=0.0),
    help="Optional delay before image acquisition begins.",
)
@click.option(
    "--placement-id",
    default="static-camera",
    show_default=True,
    help="Human-readable label for this camera/table placement.",
)
@click.option("--gantry-x-mm", default=0.0, show_default=True, type=float)
@click.option("--gantry-y-mm", default=0.0, show_default=True, type=float)
@click.option("--gantry-z-mm", default=0.0, show_default=True, type=float)
@click.option("--table-x-mm", default=0.0, show_default=True, type=float)
@click.option("--table-y-mm", default=0.0, show_default=True, type=float)
@click.option("--table-z-mm", default=0.0, show_default=True, type=float)
@click.option(
    "--metadata",
    multiple=True,
    help="Extra manifest metadata as KEY=VALUE. May be repeated.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force acquisition even if the camera is already streaming. "
    "This may cause the camera to drop frames.",
)
def static(
    camera_settings_path: Path,
    dataset_root: Path,
    nshots: int,
    camera_index: int,
    acquisition_timeout_ms: int,
    settle_time_s: float,
    placement_id: str,
    gantry_x_mm: float,
    gantry_y_mm: float,
    gantry_z_mm: float,
    table_x_mm: float,
    table_y_mm: float,
    table_z_mm: float,
    metadata: tuple[str, ...],
    force: bool,
) -> None:
    """
    Acquire one or more images with a fixed/static camera.

    This is the no-gantry acquisition path.
    """

    from coordinates import Vec3D
    from dataset_writer import DatasetWriterConfig, FLIRDatasetWriter

    camera_settings = _load_camera_settings_for_software_trigger(camera_settings_path)

    config = DatasetWriterConfig(
        DatasetRoot=dataset_root,
        AcquisitionTimeout_ms=acquisition_timeout_ms,
        SettleTime_s=settle_time_s,
        JobType=DatasetWriterJobType.STATIC,
    )

    run_metadata = _parse_metadata(metadata)

    writer = FLIRDatasetWriter(
        camera_index=camera_index,
        camera_settings=camera_settings,
        config=config,
    )

    run_dir = writer.prepare_run()

    # Mirror the log into the run directory (process exits after this
    # command, so the handler is not detached explicitly).
    add_file_log(run_dir / "scan.log")

    records = writer.acquire_static(
        nshots=nshots,
        placement_id=placement_id,
        gantry_position_mm=Vec3D(
            x_mm=gantry_x_mm,
            y_mm=gantry_y_mm,
            z_mm=gantry_z_mm,
        ),
        table_position_mm=Vec3D(
            x_mm=table_x_mm,
            y_mm=table_y_mm,
            z_mm=table_z_mm,
        ),
        metadata=run_metadata,
    )
    # PySpin system.ReleaseInstance() will raise: _PySpin.SpinnakerException: Spinnaker: Can't clear a camera because something still holds a reference to the camera [-1004]
    # if we do not explicitly delete the writer here, which holds a reference to the camera.
    del writer


    click.echo(f"Run directory: {run_dir}")
    click.echo(f"Frames acquired: {len(records)}")

    for record in records:
        click.echo(record.Path)


@dataset.command("manual")
@click.option(
    "--camera-settings",
    "camera_settings_path",
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a FLIRCameraSettings JSON file.",
)
@click.option(
    "--dataset-root",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Root directory where the timestamped run directory will be created.",
)
@click.option(
    "--camera-index",
    default=0,
    show_default=True,
    type=click.IntRange(min=0),
    help="Index of the camera to use from PySpin.System.GetCameras().",
)
@click.option(
    "--acquisition-timeout-ms",
    default=2000,
    show_default=True,
    type=click.IntRange(min=1),
    help="Timeout passed to cam.GetNextImage(...), in milliseconds.",
)
@click.option(
    "--placement-id",
    default="manual-stage",
    show_default=True,
    help="Human-readable label for this camera/table placement.",
)
@click.option(
    "--preview-interval-s",
    default=0.05,
    show_default=True,
    type=click.FloatRange(min=0.01),
    help="Seconds between live preview refreshes.",
)
@click.option(
    "--colormap",
    default="inferno",
    show_default=True,
    help="Matplotlib colormap for the live preview and composite PNG.",
)
@click.option(
    "--metadata",
    multiple=True,
    help="Extra manifest metadata as KEY=VALUE. May be repeated.",
)
@click.option(
    "--stitch/--no-stitch",
    "do_stitch",
    default=True,
    show_default=True,
    help="Stitch saved frames into a composite image when the session ends.",
)
@click.option(
    "--stitch-method",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "phase", "opencv"]),
    help="Registration method: FFT phase correlation, OpenCV ORB features, "
    "or auto (phase correlation with OpenCV fallback).",
)
@click.option(
    "--calibration-dir",
    default=Path("calibrations"),
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory where the exposure calibration for this sweep is saved.",
)
def manual(
    camera_settings_path: Path,
    dataset_root: Path,
    camera_index: int,
    acquisition_timeout_ms: int,
    placement_id: str,
    preview_interval_s: float,
    colormap: str,
    metadata: tuple[str, ...],
    do_stitch: bool,
    stitch_method: str,
    calibration_dir: Path,
) -> None:
    """
    Interactive acquisition with a manually operated translation stage.

    Each sweep walks through: (1) optic configuration, (2) sensor z-position
    and reference optic, (3) interactive exposure calibration saved to
    --calibration-dir, then (4) the XY scan itself using the calibrated
    settings. Press SPACE to save the current frame, move the stage by hand,
    and save again; press q (or close the window) to finish. Saved frames are
    then stitched into a composite.
    """

    from dataclasses import replace

    from calibration import ExposureCalibrationConfig, calibrate_exposure_interactive
    from dataset_writer import DatasetWriterConfig, FLIRDatasetWriter
    from manual_stage import ManualStageConfig, ManualStageSession

    # 1. What optics are on the beamline?
    click.echo("\n--- Step 1/4: optic configuration ---")
    optic_config = _prompt_optic_configuration()

    # 2. Where is the sensor along the beamline right now?
    click.echo("\n--- Step 2/4: sensor z-position ---")
    sensor_z_cm = click.prompt(
        "Camera sensor z-position in cm (e.g. 100.0)",
        type=float,
    )
    sensor_z_reference = click.prompt(
        "Optic the z-position is measured after (e.g. axicon3)",
        default="axicon3",
        show_default=True,
    )

    position_slug = _position_slug(sensor_z_reference, sensor_z_cm)

    # 3. Calibrate exposure at this position, saving the calibrated settings
    #    (plus PNG/NPY of the final frame) beside previous calibrations.
    click.echo("\n--- Step 3/4: exposure calibration ---")

    optic_notes = ", ".join(f"{name}={value:g}" for name, value in optic_config.items())
    base_settings = replace(
        _load_base_camera_settings(camera_settings_path),
        Notes=f"z={sensor_z_cm:g}cm after {sensor_z_reference}; {optic_notes}",
    )

    calibration_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = calibration_dir / (
        f"calibrated_camera_settings_{position_slug}_"
        f"{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    )

    calibration_result = calibrate_exposure_interactive(
        camera_index=camera_index,
        base_settings=base_settings,
        output_json_path=calibration_path,
        config=ExposureCalibrationConfig(
            AcquisitionTimeout_ms=acquisition_timeout_ms,
        ),
    )

    click.echo(f"Calibrated exposure: {calibration_result.FinalExposure_us:.3f} us")
    click.echo(f"Saved calibration: {calibration_path}")

    # 4. Run the XY scan with the calibrated settings, forcing the software
    #    triggering that dataset acquisition requires. The position slug goes
    #    into the placement ID so every frame filename records z + optic.
    click.echo("\n--- Step 4/4: manual XY scan ---")

    camera_settings = _force_software_trigger(calibration_result.Settings)

    config = DatasetWriterConfig(
        JobType=DatasetWriterJobType.MANUAL_SCAN,
        DatasetRoot=dataset_root,
        AcquisitionTimeout_ms=acquisition_timeout_ms,
    )

    writer = FLIRDatasetWriter(
        camera_index=camera_index,
        camera_settings=camera_settings,
        config=config,
    )

    run_dir = writer.prepare_run()

    # Mirror the log into the run directory (process exits after this
    # command, so the handler is not detached explicitly).
    add_file_log(run_dir / "scan.log")

    writer.write_json_artifact(
        "sweep_setup.json",
        {
            "OpticConfiguration": optic_config,
            "SensorZ_cm": sensor_z_cm,
            "SensorZReference": sensor_z_reference,
            "ExposureCalibrationPath": str(calibration_path),
            "CalibratedExposure_us": calibration_result.FinalExposure_us,
        },
    )

    session = ManualStageSession(
        writer,
        ManualStageConfig(
            SensorZ_mm=sensor_z_cm * 10.0,
            SensorZReference=sensor_z_reference,
            PlacementID=f"{placement_id}-{position_slug}",
            PreviewInterval_s=preview_interval_s,
            AcquisitionTimeout_ms=acquisition_timeout_ms,
            Colormap=colormap,
            Metadata={
                "OpticConfiguration": optic_config,
                "ExposureCalibrationPath": str(calibration_path),
                **_parse_metadata(metadata),
            },
        ),
    )

    click.echo("")
    click.echo("Live preview controls:")
    click.echo("  SPACE / s ............ save the current frame")
    click.echo("  q / ESC / close ...... finish the session")
    click.echo("")

    records = session.run()

    # PySpin system.ReleaseInstance() will raise if anything still holds a
    # camera reference; drop the writer before doing CPU-only stitching.
    del session
    del writer

    click.echo(f"Run directory: {run_dir}")
    click.echo(f"Frames saved: {len(records)}")

    for record in records:
        click.echo(record.Path)

    if not do_stitch:
        return

    if len(records) < 2:
        click.echo("Fewer than two frames saved; skipping stitching.")
        return

    from stitcher import StitchConfig, stitch_run_dir

    click.echo("Stitching composite image...")

    outputs = stitch_run_dir(
        run_dir,
        StitchConfig(Method=stitch_method, Colormap=colormap),
    )

    click.echo(f"Composite image: {outputs['png']}")
    click.echo(f"Composite array: {outputs['npy']}")
    click.echo(f"Offsets:         {outputs['offsets']}")


@dataset.command("stitch")
@click.argument(
    "run_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--method",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "phase", "opencv"]),
    help="Registration method: FFT phase correlation, OpenCV ORB features, "
    "or auto (phase correlation with OpenCV fallback).",
)
@click.option(
    "--output-stem",
    default="composite",
    show_default=True,
    help="Filename stem for the composite outputs written into RUN_DIR.",
)
@click.option(
    "--colormap",
    default="inferno",
    show_default=True,
    help="Matplotlib colormap for the composite PNG.",
)
def stitch(run_dir: Path, method: str, output_stem: str, colormap: str) -> None:
    """
    (Re-)stitch the frames of an existing run directory into a composite.

    Reads acquisition order from frames.jsonl when present, otherwise all
    *.npy files in sorted filename order. No camera required.
    """

    from stitcher import StitchConfig, stitch_run_dir

    add_file_log(run_dir / "scan.log")

    outputs = stitch_run_dir(
        run_dir,
        StitchConfig(Method=method, Colormap=colormap),
        output_stem=output_stem,
    )

    click.echo(f"Composite image: {outputs['png']}")
    click.echo(f"Composite array: {outputs['npy']}")
    click.echo(f"Offsets:         {outputs['offsets']}")