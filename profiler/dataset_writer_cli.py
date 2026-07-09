from __future__ import annotations

from pathlib import Path

import click
from dataset_writer import DatasetWriterJobType

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


def _load_camera_settings_for_software_trigger(camera_settings_path: Path | None):
    """
    Load FLIRCameraSettings from JSON (or defaults) and force software
    triggering, which all dataset acquisition paths require.
    """

    from camera_settings import FLIRCameraSettings

    if camera_settings_path is not None:
        camera_settings = FLIRCameraSettings.from_json_file(camera_settings_path)
    else:
        # use default settings if no camera settings file is provided
        camera_settings = FLIRCameraSettings()

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
) -> None:
    """
    Interactive acquisition with a manually operated translation stage.

    Shows a live matplotlib preview of the beam. Press SPACE to save the
    current frame, move the stage by hand, and save again; press q (or close
    the window) to finish. Saved frames are then stitched into a composite.
    """

    from dataset_writer import DatasetWriterConfig, FLIRDatasetWriter
    from manual_stage import ManualStageConfig, ManualStageSession

    # 1. Where is the sensor along the beamline right now?
    sensor_z_cm = click.prompt(
        "Camera sensor z-position in cm (e.g. 6.5)",
        type=float,
    )
    sensor_z_reference = click.prompt(
        "Measured from (e.g. 'front face of axicon #1')",
        default="axicon #1",
        show_default=True,
    )

    camera_settings = _load_camera_settings_for_software_trigger(camera_settings_path)

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

    session = ManualStageSession(
        writer,
        ManualStageConfig(
            SensorZ_mm=sensor_z_cm * 10.0,
            SensorZReference=sensor_z_reference,
            PlacementID=placement_id,
            PreviewInterval_s=preview_interval_s,
            AcquisitionTimeout_ms=acquisition_timeout_ms,
            Colormap=colormap,
            Metadata=_parse_metadata(metadata),
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

    outputs = stitch_run_dir(
        run_dir,
        StitchConfig(Method=method, Colormap=colormap),
        output_stem=output_stem,
    )

    click.echo(f"Composite image: {outputs['png']}")
    click.echo(f"Composite array: {outputs['npy']}")
    click.echo(f"Offsets:         {outputs['offsets']}")