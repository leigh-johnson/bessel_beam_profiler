from __future__ import annotations

from pathlib import Path

import click

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

    from camera_settings import FLIRCameraSettings
    from coordinates import Vec3D
    from dataset_writer import DatasetWriterConfig, FLIRDatasetWriter

    if camera_settings_path is not None:
        camera_settings = FLIRCameraSettings.from_json_file(camera_settings_path)
    else:
        # use default settings if no camera settings file is provided
        camera_settings = FLIRCameraSettings()

    config = DatasetWriterConfig(
        DatasetRoot=dataset_root,
        AcquisitionTimeout_ms=acquisition_timeout_ms,
        SettleTime_s=settle_time_s,
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

