"""
`dataset auto` — automated beam-stack scans on the FluidNC gantry.

Coordinate convention (one frame, used everywhere): X = horizontal
transverse, Y = beam propagation down the table, Z = vertical. The scan
captures X-Z cross-section slices and steps along Y.

Implements the placement workflow end to end: prompt for the manually
measured optic->sensor distance (with the camera at machine Y = --y-start),
home the machine, then for each Y step run a headless exposure calibration,
an off-axis ambient background when the exposure has drifted, and the X-Z
raster (adaptive by default). Frames are grouped into per-slice subfolders
(y0100.00cm/...) inside a timestamped run directory, one run directory per
gantry placement.
"""

from __future__ import annotations

from pathlib import Path
import logging

import click

from log_utils import add_file_log, remove_file_log

logger = logging.getLogger(__name__)


@click.command("auto")
# -- connection -------------------------------------------------------------
@click.option("--host", default="fluidnc-sr2.local", show_default=True, help="FluidNC hostname or IP.")
@click.option("--port", default=23, show_default=True, type=int)
@click.option("--feed", default=400.0, show_default=True, type=float, help="Feed rate for scan moves, mm/min.")
# -- camera -----------------------------------------------------------------
@click.option(
    "--camera-settings",
    "camera_settings_path",
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a FLIRCameraSettings JSON file.",
)
@click.option("--camera-index", default=0, show_default=True, type=click.IntRange(min=0))
@click.option("--acquisition-timeout-ms", default=2000, show_default=True, type=click.IntRange(min=1))
@click.option("--nshots", default=1, show_default=True, type=click.IntRange(min=1), help="Shots per raster point.")
@click.option("--settle-time-s", default=0.0, show_default=True, type=click.FloatRange(min=0.0), help="Extra pause between motion-complete and trigger.")
# -- output -----------------------------------------------------------------
@click.option(
    "--dataset-root",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Root directory; one timestamped run directory is created per placement.",
)
# -- X-Z cross-section raster -----------------------------------------------
@click.option(
    "--raster",
    "raster_mode",
    default="adaptive",
    show_default=True,
    type=click.Choice(["adaptive", "fixed"]),
    help="adaptive: grow the raster from one seed frame until its edges are "
    "dark (x/z ranges act as the CAP; a beam that fits in one frame takes "
    "one frame). fixed: always raster the full x/z grid.",
)
@click.option("--x-min", default=45.0, show_default=True, type=float, help="Horizontal transverse (machine X), mm.")
@click.option("--x-max", default=75.0, show_default=True, type=float)
@click.option("--x-step", default=5.0, show_default=True, type=float, help="~30% overlap for the 7.1 mm-wide sensor.")
@click.option("--z-min", default=-75.0, show_default=True, type=float, help="Vertical (machine Z, negative = down), mm.")
@click.option("--z-max", default=-45.0, show_default=True, type=float)
@click.option("--z-step", default=4.0, show_default=True, type=float, help="~25% overlap for the 5.3 mm-tall sensor.")
@click.option("--signal-margin", default=8.0, show_default=True, type=float, help="Adaptive: counts above the background p99 that count as beam signal.")
@click.option("--min-signal-pixels", default=50, show_default=True, type=click.IntRange(min=1), help="Adaptive: pixels above threshold needed to call a border strip 'signal'.")
# -- Y stack: stepping along the beam ---------------------------------------
@click.option("--y-start", default=10.0, show_default=True, type=float, help="Machine Y of the FIRST slice (where you measure the optic->sensor distance).")
@click.option("--y-stop", default=150.0, show_default=True, type=float, help="Machine Y to scan up to.")
@click.option("--y-step", default=10.0, show_default=True, type=float, help="Beam-direction interval, mm (10 = 1 cm).")
@click.option(
    "--beam-direction",
    default="+y",
    show_default=True,
    type=click.Choice(["+y", "-y"]),
    help="+y if machine +Y points downstream (away from the optic), -y if "
    "toward it. Verify once: jog +Y and check the camera moves away from "
    "the optic.",
)
# -- backgrounds ------------------------------------------------------------
@click.option(
    "--background-mode",
    default="offaxis",
    show_default=True,
    type=click.Choice(["offaxis", "ladder", "none"]),
    help="offaxis: per-slice ambient backgrounds with the camera parked "
    "outside the beam (in X/Z) at that slice's calibrated exposure (no "
    "beam blocking). ladder: beam-blocked exposure ladder once per "
    "placement. none: skip.",
)
@click.option("--background-x", "background_x_mm", default=None, type=float, help="Off-axis background X (machine mm). Default: farthest machine-limit X/Z corner.")
@click.option("--background-z", "background_z_mm", default=None, type=float, help="Off-axis background Z (machine mm). Default: farthest machine-limit X/Z corner.")
@click.option(
    "--background-exposure-change",
    default=0.10,
    show_default=True,
    type=click.FloatRange(min=0.0),
    help="Off-axis mode: recapture the background only when the calibrated "
    "exposure changed by at least this fraction since the last captured "
    "background (0 = every slice). Skipped slices reuse the previous one, "
    "recorded in background_reference.json.",
)
@click.option("--background-shots", default=3, show_default=True, type=click.IntRange(min=1))
@click.option("--background-min-us", default=25.0, show_default=True, type=float, help="Ladder mode only.")
@click.option("--background-max-us", default=100000.0, show_default=True, type=float, help="Ladder mode only.")
@click.option("--background-count", default=10, show_default=True, type=click.IntRange(min=2), help="Ladder mode only.")
@click.option("--skip-background", is_flag=True, help="Shorthand for --background-mode none.")
# -- misc -------------------------------------------------------------------
@click.option("--metadata", multiple=True, help="Extra manifest metadata as KEY=VALUE. May be repeated.")
@click.option("--skip-homing", is_flag=True, help="Skip $H (only if already homed THIS power-cycle and placement).")
def auto_scan(
    host: str,
    port: int,
    feed: float,
    camera_settings_path,
    camera_index: int,
    acquisition_timeout_ms: int,
    nshots: int,
    settle_time_s: float,
    dataset_root: Path,
    raster_mode: str,
    x_min: float,
    x_max: float,
    x_step: float,
    z_min: float,
    z_max: float,
    z_step: float,
    signal_margin: float,
    min_signal_pixels: int,
    y_start: float,
    y_stop: float,
    y_step: float,
    beam_direction: str,
    background_mode: str,
    background_x_mm,
    background_z_mm,
    background_exposure_change: float,
    background_shots: int,
    background_min_us: float,
    background_max_us: float,
    background_count: int,
    skip_background: bool,
    metadata,
    skip_homing: bool,
) -> None:
    """
    Automated stack of X-Z beam cross-sections stepped along Y (the beam).

    Repeats for as many gantry placements as you like; each placement gets
    its own run directory with per-slice subfolders.
    """

    from auto_scan import (
        AutoScanConfig,
        AutoScanSession,
        default_background_ladder_us,
    )
    from coordinates import AxisRange
    from dataset_writer import (
        DatasetWriterConfig,
        DatasetWriterJobType,
        FLIRDatasetWriter,
    )
    from dataset_writer_cli import (
        _load_camera_settings_for_software_trigger,
        _parse_metadata,
        _prompt_optic_configuration,
    )
    from fluidnc_stage import (
        FluidNCClient,
        FluidNCClientConfig,
        FluidNCStageConfig,
        FluidNCStageController,
    )

    run_metadata = _parse_metadata(metadata)

    click.echo("\n--- Optic configuration ---")
    optic_config = _prompt_optic_configuration()

    click.echo(f"\nConnecting to FluidNC at {host}:{port} ...")

    client = FluidNCClient(FluidNCClientConfig(Host=host, Port=port, Feed_mm_min=feed))
    client.connect()

    stage = FluidNCStageController(client, FluidNCStageConfig(Feed_mm_min=feed))

    status = client.query_status()
    click.echo(f"FluidNC: {status.Raw}")

    if skip_background:
        background_mode = "none"

    background_ladder = (
        default_background_ladder_us(
            background_min_us, background_max_us, background_count
        )
        if background_mode == "ladder"
        else ()
    )

    placement_number = 1

    try:
        while True:
            click.echo(f"\n=== Placement {placement_number} ===")

            if not skip_homing or status.is_alarm:
                click.confirm(
                    "About to home ($H): the gantry will move to its limit "
                    "switches. Area clear?",
                    abort=True,
                )
                click.echo("Homing...")
                client.home()
                click.echo(f"Homed: {client.query_status().Raw}")

            placement_id = click.prompt(
                "Placement ID",
                default=f"placement-{placement_number:02d}",
                show_default=True,
            )

            click.echo(
                f"\nMeasure the distance from the optic to the camera sensor "
                f"along the beam, with the camera at machine Y = {y_start:g} mm."
            )
            measured_cm = click.prompt(
                "Measured optic -> sensor distance (cm)", type=float
            )
            measured_from = click.prompt(
                "Optic the distance is measured from",
                default="axicon3",
                show_default=True,
            )

            config = AutoScanConfig(
                PlacementID=placement_id,
                MeasuredSensorY_mm=measured_cm * 10.0,
                MeasuredFrom=measured_from,
                YStart_machine_mm=y_start,
                YStop_machine_mm=y_stop,
                YStep_mm=y_step,
                BeamDirectionSign=1 if beam_direction == "+y" else -1,
                X=AxisRange(start_mm=x_min, stop_mm=x_max, step_mm=x_step),
                Z=AxisRange(start_mm=z_min, stop_mm=z_max, step_mm=z_step),
                RasterMode=raster_mode,
                SignalMargin_counts=signal_margin,
                MinSignalPixels=min_signal_pixels,
                NShots=nshots,
                BackgroundMode=background_mode,
                BackgroundX_mm=background_x_mm,
                BackgroundZ_mm=background_z_mm,
                BackgroundExposureChangeFraction=background_exposure_change,
                BackgroundExposures_us=background_ladder,
                BackgroundShots=background_shots,
                Metadata={
                    "OpticConfiguration": optic_config,
                    **run_metadata,
                },
            )

            n_slices = len(config.y_values_machine_mm())
            n_xz = len(config.X.values()) * len(config.Z.values())

            if background_mode == "offaxis":
                # Upper bound: slices whose exposure moved < the change
                # threshold reuse the previous background.
                n_background = f"up to {n_slices * background_shots}"
            elif background_mode == "ladder":
                n_background = str(len(background_ladder) * background_shots)
            else:
                n_background = "0"

            if raster_mode == "adaptive":
                click.echo(
                    f"\nPlan: {n_slices} Y-slices, adaptive X-Z raster capped "
                    f"at {n_xz} points x {nshots} shot(s) per slice "
                    f"(up to {n_slices * n_xz * nshots} frames; typically far "
                    f"fewer; + {n_background} background frames, "
                    f"mode={background_mode})."
                )
            else:
                click.echo(
                    f"\nPlan: {n_slices} Y-slices x {n_xz} X-Z points x "
                    f"{nshots} shot(s) = {n_slices * n_xz * nshots} frames "
                    f"(+ {n_background} background frames, mode={background_mode})."
                )
            click.confirm("Start this placement's scan?", abort=True)

            camera_settings = _load_camera_settings_for_software_trigger(
                camera_settings_path
            )

            writer = FLIRDatasetWriter(
                camera_index=camera_index,
                camera_settings=camera_settings,
                config=DatasetWriterConfig(
                    JobType=DatasetWriterJobType.AUTO_SCAN,
                    DatasetRoot=dataset_root,
                    AcquisitionTimeout_ms=acquisition_timeout_ms,
                    SettleTime_s=settle_time_s,
                ),
                stage_controller=stage,
            )

            run_dir = writer.prepare_run()

            session = AutoScanSession(
                writer,
                config,
                pause_fn=lambda message: click.pause(f"\n>>> {message}\nPress any key when ready..."),
            )

            # Everything logged during this placement (any module) also
            # lands in a scan.log next to the data.
            log_handler = add_file_log(run_dir / "scan.log")
            logger.info(f"Run directory: {run_dir}")

            try:
                records = session.run(stage.config.MachineLimits_mm)
                logger.info(
                    f"Placement done: {len(records)} frames in {run_dir}"
                )
            finally:
                remove_file_log(log_handler)

            # Release the camera before the next placement (PySpin cleanup).
            del session
            del writer

            if not click.confirm(
                "\nMove the gantry to a NEW placement and scan again?",
                default=False,
            ):
                break

            click.echo(
                "Reposition the gantry, then continue. The machine will be "
                "re-homed and you will be asked for a fresh distance "
                "measurement."
            )
            skip_homing = False  # always re-home after a physical move
            placement_number += 1
            status = client.query_status()

    except KeyboardInterrupt:
        click.echo("\nInterrupted — sending feed hold (!) to the gantry.")
        try:
            client.feed_hold()
        except Exception as ex:  # noqa: BLE001 - best-effort safety stop
            click.echo(f"Feed hold failed: {ex}")
        click.echo(
            "Machine is holding. Resume from the WebUI (~) or re-home with "
            "'python cli.py gantry home'."
        )
        raise

    finally:
        client.close()
