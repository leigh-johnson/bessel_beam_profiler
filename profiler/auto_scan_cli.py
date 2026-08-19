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


def resolve_scan_caps(limits, x_min, x_max, z_min, z_max):
    """
    Fill unset X/Z caps from the machine envelope (already margin-shrunk).
    With the adaptive raster, wide caps cost nothing — the raster only
    visits the beam — so the whole envelope is the sane default; explicit
    values are kept as-is (and validated upstream).
    """

    return (
        limits.x_min_mm if x_min is None else x_min,
        limits.x_max_mm if x_max is None else x_max,
        limits.z_min_mm if z_min is None else z_min,
        limits.z_max_mm if z_max is None else z_max,
    )


@click.command("auto")
# -- connection -------------------------------------------------------------
@click.option("--host", default="fluidnc-sr2.local", show_default=True, help="FluidNC hostname or IP.")
@click.option("--port", default=23, show_default=True, type=int)
@click.option("--feed", default=400.0, show_default=True, type=float, help="Feed rate for scan moves, mm/min.")
@click.option(
    "--soft-limit-margin",
    default=1.0,
    show_default=True,
    type=click.FloatRange(min=0.0),
    help="Keep-out margin (mm) inside the firmware's soft limits; all "
    "bounds are validated against firmware limits minus this margin.",
)
# -- camera -----------------------------------------------------------------
@click.option(
    "--camera-settings",
    "camera_settings_path",
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a FLIRCameraSettings JSON file.",
)
@click.option("--camera-index", default=0, show_default=True, type=click.IntRange(min=0))
@click.option(
    "--camera-serial",
    default=24520699,
    type=str,
    help="Open the camera with this DeviceSerialNumber (e.g. 24520699) "
    "instead of trusting enumeration order. Takes precedence over "
    "--camera-index; errors out (listing detected serials) if absent.",
)
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
@click.option("--x-min", default=None, type=float, help="Horizontal transverse cap (machine X), mm. Default: the firmware soft-limit envelope (minus margin).")
@click.option("--x-max", default=None, type=float, help="Default: firmware envelope.")
@click.option("--x-step", default=5.0, show_default=True, type=float, help="~30% overlap for the 7.1 mm-wide sensor.")
@click.option("--z-min", default=None, type=float, help="Vertical cap (machine Z, negative = down), mm. Default: firmware envelope.")
@click.option("--z-max", default=None, type=float, help="Default: firmware envelope.")
@click.option("--z-step", default=4.0, show_default=True, type=float, help="~25% overlap for the 5.3 mm-tall sensor.")
@click.option("--calibration-x", "calibration_x_mm", default=None, type=float, help="Where the camera sits during exposure calibration (machine X). Default: center of the X caps. Point this at a known-bright part of the beam.")
@click.option("--calibration-z", "calibration_z_mm", default=None, type=float, help="Where the camera sits during exposure calibration (machine Z). Default: center of the Z caps.")
@click.option(
    "--follow-beam/--no-follow-beam",
    "follow_beam",
    default=True,
    show_default=True,
    help="Track the beam along the stack: each slice calibrates and seeds "
    "its raster at the previous slice's brightest cell (ring beams drift "
    "and change radius with Y).",
)
@click.option(
    "--find-beam/--no-find-beam",
    "find_beam",
    default=True,
    show_default=True,
    help="When no --calibration-x/-z is given (or the beam is lost), sweep "
    "a Z column from the far extremum looking for structured light "
    "(contrast, not brightness — ambient can't fake it) and seed there.",
)
@click.option("--signal-margin", default=8.0, show_default=True, type=float, help="Adaptive: counts above the background p99 that count as beam signal.")
@click.option("--min-signal-pixels", default=50, show_default=True, type=click.IntRange(min=1), help="Adaptive: pixels above threshold needed to call a border strip 'signal'.")
# -- Y stack: stepping along the beam ---------------------------------------
@click.option(
    "--preview/--no-preview",
    default=False,
    show_default=True,
    help="Open a live viewer (separate process) showing each frame as it "
    "is saved. The viewer only reads files — it cannot slow or block "
    "the capture loop, and closing it does not affect the scan. "
    "Equivalent to running `dataset watch <run_dir>` yourself.",
)
@click.option("--y-start", default=10.0, show_default=True, type=float, help="Machine Y of the FIRST slice (where you measure the optic->sensor distance). May be LARGER than --y-stop: the scan then walks Y downward — useful for diverging beams, to bootstrap near the optic where the beam is smallest/brightest and track it outward slice by slice.")
@click.option("--y-stop", default=150.0, show_default=True, type=float, help="Machine Y of the last slice (either side of --y-start).")
@click.option("--y-step", default=10.0, show_default=True, type=float, help="Beam-direction interval, mm (10 = 1 cm).")
@click.option(
    "--beam-direction",
    default="-y",
    show_default=True,
    type=click.Choice(["+y", "-y"]),
    help="-y if machine +Y points TOWARD the optic (verified on this rig "
    "2026-07-22 via preflight), +y if it points downstream. Re-verify "
    "after any gantry re-orientation: jog +Y and watch the camera.",
)
# -- backgrounds ------------------------------------------------------------
@click.option(
    "--background-mode",
    default="offaxis",
    show_default=True,
    type=click.Choice(["offaxis", "none"]),
    help="offaxis: per-slice ambient backgrounds with the camera parked "
    "outside the beam (in X/Z) at that slice's calibrated exposure (no "
    "beam blocking). none: skip.",
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
@click.option("--skip-background", is_flag=True, help="Shorthand for --background-mode none.")
# -- notifications ----------------------------------------------------------
@click.option(
    "--slack-webhook",
    "slack_webhook",
    envvar="SLACK_WEBHOOK_URL",
    default=None,
    show_envvar=True,
    help="Slack incoming-webhook URL (a secret — prefer the env var). "
    "Pings on placement finish/failure. Superseded by --slack-bot-token "
    "when both are set.",
)
@click.option(
    "--slack-bot-token",
    "slack_bot_token",
    envvar="SLACK_BOT_TOKEN",
    default=None,
    show_envvar=True,
    help="Slack bot token (xoxb-..., a secret — prefer the env var; needs "
    "chat:write and the bot invited to the channel). Enables the thread "
    "pattern: one 'scan started' parent per placement, logs streamed as "
    "batched thread replies, finish ping broadcast to the channel.",
)
@click.option(
    "--slack-channel",
    "slack_channel",
    envvar="SLACK_CHANNEL",
    default="#logs-bessel-beam",
    show_default=True,
    show_envvar=True,
    help="Channel for bot-token posts (webhooks are bound at creation).",
)
@click.option(
    "--slack-log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Minimum level of log records streamed to the Slack thread.",
)
# -- misc -------------------------------------------------------------------
@click.option("--metadata", multiple=True, help="Extra manifest metadata as KEY=VALUE. May be repeated.")
@click.option("--skip-homing", is_flag=True, help="Skip $H (only if already homed THIS power-cycle and placement).")
def auto_scan(
    host: str,
    port: int,
    feed: float,
    soft_limit_margin: float,
    camera_settings_path,
    camera_index: int,
    camera_serial,
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
    calibration_x_mm,
    calibration_z_mm,
    follow_beam: bool,
    find_beam: bool,
    signal_margin: float,
    min_signal_pixels: int,
    preview: bool,
    y_start: float,
    y_stop: float,
    y_step: float,
    beam_direction: str,
    background_mode: str,
    background_x_mm,
    background_z_mm,
    background_exposure_change: float,
    background_shots: int,
    skip_background: bool,
    slack_webhook,
    slack_bot_token,
    slack_channel: str,
    slack_log_level: str,
    metadata,
    skip_homing: bool,
) -> None:
    """
    Automated stack of X-Z beam cross-sections stepped along Y (the beam).

    Repeats for as many gantry placements as you like; each placement gets
    its own run directory with per-slice subfolders.
    """

    from auto_scan import AutoScanConfig, AutoScanSession
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
    from fluidnc_stage_cli import (
        check_bounds_against_limits,
        report_soft_limit_violations,
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

    # Validate every requested bound against the firmware's ACTUAL soft
    # limits (minus the keep-out margin) before anything moves — a
    # soft-limit alarm mid-scan loses machine position and aborts the run.
    firmware_limits = client.read_soft_limits(margin_mm=soft_limit_margin)

    if firmware_limits is None:
        click.secho(
            "Could not read firmware soft limits; validating against the "
            "conservative hardcoded envelope instead.",
            fg="yellow",
        )
    else:
        # The runtime stage validation should agree with what we verified.
        stage.config.MachineLimits_mm = firmware_limits

    effective_limits = stage.config.MachineLimits_mm
    click.echo(
        f"Motion envelope: X {effective_limits.x_min_mm:g}..{effective_limits.x_max_mm:g}  "
        f"Y {effective_limits.y_min_mm:g}..{effective_limits.y_max_mm:g}  "
        f"Z {effective_limits.z_min_mm:g}..{effective_limits.z_max_mm:g}"
    )

    x_min, x_max, z_min, z_max = resolve_scan_caps(
        effective_limits, x_min, x_max, z_min, z_max
    )
    click.echo(
        f"Raster caps: X {x_min:g}..{x_max:g}  Z {z_min:g}..{z_max:g}"
    )

    violations, replacements = check_bounds_against_limits(
        {
            "--x-min": (x_min, "x"),
            "--x-max": (x_max, "x"),
            "--z-min": (z_min, "z"),
            "--z-max": (z_max, "z"),
            "--y-start": (y_start, "y"),
            "--y-stop": (y_stop, "y"),
            "--calibration-x": (calibration_x_mm, "x"),
            "--calibration-z": (calibration_z_mm, "z"),
            "--background-x": (background_x_mm, "x"),
            "--background-z": (background_z_mm, "z"),
        },
        effective_limits,
    )

    if violations:
        client.close()
        report_soft_limit_violations(violations, effective_limits, replacements)

    if skip_background:
        background_mode = "none"

    from notify import slack_config_notice

    notice_level, notice = slack_config_notice(
        slack_bot_token, slack_webhook, slack_channel
    )
    logger.log(notice_level, notice)

    placement_number = 1
    current_placement_id = None
    current_thread = {"ts": None}  # Slack thread of the running placement

    def slack_notify(text: str, broadcast: bool = True) -> None:
        """Ping via bot token (into the placement thread) or webhook."""

        if slack_bot_token:
            from notify import post_message

            post_message(
                slack_bot_token,
                slack_channel,
                text,
                thread_ts=current_thread["ts"],
                reply_broadcast=broadcast,
            )
        elif slack_webhook:
            from notify import send_slack_message

            send_slack_message(text, slack_webhook)

    preview_process = None

    def stop_preview() -> None:
        nonlocal preview_process
        if preview_process is not None and preview_process.poll() is None:
            preview_process.terminate()
        preview_process = None

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
            current_placement_id = placement_id

            # Drive the camera to the measurement plane BEFORE prompting —
            # the prompt says "at machine Y = y_start", so the gantry must
            # actually be there. (Historic bug: with the old default
            # --y-start 10 the camera sat at home Y=3 and the ~7 mm error
            # went unnoticed; a descending scan starting at Y=130 would
            # have made it a 126 mm error.)
            click.echo(
                f"Moving the camera to the measurement plane "
                f"(machine Y = {y_start:g} mm)..."
            )
            client.move_machine(y_mm=y_start)

            click.echo(
                f"\nMeasure the distance from the optic to the camera sensor "
                f"along the beam, with the camera at machine Y = {y_start:g} mm "
                "(it is there now)."
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
                CalibrationX_mm=calibration_x_mm,
                CalibrationZ_mm=calibration_z_mm,
                FollowBeam=follow_beam,
                FindBeam=find_beam,
                RasterMode=raster_mode,
                SignalMargin_counts=signal_margin,
                MinSignalPixels=min_signal_pixels,
                NShots=nshots,
                BackgroundMode=background_mode,
                BackgroundX_mm=background_x_mm,
                BackgroundZ_mm=background_z_mm,
                BackgroundExposureChangeFraction=background_exposure_change,
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

            if camera_settings_path is None:
                # The default FLIRCameraSettings enable a 3 fps frame-rate
                # limiter, which is pointless (and a potential 6x slowdown)
                # in software-trigger mode. A user-provided settings file is
                # respected as-is.
                from dataclasses import replace as dataclass_replace

                # Rate must be None too: with the limiter disabled the
                # AcquisitionFrameRate node is read-only, and apply()
                # would raise trying to set it.
                camera_settings = dataclass_replace(
                    camera_settings,
                    AcquisitionFrameRateEnable=False,
                    AcquisitionFrameRate=None,
                )
                logger.info(
                    "No --camera-settings file: disabling the default 3 fps "
                    "frame-rate limiter for software-triggered acquisition."
                )

            writer = FLIRDatasetWriter(
                camera_index=camera_index,
                camera_serial=camera_serial,
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

            # Live viewer in its OWN process: it tails run_dir and shows
            # frames as they are saved. File-reads only — it cannot slow
            # or block this capture loop, and its death is harmless.
            if preview:
                import subprocess
                import sys

                preview_process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve().parent / "cli.py"),
                        "dataset",
                        "watch",
                        str(run_dir),
                    ]
                )
                logger.info(
                    f"Live preview watching {run_dir} "
                    f"(pid {preview_process.pid}; closing it does not "
                    "affect the scan)."
                )

            session = AutoScanSession(writer, config)

            # Everything logged during this placement (any module) also
            # lands in a scan.log next to the data.
            log_handler = add_file_log(run_dir / "scan.log")
            logger.info(f"Run directory: {run_dir}")

            # Bot token: open the placement's Slack thread and stream the
            # same log records into it as batched replies.
            slack_log_handler = None
            current_thread["ts"] = None

            if slack_bot_token:
                from log_utils import DATE_FORMAT, LOG_FORMAT
                from notify import SlackLogHandler, post_message

                current_thread["ts"] = post_message(
                    slack_bot_token,
                    slack_channel,
                    f":satellite_antenna: Scan with placement ID "
                    f"`{placement_id}` started (`{run_dir.name}`) — logs "
                    "stream in this thread.",
                )

                if current_thread["ts"] is not None:
                    slack_log_handler = SlackLogHandler(
                        slack_bot_token,
                        slack_channel,
                        current_thread["ts"],
                        level=getattr(logging, slack_log_level.upper()),
                    )
                    slack_log_handler.setFormatter(
                        logging.Formatter(LOG_FORMAT, DATE_FORMAT)
                    )
                    logging.getLogger().addHandler(slack_log_handler)

            try:
                records = session.run(stage.config.MachineLimits_mm)
                logger.info(
                    f"Placement done: {len(records)} frames in {run_dir}"
                )
            except Exception as ex:
                slack_notify(
                    f":rotating_light: Scan with placement ID "
                    f"`{placement_id}` FAILED: {ex}"
                )
                raise
            finally:
                stop_preview()
                remove_file_log(log_handler)

                if slack_log_handler is not None:
                    logging.getLogger().removeHandler(slack_log_handler)
                    slack_log_handler.close()  # final flush of queued lines

            slack_notify(
                f":bell: Scan with placement ID `{placement_id}` finished. "
                f"{len(records)} frames in `{run_dir.name}` — the gantry is "
                "ready to be repositioned."
            )

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

        which = f" `{current_placement_id}`" if current_placement_id else ""
        slack_notify(f":warning: Scan{which} interrupted — gantry is in feed hold.")
        click.echo(
            "Machine is holding. Resume from the WebUI (~) or re-home with "
            "'python cli.py gantry home'."
        )
        raise

    finally:
        stop_preview()  # covers interrupts before/around the scan body
        client.close()
