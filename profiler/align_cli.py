"""
`align` — live axicon alignment feedback on the FluidNC gantry.

Finds the annulus after axicon 3 (find-beam sweep + chord survey), then
patrols stations around it and shows a live preview window with the
fitted ring, center offset vs a reference, roundness, azimuthal
uniformity, and (with --y2) the two-plane pointing tilt. Tweak the
optic, watch the numbers move, press r to zero the readout when happy.

Every run writes an alignment_log.jsonl (one line of metrics per cycle)
plus a preview_latest.png snapshot into a timestamped run directory —
the before/after record of each alignment session.
"""

from __future__ import annotations

from pathlib import Path
import logging

import click

from log_utils import add_file_log, remove_file_log

logger = logging.getLogger(__name__)


@click.command("align")
# -- connection -------------------------------------------------------------
@click.option("--host", default="fluidnc-sr2.local", show_default=True, help="FluidNC hostname or IP.")
@click.option("--port", default=23, show_default=True, type=int)
@click.option("--feed", default=400.0, show_default=True, type=float, help="Feed rate for patrol moves, mm/min.")
@click.option(
    "--soft-limit-margin",
    default=1.0,
    show_default=True,
    type=click.FloatRange(min=0.0),
    help="Keep-out margin (mm) inside the firmware's soft limits.",
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
# -- alignment geometry -----------------------------------------------------
@click.option("--y", "machine_y", default=20.0, show_default=True, type=float, help="Machine Y plane to patrol (near the optic keeps the ring small).")
@click.option("--y2", "machine_y2", default=None, type=float, help="Second machine Y plane: enables the two-plane pointing-tilt readout (cycles alternate between planes).")
@click.option(
    "--beam-direction",
    default="-y",
    show_default=True,
    type=click.Choice(["+y", "-y"]),
    help="-y if machine +Y points TOWARD the optic (rig default, verified "
    "2026-07-22). Only the tilt sign depends on this.",
)
@click.option(
    "--notes",
    multiple=True,
    help="Free-text note recorded in the run's align_session.json (e.g. "
    "\"camera y2 at L_12=290mm\"). Repeatable: pass --notes several "
    "times to record several notes.",
)
@click.option(
    "--optic",
    default="axicon3",
    show_default=True,
    help="Which optic the ring is being observed after (e.g. axicon2, "
    "axicon3) — a label recorded in the run's align_session.json. The "
    "tool itself is optic-agnostic: it aligns whatever ring it finds.",
)
@click.option("--probe-x", default=None, type=float, help="X of the find-beam sweep and chord-survey columns. Default: center of the machine X envelope. Must cross the ring.")
@click.option("--stations", default=8, show_default=True, type=click.IntRange(min=3), help="Patrol stations per cycle around the ring.")
@click.option(
    "--cover",
    default="ring",
    show_default=True,
    type=click.Choice(["ring", "disk"]),
    help="ring: image only stations on the fitted ring (fastest; a thin "
    "annulus has nothing inside). disk: also image a half-radius inner "
    "ring + the center so the composite has no blind spot in the middle "
    "(interior frames are display-only — excluded from the ring fit).",
)
@click.option("--ring-diameter", default=None, type=float, help="Expected ring diameter (mm) at the --y plane — sanity bound for the chord survey, not required.")
@click.option("--ring-diameter2", default=None, type=float, help="Expected ring diameter (mm) at the --y2 plane, when it differs from --ring-diameter (diverging cone, e.g. after axicon 1). Defaults to --ring-diameter.")
@click.option("--survey-dx", default=5.0, show_default=True, type=float, help="X offset between the two bootstrap survey columns.")
@click.option("--max-shift", default=3.0, show_default=True, type=click.FloatRange(min=0.1), help="Max ring-estimate change per lap (mm) after the first fit; raise for beams you expect to move a lot per adjustment.")
@click.option("--max-exposure", default=None, type=click.FloatRange(min=100.0), help="Hard exposure ceiling (us). Dim beams otherwise calibrate to very long exposures; the background-referenced threshold detects dim rings fine, so capping (e.g. 100000) buys lap/stream speed.")
# -- signal / imaging -------------------------------------------------------
@click.option("--downsample", default=8, show_default=True, type=click.IntRange(min=1), help="Mean-pool factor before analysis (8 -> 27.6 um/px).")
@click.option("--signal-margin", default=8.0, show_default=True, type=float, help="Counts above the frame median that count as ring signal.")
@click.option("--min-signal-pixels", default=30, show_default=True, type=click.IntRange(min=1), help="Lit pixels needed to call a station frame 'signal'.")
# -- loop -------------------------------------------------------------------
@click.option(
    "--mode",
    default="stream",
    show_default=True,
    type=click.Choice(["stream", "patrol"]),
    help="stream: find the ring, orbit it once, then PARK and stream "
    "single frames at a few Hz (center drift + ring width live; press o "
    "for a fresh orbit lap, f to re-find). patrol: orbit continuously, "
    "refreshing ALL metrics every lap (~10-20 s).",
)
@click.option("--park-azimuth", default=None, type=float, help="Stream mode: ring azimuth (deg, 0=+X, 90=+Z/up) to park at. Default: the brightest station of the last orbit.")
@click.option("--orbit-every", default=0.0, show_default=True, type=click.FloatRange(min=0.0), help="Stream mode: also run a full orbit lap every N seconds (0 = only on demand via o).")
@click.option("--frames", "max_frames", default=None, type=click.IntRange(min=1), help="Stream mode: stop after this many streamed frames.")
@click.option("--cycles", default=None, type=click.IntRange(min=1), help="Patrol mode: stop after this many cycles. Default: run until q/window close.")
@click.option("--interval", default=0.0, show_default=True, type=click.FloatRange(min=0.0), help="Patrol mode: minimum seconds per cycle (0 = free-running).")
# -- output -----------------------------------------------------------------
@click.option(
    "--dataset-root",
    default=Path("data/align-runs"),
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Root directory; one timestamped run directory per session "
    "(alignment_log.jsonl + preview snapshots + align log).",
)
@click.option("--no-display", is_flag=True, help="Headless: no window (metrics still logged, snapshots still saved). Useful over SSH.")
@click.option("--skip-homing", is_flag=True, help="Skip $H (only if already homed this power-cycle).")
def align(
    host: str,
    port: int,
    feed: float,
    soft_limit_margin: float,
    camera_settings_path,
    camera_index: int,
    camera_serial,
    acquisition_timeout_ms: int,
    machine_y: float,
    machine_y2,
    beam_direction: str,
    notes: tuple,
    optic: str,
    probe_x,
    stations: int,
    cover: str,
    ring_diameter,
    ring_diameter2,
    survey_dx: float,
    max_shift: float,
    max_exposure,
    downsample: int,
    signal_margin: float,
    min_signal_pixels: int,
    mode: str,
    park_azimuth,
    orbit_every: float,
    max_frames,
    cycles,
    interval: float,
    dataset_root: Path,
    no_display: bool,
    skip_homing: bool,
) -> None:
    """
    Live axicon-alignment feedback: patrol the ring after axicon 3 and
    watch center offset / roundness / azimuthal uniformity update as
    you adjust the optic.
    """

    from align_axicon import (
        STREAM_CONTINUE,
        STREAM_ORBIT,
        STREAM_REFIND,
        STREAM_STOP,
        AlignConfig,
        AlignError,
        AxiconAlignSession,
        append_cycle_log,
    )
    from align_preview import AlignPreview
    from dataset_writer import DatasetWriterConfig, FLIRDatasetWriter
    from dataset_writer_cli import _load_camera_settings_for_software_trigger
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

    click.echo(f"Connecting to FluidNC at {host}:{port} ...")
    client = FluidNCClient(FluidNCClientConfig(Host=host, Port=port, Feed_mm_min=feed))
    client.connect()
    stage = FluidNCStageController(client, FluidNCStageConfig(Feed_mm_min=feed))

    status = client.query_status()
    click.echo(f"FluidNC: {status.Raw}")

    firmware_limits = client.read_soft_limits(margin_mm=soft_limit_margin)
    if firmware_limits is None:
        click.secho(
            "Could not read firmware soft limits; validating against the "
            "conservative hardcoded envelope instead.",
            fg="yellow",
        )
    else:
        stage.config.MachineLimits_mm = firmware_limits

    limits = stage.config.MachineLimits_mm
    click.echo(
        f"Motion envelope: X {limits.x_min_mm:g}..{limits.x_max_mm:g}  "
        f"Y {limits.y_min_mm:g}..{limits.y_max_mm:g}  "
        f"Z {limits.z_min_mm:g}..{limits.z_max_mm:g}"
    )

    violations, replacements = check_bounds_against_limits(
        {
            "--y": (machine_y, "y"),
            "--y2": (machine_y2, "y"),
            "--probe-x": (probe_x, "x"),
        },
        limits,
    )
    if violations:
        client.close()
        report_soft_limit_violations(violations, limits, replacements)

    try:
        if not skip_homing or status.is_alarm:
            click.confirm(
                "About to home ($H): the gantry will move to its limit "
                "switches. Area clear?",
                abort=True,
            )
            click.echo("Homing...")
            client.home()
            click.echo(f"Homed: {client.query_status().Raw}")

        camera_settings = _load_camera_settings_for_software_trigger(
            camera_settings_path
        )
        if camera_settings_path is None:
            # Same rationale as `dataset auto`: the default 3 fps limiter
            # only slows software-triggered acquisition down.
            from dataclasses import replace as dataclass_replace

            camera_settings = dataclass_replace(
                camera_settings,
                AcquisitionFrameRateEnable=False,
                AcquisitionFrameRate=None,
            )

        writer = FLIRDatasetWriter(
            camera_index=camera_index,
            camera_serial=camera_serial,
            camera_settings=camera_settings,
            config=DatasetWriterConfig(
                JobType="align",
                DatasetRoot=dataset_root,
                AcquisitionTimeout_ms=acquisition_timeout_ms,
            ),
            stage_controller=stage,
        )
        run_dir = writer.prepare_run()
        log_handler = add_file_log(run_dir / "align.log")
        logger.info(f"Alignment run directory: {run_dir} (optic: {optic})")

        writer.write_json_artifact(
            "align_session.json",
            {
                "Optic": optic,
                "Notes": list(notes),
                "Mode": mode,
                "MachineY_mm": machine_y,
                "MachineY2_mm": machine_y2,
                "BeamDirection": beam_direction,
                "Stations": stations,
                "RingDiameterPrior_mm": ring_diameter,
            },
        )
        if notes:
            for note in notes:
                logger.info(f"Run note: {note}")

        config = AlignConfig(
            MachineY_mm=machine_y,
            MachineY2_mm=machine_y2,
            BeamDirectionSign=1 if beam_direction == "+y" else -1,
            ProbeX_mm=probe_x,
            Stations=stations,
            CoverMode=cover,
            RingDiameter_mm=ring_diameter,
            RingDiameter2_mm=ring_diameter2,
            SurveyDX_mm=survey_dx,
            MaxRingShift_mm=max_shift,
            MaxExposure_us=max_exposure,
            Downsample=downsample,
            SignalMargin_counts=signal_margin,
            MinSignalPixels=min_signal_pixels,
            CycleInterval_s=interval,
        )

        session = AxiconAlignSession(writer, config, limits)
        preview = AlignPreview(config, display=not no_display)
        cycle_log = run_dir / "alignment_log.jsonl"

        if mode == "stream" and machine_y2 is not None:
            click.secho(
                "Note: the two-plane tilt readout needs --mode patrol "
                "(stream mode parks at one Y plane).",
                fg="yellow",
            )

        def handle_reference() -> None:
            if preview.reference_requested:
                preview.reference_requested = False
                session.set_reference_here()

        def on_cycle(result, frames) -> bool:
            handle_reference()
            append_cycle_log(cycle_log, result)
            preview.update(result, frames)
            preview.save_png(run_dir / "preview_latest.png")
            return not preview.quit_requested

        click.echo(
            "\nStarting alignment. In the preview window: r = set "
            "reference (zero the offset), o = orbit lap, f = re-find, "
            "q = quit."
        )

        try:
            if mode == "patrol":
                results = session.run(
                    on_cycle, max_cycles=cycles, on_station=preview.on_station
                )
                logger.info(
                    f"Alignment session done: {len(results)} cycles, "
                    f"metrics in {cycle_log}"
                )
            else:

                def on_orbit(result, frames) -> None:
                    on_cycle(result, frames)

                def on_frame(sample, frame) -> str:
                    handle_reference()
                    preview.update_stream(sample, frame)
                    if sample.Index % 20 == 0:
                        preview.save_png(run_dir / "preview_latest.png")
                    if preview.quit_requested:
                        return STREAM_STOP
                    if preview.orbit_requested:
                        preview.orbit_requested = False
                        return STREAM_ORBIT
                    if preview.refind_requested:
                        preview.refind_requested = False
                        return STREAM_REFIND
                    return STREAM_CONTINUE

                session.run_stream(
                    on_frame,
                    on_cycle=on_orbit,
                    park_azimuth_deg=park_azimuth,
                    orbit_every_s=orbit_every,
                    max_frames=max_frames,
                )
                logger.info(f"Alignment stream done: metrics in {cycle_log}")
        except AlignError as ex:
            logger.error(f"Alignment failed: {ex}")
            raise click.ClickException(str(ex))
        finally:
            session.close()
            preview.save_png(run_dir / "preview_final.png")
            preview.close()
            remove_file_log(log_handler)

    except KeyboardInterrupt:
        click.echo("\nInterrupted — sending feed hold (!) to the gantry.")
        try:
            client.feed_hold()
        except Exception as ex:  # noqa: BLE001 - best-effort safety stop
            click.echo(f"Feed hold failed: {ex}")
        click.echo(
            "Machine is holding. Resume from the WebUI (~) or re-home "
            "with 'python cli.py gantry home'."
        )
        raise

    finally:
        client.close()
