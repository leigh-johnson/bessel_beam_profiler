"""
`preflight` — sanity-check the scan rig before a dataset session.

Safe by default: without --motion nothing moves. Checks are grouped:

    software  : Python env, required imports, profiler modules, disk space
    gantry    : DNS/TCP to FluidNC, status/alarm state, firmware info
    camera    : PySpin enumeration, exposure limits, frame-rate throttle,
                optional free-run frame grabs with per-frame latency
    motion    : (--motion only, confirms first) homing + interactive
                beam-direction verification (machine +Y vs downstream)

Exit code is non-zero if any check FAILs, so this can gate a scan script.
"""

from __future__ import annotations

from pathlib import Path
import importlib
import shutil
import socket
import sys
import time

import click


class Tally:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warned = 0
        self.skipped = 0

    def ok(self, message: str) -> None:
        self.passed += 1
        click.secho(f"[ OK ] {message}", fg="green")

    def fail(self, message: str) -> None:
        self.failed += 1
        click.secho(f"[FAIL] {message}", fg="red", bold=True)

    def warn(self, message: str) -> None:
        self.warned += 1
        click.secho(f"[WARN] {message}", fg="yellow")

    def skip(self, message: str) -> None:
        self.skipped += 1
        click.secho(f"[SKIP] {message}", fg="cyan")

    @staticmethod
    def note(message: str) -> None:
        click.echo(f"       {message}")


def _section(title: str) -> None:
    click.secho(f"\n--- {title} ---", bold=True)


# ---------------------------------------------------------------------------
# Software checks
# ---------------------------------------------------------------------------


def check_software(tally: Tally, dataset_root: Path) -> None:
    _section("Software")

    tally.note(f"Python {sys.version.split()[0]} at {sys.executable}")

    for module_name, why in (
        ("numpy", "arrays"),
        ("click", "CLI"),
        ("matplotlib", "previews/composites"),
        ("cv2", "JPG previews"),
        ("PySpin", "FLIR camera — wrong Python env if missing"),
    ):
        try:
            importlib.import_module(module_name)
            tally.ok(f"import {module_name}")
        except Exception as ex:  # noqa: BLE001 - report any import problem
            tally.fail(f"import {module_name} ({why}): {ex}")

    for module_name in (
        "fluidnc_stage",
        "adaptive_raster",
        "headless_calibration",
        "log_utils",
        "auto_scan",
        "dataset_writer",
    ):
        try:
            importlib.import_module(module_name)
            tally.ok(f"import profiler module {module_name}")
        except Exception as ex:  # noqa: BLE001
            tally.fail(f"import profiler module {module_name}: {ex}")

    # Disk space where the dataset will land.
    probe = dataset_root
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent

    usage = shutil.disk_usage(probe)
    free_gb = usage.free / 1e9

    if free_gb < 5:
        tally.fail(f"only {free_gb:.1f} GB free at {probe} for --dataset-root {dataset_root}")
    elif free_gb < 20:
        tally.warn(
            f"{free_gb:.1f} GB free at {probe} — a full placement can be "
            "several GB; consider clearing space"
        )
    else:
        tally.ok(f"{free_gb:.0f} GB free at {probe} for --dataset-root {dataset_root}")


# ---------------------------------------------------------------------------
# Gantry checks
# ---------------------------------------------------------------------------


def check_gantry(tally: Tally, host: str, port: int):
    """Returns a connected FluidNCClient (for --motion) or None."""

    _section("Gantry (FluidNC)")

    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        ip = infos[0][4][0]
        tally.ok(f"resolved {host} -> {ip}")
    except OSError as ex:
        tally.fail(
            f"cannot resolve {host}: {ex}. If mDNS is flaky, retry with "
            "--host 10.43.19.86"
        )
        return None

    from fluidnc_stage import FluidNCClient, FluidNCClientConfig, FluidNCError

    client = FluidNCClient(FluidNCClientConfig(Host=host, Port=port))

    try:
        client.connect()
        tally.ok(f"TCP connection to {host}:{port}")
    except (FluidNCError, OSError) as ex:
        tally.fail(f"cannot connect to {host}:{port}: {ex}")
        return None

    try:
        status = client.query_status()
        tally.note(f"status: {status.Raw}")

        if status.is_alarm:
            tally.warn(
                "machine is in ALARM (expected after power-up: must_home). "
                "Home it: python cli.py gantry home  (or preflight --motion)"
            )
        elif status.is_idle:
            tally.ok("machine is Idle")
        else:
            tally.warn(f"machine is {status.State} (not Idle)")

        if status.MPos is not None:
            tally.ok(
                f"MPos X{status.MPos.x_mm:g} Y{status.MPos.y_mm:g} "
                f"Z{status.MPos.z_mm:g}"
            )
    except FluidNCError as ex:
        tally.fail(f"status query failed: {ex}")
        client.close()
        return None

    try:
        build_lines = client.send_command("$I", timeout_s=5.0)
        for line in build_lines[:3]:
            tally.note(line)
        tally.ok("firmware build info ($I)")
    except FluidNCError as ex:
        tally.warn(f"$I build info failed (non-critical): {ex}")

    return client


def check_motion(tally: Tally, client) -> None:
    """Homing + interactive beam-direction verification. MOVES THE GANTRY."""

    _section("Motion (gantry WILL move)")

    from fluidnc_stage import FluidNCError

    if not click.confirm(
        "Home the machine now ($H — travels to limit switches)? Area clear?",
        default=True,
    ):
        tally.skip("homing")
        return

    try:
        click.echo("Homing (Z retracts first, then X and Y auto-square)...")
        client.home()
        status = client.query_status()
        tally.ok(f"homed: {status.Raw}")
    except FluidNCError as ex:
        tally.fail(f"homing failed: {ex}")
        return

    if not click.confirm(
        "Run the beam-direction check (moves to X60 Y20 Z-60, then Y+30)?",
        default=True,
    ):
        tally.skip("beam-direction check")
        return

    try:
        client.move_machine(x_mm=60.0, y_mm=20.0, z_mm=-60.0)
        click.echo("At X60 Y20 Z-60. Watch the camera; moving +30 mm in Y...")
        time.sleep(1.0)
        client.move_machine(y_mm=50.0)
    except FluidNCError as ex:
        tally.fail(f"beam-direction moves failed: {ex}")
        return

    moved_away = click.confirm(
        "Did the camera move AWAY from the optic (downstream)?", default=True
    )

    if moved_away:
        tally.ok("machine +Y is downstream: use --beam-direction +y (the default)")
    else:
        tally.warn(
            "machine +Y points TOWARD the optic: pass --beam-direction -y "
            "to dataset auto"
        )


# ---------------------------------------------------------------------------
# Camera checks
# ---------------------------------------------------------------------------


def check_camera(tally: Tally, camera_index: int, grab: bool) -> None:
    _section("Camera (FLIR / PySpin)")

    try:
        PySpin = importlib.import_module("PySpin")
    except Exception as ex:  # noqa: BLE001
        tally.fail(f"PySpin not importable: {ex}")
        return

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    try:
        n_cameras = cam_list.GetSize()

        if n_cameras == 0:
            tally.fail(
                "no FLIR cameras detected. Check the GigE link — and make "
                "sure SpinView is CLOSED (it holds the camera)."
            )
            return

        tally.ok(f"{n_cameras} camera(s) detected")

        if camera_index >= n_cameras:
            tally.fail(f"--camera-index {camera_index} but only {n_cameras} present")
            return

        cam = cam_list.GetByIndex(camera_index)
        cam.Init()

        try:
            try:
                model = cam.TLDevice.DeviceModelName.GetValue()
                serial = cam.TLDevice.DeviceSerialNumber.GetValue()
                tally.ok(f"camera {camera_index}: {model} (SN {serial})")
            except Exception as ex:  # noqa: BLE001
                tally.warn(f"could not read model/serial: {ex}")

            try:
                exp_lo = cam.ExposureTime.GetMin()
                exp_hi = cam.ExposureTime.GetMax()
                tally.ok(f"exposure range {exp_lo:.1f} .. {exp_hi:.0f} us")
            except Exception as ex:  # noqa: BLE001
                tally.warn(f"could not read exposure limits: {ex}")

            try:
                rate_enabled = bool(cam.AcquisitionFrameRateEnable.GetValue())
                rate = float(cam.AcquisitionFrameRate.GetValue())

                if rate_enabled and rate < 10.0:
                    tally.warn(
                        f"AcquisitionFrameRate is {rate:g} fps with the "
                        "limiter ENABLED — if triggered frames honor it, "
                        "every frame waits up to "
                        f"{1000.0 / rate:.0f} ms. Verify with the grab "
                        "timing below; raise it in camera settings if slow."
                    )
                else:
                    tally.ok(
                        f"frame-rate limiter: enabled={rate_enabled}, "
                        f"{rate:g} fps"
                    )
            except Exception as ex:  # noqa: BLE001
                tally.warn(f"could not read frame-rate nodes: {ex}")

            if grab:
                _grab_frames(tally, PySpin, cam)
            else:
                tally.skip("frame grabs (--no-grab)")

        finally:
            try:
                if cam.IsStreaming():
                    cam.EndAcquisition()
            except Exception:  # noqa: BLE001
                pass
            cam.DeInit()
            del cam

    finally:
        cam_list.Clear()
        system.ReleaseInstance()


def _grab_frames(tally: Tally, PySpin, cam, n_frames: int = 3) -> None:
    """
    Free-run grab a few frames and time them — measures real per-frame
    latency (network transfer + any frame-rate throttle) with whatever
    settings the camera currently has.
    """

    try:
        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
        cam.BeginAcquisition()

        try:
            # Discard the first frame (pipeline warm-up).
            image = cam.GetNextImage(5000)
            image.Release()

            durations = []
            last_shape, last_max = None, None

            for _ in range(n_frames):
                started = time.monotonic()
                image = cam.GetNextImage(5000)

                try:
                    if not image.IsIncomplete():
                        arr = image.GetNDArray()
                        last_shape = arr.shape
                        last_max = int(arr.max())
                    durations.append(time.monotonic() - started)
                finally:
                    image.Release()
                    image = None

            mean_ms = 1000.0 * sum(durations) / len(durations)

            tally.ok(
                f"grabbed {n_frames} frames: shape {last_shape}, "
                f"max {last_max}, mean {mean_ms:.0f} ms/frame"
            )

            if mean_ms > 250:
                tally.warn(
                    f"{mean_ms:.0f} ms/frame is slow — check the frame-rate "
                    "limiter above and prefer wired GigE over WiFi for the "
                    "camera link."
                )

            if last_max == 0:
                tally.note(
                    "frame max is 0 — lens cap on, or no light. Fine for "
                    "preflight; remember to find the beam before scanning."
                )

        finally:
            cam.EndAcquisition()

    except Exception as ex:  # noqa: BLE001
        tally.fail(f"frame grab failed: {ex}")


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


@click.command("preflight")
@click.option("--host", default="fluidnc-sr2.local", show_default=True, help="FluidNC hostname or IP.")
@click.option("--port", default=23, show_default=True, type=int)
@click.option("--camera-index", default=0, show_default=True, type=click.IntRange(min=0))
@click.option(
    "--dataset-root",
    default=Path("data"),
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Where scans will be written (disk-space check).",
)
@click.option("--skip-gantry", is_flag=True, help="Skip FluidNC checks.")
@click.option("--skip-camera", is_flag=True, help="Skip camera checks.")
@click.option(
    "--grab/--no-grab",
    default=True,
    show_default=True,
    help="Grab test frames to measure per-frame latency.",
)
@click.option(
    "--motion",
    is_flag=True,
    help="Include checks that MOVE the gantry: homing + interactive "
    "beam-direction verification (asks before each move).",
)
def preflight(
    host: str,
    port: int,
    camera_index: int,
    dataset_root: Path,
    skip_gantry: bool,
    skip_camera: bool,
    grab: bool,
    motion: bool,
) -> None:
    """
    Sanity-check software, gantry, and camera before a scan session.

    Safe by default (nothing moves without --motion). Exits non-zero if
    any check fails.
    """

    tally = Tally()

    check_software(tally, dataset_root)

    client = None

    if skip_gantry:
        _section("Gantry (FluidNC)")
        tally.skip("gantry checks (--skip-gantry)")
    else:
        client = check_gantry(tally, host, port)

        if motion and client is not None:
            check_motion(tally, client)
        elif motion:
            tally.skip("motion checks (no gantry connection)")

    if client is not None:
        client.close()

    if skip_camera:
        _section("Camera (FLIR / PySpin)")
        tally.skip("camera checks (--skip-camera)")
    else:
        check_camera(tally, camera_index, grab)

    _section("Summary")
    click.echo(
        f"{tally.passed} ok, {tally.warned} warnings, "
        f"{tally.failed} failed, {tally.skipped} skipped"
    )

    if tally.failed:
        click.secho("NOT ready — fix the failures above.", fg="red", bold=True)
        raise SystemExit(1)

    if tally.warned:
        click.secho(
            "Ready with warnings — read them before scanning.",
            fg="yellow",
        )
    else:
        click.secho("All clear.", fg="green", bold=True)
