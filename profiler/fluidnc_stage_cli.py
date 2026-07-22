"""
Standalone gantry commands for sanity-checking FluidNC comms before a scan:

    python cli.py gantry status
    python cli.py gantry home
    python cli.py gantry move --x 60 --y 80 --z -50
    python cli.py gantry send '$SS'
"""

from __future__ import annotations

import click


def _common_options(fn):
    fn = click.option(
        "--host",
        default="fluidnc-sr2.local",
        show_default=True,
        help="FluidNC hostname or IP (Telnet port 23).",
    )(fn)
    fn = click.option("--port", default=23, show_default=True, type=int)(fn)
    return fn


def _make_client(host: str, port: int):
    from fluidnc_stage import FluidNCClient, FluidNCClientConfig

    return FluidNCClient(FluidNCClientConfig(Host=host, Port=port))


@click.group(name="gantry")
def gantry() -> None:
    """
    Talk to the FluidNC CNC gantry directly (no camera involved).
    """


@gantry.command("status")
@_common_options
def status(host: str, port: int) -> None:
    """Query and print the current machine state and position."""

    with _make_client(host, port) as client:
        report = client.query_status()
        click.echo(report.Raw)

        if report.is_alarm:
            click.echo("Machine is in ALARM state. Home it with: python cli.py gantry home")


@gantry.command("home")
@_common_options
@click.option(
    "--axes",
    default="",
    help="Optional axis subset, e.g. 'Z' for $HZ. Default homes all axes.",
)
def home(host: str, port: int, axes: str) -> None:
    """Run $H (Z retracts first, then X and Y auto-square). Machine will move!"""

    click.confirm(
        "The gantry will move to its limit switches. Area clear?", abort=True
    )

    with _make_client(host, port) as client:
        client.home(axes)
        report = client.query_status()
        click.echo(f"Homed. {report.Raw}")


@gantry.command("move")
@_common_options
@click.option("--x", "x_mm", type=float, default=None, help="Machine X target, mm.")
@click.option("--y", "y_mm", type=float, default=None, help="Machine Y target, mm.")
@click.option("--z", "z_mm", type=float, default=None, help="Machine Z target, mm (negative = toward the optics).")
@click.option("--feed", default=400.0, show_default=True, type=float, help="Feed, mm/min.")
@click.option(
    "--unsafe",
    is_flag=True,
    help="Skip the software machine-limit check (soft limits still apply).",
)
def move(host: str, port: int, x_mm, y_mm, z_mm, feed: float, unsafe: bool) -> None:
    """Absolute machine-coordinate move (G53 G1), waiting for completion."""

    from coordinates import Vec3D
    from fluidnc_stage import FluidNCStageConfig, FluidNCStageController

    if x_mm is None and y_mm is None and z_mm is None:
        raise click.ClickException("Provide at least one of --x / --y / --z.")

    with _make_client(host, port) as client:
        if not unsafe:
            current = client.query_status()

            if current.MPos is None:
                raise click.ClickException(
                    "No MPos in status report; cannot validate limits."
                )

            target = Vec3D(
                x_mm=x_mm if x_mm is not None else current.MPos.x_mm,
                y_mm=y_mm if y_mm is not None else current.MPos.y_mm,
                z_mm=z_mm if z_mm is not None else current.MPos.z_mm,
            )
            FluidNCStageController(client, FluidNCStageConfig()).validate_point(target)

        client.move_machine(x_mm=x_mm, y_mm=y_mm, z_mm=z_mm, feed_mm_min=feed)
        report = client.query_status()
        click.echo(f"Done. {report.Raw}")


@gantry.command("send")
@_common_options
@click.argument("line")
def send(host: str, port: int, line: str) -> None:
    """Send one raw G-code / $-command line and print the response."""

    with _make_client(host, port) as client:
        for reply in client.send_command(line, timeout_s=30.0):
            click.echo(reply)

        click.echo("ok")
