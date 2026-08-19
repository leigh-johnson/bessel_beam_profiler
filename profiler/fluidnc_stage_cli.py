"""
Standalone gantry commands for sanity-checking FluidNC comms before a scan:

    python cli.py gantry status
    python cli.py gantry home
    python cli.py gantry move --x 60 --y 80 --z -50
    python cli.py gantry send '$SS'
"""

from __future__ import annotations

import sys

import click


def rewrite_argv(argv: list[str], replacements: dict[str, float]) -> str:
    """
    Rebuild a command line with some flag values replaced (handles both
    '--flag value' and '--flag=value' forms; flags not present in argv are
    appended). Used to print a corrected, re-runnable command when the
    requested bounds exceed the machine's soft limits.
    """

    tokens = list(argv)
    seen: set[str] = set()
    i = 0

    while i < len(tokens):
        token = tokens[i]

        for flag, value in replacements.items():
            if token == flag and i + 1 < len(tokens):
                tokens[i + 1] = f"{value:g}"
                seen.add(flag)
            elif token.startswith(f"{flag}="):
                tokens[i] = f"{flag}={value:g}"
                seen.add(flag)

        i += 1

    for flag, value in replacements.items():
        if flag not in seen:
            tokens.extend([flag, f"{value:g}"])

    # argv[0] is the script path; present it the way it was invoked.
    return "python " + " ".join(tokens) if tokens and tokens[0].endswith(".py") else " ".join(tokens)


def report_soft_limit_violations(
    violations: dict[str, tuple[float, float]],
    limits,
    replacements: dict[str, float],
) -> None:
    """
    Print the out-of-range flags, the machine's limits, and a corrected
    command line, then abort with a non-zero exit code.
    """

    click.secho(
        "\nRequested bounds exceed the machine's soft limits:",
        fg="red",
        bold=True,
    )

    for flag, (value, clamped) in violations.items():
        click.echo(f"  {flag} {value:g}  ->  {clamped:g}")

    click.echo(
        f"\nSoft limits: X {limits.x_min_mm:g}..{limits.x_max_mm:g}  "
        f"Y {limits.y_min_mm:g}..{limits.y_max_mm:g}  "
        f"Z {limits.z_min_mm:g}..{limits.z_max_mm:g}"
    )

    click.secho("\nRe-run within limits:", bold=True)
    click.echo(f"  {rewrite_argv(sys.argv, replacements)}\n")

    raise SystemExit(1)


def check_bounds_against_limits(values: dict[str, float], limits) -> tuple[dict, dict]:
    """
    values maps CLI flag -> (requested value, axis letter). Returns
    (violations, replacements): violations maps flag -> (value, clamped);
    replacements maps flag -> clamped value, for rewrite_argv.
    """

    axis_ranges = {
        "x": (limits.x_min_mm, limits.x_max_mm),
        "y": (limits.y_min_mm, limits.y_max_mm),
        "z": (limits.z_min_mm, limits.z_max_mm),
    }

    violations: dict[str, tuple[float, float]] = {}
    replacements: dict[str, float] = {}

    for flag, (value, axis) in values.items():
        if value is None:
            continue

        lo, hi = axis_ranges[axis]

        if value < lo or value > hi:
            clamped = round(min(max(value, lo), hi), 3)
            violations[flag] = (value, clamped)
            replacements[flag] = clamped

    return violations, replacements


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
    Communicate with FluidNC CNC gantry controller via telnet.
    """


@gantry.command("status")
@_common_options
def status(host: str, port: int) -> None:
    """Query and print the current machine state and position."""

    with _make_client(host, port) as client:
        report = client.query_status()
        click.echo(report.Raw)

        limits = client.read_soft_limits()
        if limits is not None:
            click.echo(
                f"Soft limits: X {limits.x_min_mm:g}..{limits.x_max_mm:g}  "
                f"Y {limits.y_min_mm:g}..{limits.y_max_mm:g}  "
                f"Z {limits.z_min_mm:g}..{limits.z_max_mm:g}"
            )

        if report.Pins:
            click.secho(
                f"WARNING: input pins reading TRIGGERED: {report.Pins}. "
                "Switch pressed, or (NC wiring) a loose/disconnected signal "
                "wire — if the carriage is not at that switch, reseat its "
                "Dupont before homing.",
                fg="yellow",
            )

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

    from fluidnc_stage import DEFAULT_MACHINE_LIMITS

    if x_mm is None and y_mm is None and z_mm is None:
        raise click.ClickException("Provide at least one of --x / --y / --z.")

    with _make_client(host, port) as client:
        if not unsafe:
            limits = client.read_soft_limits()

            if limits is None:
                click.secho(
                    "Could not read firmware soft limits; validating against "
                    "the conservative hardcoded envelope instead.",
                    fg="yellow",
                )
                limits = DEFAULT_MACHINE_LIMITS

            violations, replacements = check_bounds_against_limits(
                {
                    "--x": (x_mm, "x"),
                    "--y": (y_mm, "y"),
                    "--z": (z_mm, "z"),
                },
                limits,
            )

            if violations:
                report_soft_limit_violations(violations, limits, replacements)

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
