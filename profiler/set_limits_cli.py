"""
`gantry set-limits` — interactively set each axis's soft-limit travel and
write the result back to the repo's config.yaml.

FluidNC's soft-limit range per axis is anchored at the homing side:

    positive-direction homing (Z): [mpos_mm - max_travel_mm, mpos_mm]
    negative-direction homing (X/Y): [mpos_mm, mpos_mm + max_travel_mm]

so the switch-side end is fixed by homing (mpos_mm is left untouched to
keep the machine coordinate frame stable), and what this tool sets is
max_travel_mm — i.e. where the FAR (away-from-switch) limit sits.

Per axis you can keep the current value, type the far-limit machine
coordinate directly, or jog the axis to the physical safe extreme and
mark it (soft limits are temporarily disabled for that axis while
jogging — small steps, eyes on the machine).

Runtime `$/...` changes are VOLATILE (lost on reboot), so the tool also
patches max_travel_mm in the local config.yaml (comment/format-preserving,
with a .bak backup). Upload that file to the controller (WebUI -> FluidNC
-> config.yaml) and reboot ($Bye) to persist.
"""

from __future__ import annotations

from pathlib import Path
import datetime as dt
import logging
import re

import click

logger = logging.getLogger(__name__)

AXES = ("x", "y", "z")


class LimitSettingError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Pure helpers (unit tested)
# ---------------------------------------------------------------------------


def far_limit_to_travel(
    mpos_mm: float, positive_direction: bool, far_limit_mm: float
) -> float:
    """
    max_travel_mm that places the far soft limit at far_limit_mm, given the
    homing anchor. Raises if the requested far limit is on the wrong side
    of (or on top of) the homing position.
    """

    travel = (mpos_mm - far_limit_mm) if positive_direction else (far_limit_mm - mpos_mm)

    if travel <= 0:
        side = "below" if positive_direction else "above"
        raise LimitSettingError(
            f"Far limit {far_limit_mm:g} must be {side} the homing position "
            f"{mpos_mm:g} (homing anchors that end of the range)."
        )

    return round(travel, 3)


def travel_to_range(
    mpos_mm: float, positive_direction: bool, travel_mm: float
) -> tuple[float, float]:
    if positive_direction:
        return (mpos_mm - travel_mm, mpos_mm)
    return (mpos_mm, mpos_mm + travel_mm)


def patch_max_travel(config_text: str, axis: str, travel_mm: float) -> str:
    """
    Replace the max_travel_mm value inside ONE axis block of a FluidNC
    config.yaml, preserving all other content, comments, and formatting.

    Axis blocks are the two-space-indented `  x:` / `  y:` / `  z:`
    sections under `axes:`; the first max_travel_mm line within the block
    (before the next axis or top-level key) is patched.
    """

    lines = config_text.splitlines(keepends=True)
    in_axis = False
    patched = False

    axis_header = re.compile(rf"^  {axis}:\s*$")
    other_block = re.compile(r"^(  \w|[A-Za-z_])")  # next 2-space key or top-level
    travel_line = re.compile(r"^(\s*max_travel_mm:\s*)[-\d.]+(\s*)$")

    for i, line in enumerate(lines):
        if axis_header.match(line):
            in_axis = True
            continue

        if in_axis and other_block.match(line) and not line.startswith("    "):
            in_axis = False  # left the axis block without finding the key

        if in_axis and not patched:
            match = travel_line.match(line)
            if match:
                lines[i] = f"{match.group(1)}{travel_mm:.6f}{match.group(2)}"
                patched = True

    if not patched:
        raise LimitSettingError(
            f"Could not find max_travel_mm inside the '{axis}:' block of "
            "config.yaml — file structure not recognized; not modifying it."
        )

    return "".join(lines)


# ---------------------------------------------------------------------------
# Interactive command
# ---------------------------------------------------------------------------


def _read_axis(client, axis: str) -> dict:
    travel_s = client.read_config_value(f"axes/{axis}/max_travel_mm")
    mpos_s = client.read_config_value(f"axes/{axis}/homing/mpos_mm")
    positive_s = client.read_config_value(f"axes/{axis}/homing/positive_direction")

    if travel_s is None or mpos_s is None or positive_s is None:
        raise click.ClickException(
            f"Could not read axis {axis.upper()} limit settings from the "
            "firmware."
        )

    return {
        "travel": float(travel_s),
        "mpos": float(mpos_s),
        "positive": positive_s.strip().lower() in ("true", "yes", "1"),
    }


def _describe(axis: str, info: dict) -> str:
    lo, hi = travel_to_range(info["mpos"], info["positive"], info["travel"])
    anchor = "max (homing side)" if info["positive"] else "min (homing side)"
    return (
        f"{axis.upper()}: range {lo:g} .. {hi:g} mm "
        f"(travel {info['travel']:g}, homing mpos {info['mpos']:g} = {anchor})"
    )


def _axis_position(client, axis: str) -> float:
    status = client.query_status()
    if status.MPos is None:
        raise click.ClickException("Status report has no MPos.")
    return getattr(status.MPos, f"{axis}_mm")


def _jog_for_far_limit(client, axis: str, info: dict, feed: float) -> float:
    """
    Soft limits OFF for this axis while the user jogs to the physical safe
    extreme; returns the marked machine coordinate. Always re-enables soft
    limits.
    """

    click.secho(
        f"\nSOFT LIMITS OFF for {axis.upper()} while you jog. The firmware "
        "will NOT stop you — use small steps and watch the machine.",
        fg="yellow",
        bold=True,
    )
    click.echo(
        "Commands: a number jogs that many mm (e.g. -5, 2.5), "
        "'m' marks the current position as the far limit, "
        "'s' shows status, 'q' aborts this axis."
    )

    client.set_config_value(f"axes/{axis}/soft_limits", "false")

    try:
        while True:
            position = _axis_position(client, axis)
            entry = click.prompt(
                f"{axis.upper()} @ {position:.3f} mm", type=str
            ).strip().lower()

            if entry == "m":
                marked = _axis_position(client, axis)
                if click.confirm(
                    f"Mark {marked:.3f} mm as the far soft limit for "
                    f"{axis.upper()}?",
                    default=True,
                ):
                    return marked
                continue

            if entry == "s":
                click.echo(client.query_status().Raw)
                continue

            if entry == "q":
                raise LimitSettingError("aborted by user")

            try:
                delta = float(entry)
            except ValueError:
                click.echo("Enter a number (mm), 'm', 's', or 'q'.")
                continue

            if abs(delta) > 20 and not click.confirm(
                f"Jog {delta:g} mm in one step with soft limits off?",
                default=False,
            ):
                continue

            try:
                client.jog_incremental(axis, delta, feed_mm_min=feed)
            except Exception as ex:  # noqa: BLE001 - report, keep session alive
                click.secho(f"Jog failed: {ex}", fg="red")

    finally:
        client.set_config_value(f"axes/{axis}/soft_limits", "true")
        click.echo(f"Soft limits re-enabled for {axis.upper()}.")


@click.command("set-limits")
@click.option("--host", default="fluidnc-sr2.local", show_default=True, help="FluidNC hostname or IP.")
@click.option("--port", default=23, show_default=True, type=int)
@click.option("--feed", default=150.0, show_default=True, type=float, help="Jog feed, mm/min (slow on purpose).")
@click.option(
    "--config",
    "config_path",
    default=Path("../cnc_gantry_mounts/config.yaml"),
    show_default=True,
    type=click.Path(path_type=Path),
    help="Local FluidNC config.yaml to patch with the new travels.",
)
def set_limits(host: str, port: int, feed: float, config_path: Path) -> None:
    """
    Interactively set each axis's soft-limit travel (keep / type the far
    limit / jog to find it), apply it live, and patch config.yaml.
    """

    from fluidnc_stage import FluidNCClient, FluidNCClientConfig

    client = FluidNCClient(FluidNCClientConfig(Host=host, Port=port))
    client.connect()

    try:
        status = client.query_status()
        click.echo(f"FluidNC: {status.Raw}")

        if status.is_alarm:
            click.confirm(
                "Machine is in ALARM (not homed). Home now ($H)? Area clear?",
                abort=True,
            )
            client.home()
            click.echo(f"Homed: {client.query_status().Raw}")

        new_travels: dict[str, float] = {}

        for axis in AXES:
            info = _read_axis(client, axis)
            click.echo(f"\n{_describe(axis, info)}")

            choice = click.prompt(
                "  [k]eep / [t]ype far limit / [j]og to find it",
                type=click.Choice(["k", "t", "j"]),
                default="k",
                show_default=True,
            )

            if choice == "k":
                continue

            try:
                if choice == "t":
                    far = click.prompt(
                        f"  Far soft limit for {axis.upper()} (machine mm)",
                        type=float,
                    )
                else:
                    far = _jog_for_far_limit(client, axis, info, feed)

                travel = far_limit_to_travel(info["mpos"], info["positive"], far)
            except LimitSettingError as ex:
                click.secho(f"  Skipping {axis.upper()}: {ex}", fg="yellow")
                continue

            lo, hi = travel_to_range(info["mpos"], info["positive"], travel)

            if not click.confirm(
                f"  Set {axis.upper()} travel to {travel:g} mm "
                f"(range {lo:g} .. {hi:g})?",
                default=True,
            ):
                continue

            client.set_config_value(f"axes/{axis}/max_travel_mm", f"{travel:.3f}")
            applied = client.read_config_value(f"axes/{axis}/max_travel_mm")
            click.echo(f"  Applied live: max_travel_mm={applied}")
            new_travels[axis] = travel

        if not new_travels:
            click.echo("\nNo changes made.")
            return

        limits = client.read_soft_limits()
        if limits is not None:
            click.echo(
                f"\nNew firmware envelope: "
                f"X {limits.x_min_mm:g}..{limits.x_max_mm:g}  "
                f"Y {limits.y_min_mm:g}..{limits.y_max_mm:g}  "
                f"Z {limits.z_min_mm:g}..{limits.z_max_mm:g}"
            )

        # -- persist to the local config.yaml ---------------------------

        if not config_path.exists():
            click.secho(
                f"\n{config_path} not found — live values are applied but "
                "VOLATILE. Patch max_travel_mm manually: "
                + ", ".join(f"{a}={v:g}" for a, v in new_travels.items()),
                fg="yellow",
            )
            return

        text = config_path.read_text()

        for axis, travel in new_travels.items():
            text = patch_max_travel(text, axis, travel)

        backup = config_path.with_suffix(
            f".yaml.bak-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        backup.write_text(config_path.read_text())
        config_path.write_text(text)

        click.echo(f"\nPatched {config_path} (backup: {backup.name}):")
        for axis, travel in new_travels.items():
            click.echo(f"  axes/{axis}/max_travel_mm: {travel:.6f}")

        click.secho(
            "\nThe live values are active NOW but are lost on reboot. To "
            "persist: upload the patched config.yaml to the controller "
            "(WebUI at fluidnc-sr2.local -> file manager -> replace "
            "config.yaml), then reboot with $Bye and re-home.",
            bold=True,
        )

    finally:
        client.close()
