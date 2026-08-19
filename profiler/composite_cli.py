"""
`dataset composite` — place one slice's frames at their commanded gantry
positions and average the overlaps. Position-based rather than
registration-based, so it handles the adaptive raster's dark perimeter
frames that registration cannot.

Accepts any mix of slice folders, run folders, and glob patterns (a
shell-expanded `data/auto_scan-2026-07-29_*` or the same pattern quoted).
Run folders composite every subdirectory matching --match-pattern
(default y*cm). With --commit, each run directory that composites
cleanly gets its own git commit `composite: <run_dir>/`.
"""

from __future__ import annotations

import glob as glob_module
import subprocess
from pathlib import Path

import click


def expand_targets(raw_targets: tuple[str, ...]) -> list[Path]:
    """
    Each argument is either an existing directory or a glob pattern that
    matches at least one directory (patterns are expanded sorted, so runs
    composite and commit in timestamp order). Duplicates — e.g. a shell
    glob overlapping an explicit path — are dropped, keeping first
    occurrence order.
    """

    resolved: list[Path] = []
    for raw in raw_targets:
        path = Path(raw)
        if path.is_dir():
            resolved.append(path)
            continue
        matches = sorted(
            Path(m) for m in glob_module.glob(raw) if Path(m).is_dir()
        )
        if not matches:
            raise click.UsageError(
                f"{raw!r} is not a directory and matches no directories "
                "as a glob pattern."
            )
        resolved.extend(matches)
    return list(dict.fromkeys(resolved))


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    )


def commit_run(run_dir: Path) -> None:
    """
    Stage RUN_DIR and commit it with message `composite: <run_dir>/`.
    Both the add and the commit are pathspec-limited to RUN_DIR, so
    unrelated staged or dirty files elsewhere in the repo are untouched.
    If RUN_DIR has nothing new, says so and commits nothing.
    Raises subprocess.CalledProcessError / OSError on git failure.
    """

    _git("add", "--", str(run_dir))
    status = _git("status", "--porcelain", "--", str(run_dir))
    if not status.stdout.strip():
        click.echo(f"{run_dir}: nothing new to commit.")
        return

    message = f"composite: {run_dir.as_posix()}/"
    result = _git("commit", "-m", message, "--", str(run_dir))
    summary = result.stdout.strip().splitlines()
    click.echo(summary[0] if summary else f"Committed: {message}")


@click.command("composite")
@click.argument("targets", nargs=-1, metavar="TARGET_DIR...")
@click.option(
    "--match-pattern",
    default="y*cm",
    show_default=True,
    help="When a target is a run directory rather than a single slice, "
    "composite every subdirectory matching this glob (sorted). A "
    "directory with no matching subdirectories is treated as a single "
    "slice, so existing single-slice invocations are unchanged.",
)
@click.option(
    "--commit",
    is_flag=True,
    help="After each target directory composites with no failures, git-add "
    "that directory and create a commit `composite: <dir>/` limited to it "
    "(one commit per run). A target with a failed slice is left "
    "uncommitted.",
)
@click.option("--pixel-size-um", default=3.45, show_default=True, type=float, help="Camera pixel pitch (BFS-PGE-31S4M: 3.45 um).")
@click.option("--downsample", default=8, show_default=True, type=click.IntRange(min=1), help="Mean-pool factor before placement.")
@click.option("--flip-x/--no-flip-x", "flip_x", default=True, show_default=True, help="Mirror frames horizontally. Default True: camera verified 180-deg rotated vs machine axes (2026-07-22).")
@click.option("--flip-z/--no-flip-z", "flip_z", default=True, show_default=True, help="Mirror frames vertically (see --flip-x).")
@click.option("--transpose", is_flag=True, help="Swap image axes (camera rotated 90 degrees).")
@click.option("--subtract/--no-subtract", "subtract_background", default=True, show_default=True, help="Subtract the slice's mean background frame.")
@click.option("--include-dark", is_flag=True, help="Include the labeled proof-of-darkness perimeter frames (excluded and cropped away by default).")
@click.option("--output-stem", default="composite", show_default=True)
@click.option("--colormap", default="inferno", show_default=True)
def composite(
    targets: tuple[str, ...],
    match_pattern: str,
    commit: bool,
    pixel_size_um: float,
    downsample: int,
    flip_x: bool,
    flip_z: bool,
    transpose: bool,
    subtract_background: bool,
    include_dark: bool,
    output_stem: str,
    colormap: str,
) -> None:
    """
    Composite slice folder(s) into composite.png/.npy using the
    commanded frame positions.

    Each TARGET_DIR may be a single slice (e.g.
    data/auto_scan-.../y0018.00cm), a run directory (e.g.
    data/auto_scan-...), or a glob pattern matching run directories
    (e.g. 'data/auto_scan-2026-07-29_*', quoted or shell-expanded). Run
    directories composite every subdirectory matching --match-pattern in
    turn — the built-in equivalent of `for d in run/y*cm/; do ... done`.
    A failing slice is reported and skipped, not fatal; the command
    exits nonzero if any slice (or any --commit) failed.
    """

    from composite import CompositeConfig, composite_slice

    run_dirs = expand_targets(targets)
    if not run_dirs:
        raise click.UsageError(
            "Provide at least one slice/run directory or glob pattern."
        )
    multi_run = len(run_dirs) > 1

    config = CompositeConfig(
        PixelSize_um=pixel_size_um,
        Downsample=downsample,
        FlipX=flip_x,
        FlipZ=flip_z,
        Transpose=transpose,
        SubtractBackground=subtract_background,
        IncludeDarkFrames=include_dark,
        Colormap=colormap,
    )

    failed_slices: list[str] = []
    failed_commits: list[str] = []

    for run_dir in run_dirs:
        if multi_run:
            click.secho(f"\n=== {run_dir} ===", bold=True)

        slice_dirs = sorted(
            p for p in run_dir.glob(match_pattern) if p.is_dir()
        )
        if not slice_dirs:
            # No matching subdirectories: the target is itself the slice
            # (the original single-slice behavior, unchanged).
            slice_dirs = [run_dir]

        run_failures = 0

        for index, slice_dir in enumerate(slice_dirs, start=1):
            prefix = (
                f"[{index}/{len(slice_dirs)}] {slice_dir.name}: "
                if len(slice_dirs) > 1
                else ""
            )
            try:
                outputs = composite_slice(
                    slice_dir, config, output_stem=output_stem
                )
            except Exception as ex:  # noqa: BLE001 - report + continue; exit nonzero at the end
                run_failures += 1
                failed_slices.append(
                    f"{run_dir.name}/{slice_dir.name}"
                    if multi_run
                    else slice_dir.name
                )
                click.secho(f"{prefix}FAILED: {ex}", fg="red")
                continue

            click.echo(f"{prefix}{outputs['png']}")
            if len(slice_dirs) == 1 and not multi_run:
                click.echo(f"Composite array: {outputs['npy']}")
                click.echo(f"Metadata:        {outputs['meta']}")

        if len(slice_dirs) > 1:
            done = len(slice_dirs) - run_failures
            click.echo(f"\n{done}/{len(slice_dirs)} slices composited.")

        if commit:
            if run_failures:
                click.secho(
                    f"{run_dir}: NOT committed "
                    f"({run_failures} slice(s) failed).",
                    fg="red",
                )
            else:
                try:
                    commit_run(run_dir)
                except (subprocess.CalledProcessError, OSError) as ex:
                    stderr = (
                        ex.stderr.strip()
                        if isinstance(ex, subprocess.CalledProcessError)
                        and ex.stderr
                        else str(ex)
                    )
                    failed_commits.append(run_dir.name)
                    click.secho(
                        f"{run_dir}: git commit FAILED: {stderr}", fg="red"
                    )

    problems = []
    if failed_slices:
        problems.append(
            f"{len(failed_slices)} slice(s) failed: "
            + ", ".join(failed_slices)
        )
    if failed_commits:
        problems.append(
            f"{len(failed_commits)} commit(s) failed: "
            + ", ".join(failed_commits)
        )
    if problems:
        raise click.ClickException("; ".join(problems))
