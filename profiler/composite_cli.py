"""
`dataset composite` — place one slice's frames at their commanded gantry
positions and average the overlaps. The position-based counterpart of
`dataset stitch` (registration-based, for legacy manual scans); use this
one for gantry runs — it handles the adaptive raster's dark perimeter
frames that registration cannot.

Accepts either one slice folder, or a RUN folder plus --match-pattern
(default y*cm) to composite every matching slice in one invocation.
"""

from __future__ import annotations

from pathlib import Path

import click


@click.command("composite")
@click.argument(
    "target_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--match-pattern",
    default="y*cm",
    show_default=True,
    help="When TARGET_DIR is a run directory rather than a single slice, "
    "composite every subdirectory matching this glob (sorted). A "
    "directory with no matching subdirectories is treated as a single "
    "slice, so existing single-slice invocations are unchanged.",
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
    target_dir: Path,
    match_pattern: str,
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

    TARGET_DIR may be a single slice (e.g. data/auto_scan-.../y0018.00cm)
    or a whole run directory (e.g. data/auto_scan-...), in which case
    every subdirectory matching --match-pattern is composited in turn —
    the built-in equivalent of `for d in run/y*cm/; do ... done`. A
    failing slice is reported and skipped, not fatal; the command exits
    nonzero if any slice failed.
    """

    from composite import CompositeConfig, composite_slice

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

    slice_dirs = sorted(
        p for p in target_dir.glob(match_pattern) if p.is_dir()
    )
    if not slice_dirs:
        # No matching subdirectories: TARGET_DIR is itself the slice
        # (the original single-slice behavior, unchanged).
        slice_dirs = [target_dir]

    failures: list[Path] = []

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
            failures.append(slice_dir)
            click.secho(f"{prefix}FAILED: {ex}", fg="red")
            continue

        click.echo(f"{prefix}{outputs['png']}")
        if len(slice_dirs) == 1:
            click.echo(f"Composite array: {outputs['npy']}")
            click.echo(f"Metadata:        {outputs['meta']}")

    if len(slice_dirs) > 1:
        done = len(slice_dirs) - len(failures)
        click.echo(f"\n{done}/{len(slice_dirs)} slices composited.")

    if failures:
        raise click.ClickException(
            f"{len(failures)} slice(s) failed: "
            + ", ".join(path.name for path in failures)
        )
