"""
`dataset composite` — place one slice's frames at their commanded gantry
positions and average the overlaps. The position-based counterpart of
`dataset stitch` (registration-based, for legacy manual scans); use this
one for gantry runs — it handles the adaptive raster's dark perimeter
frames that registration cannot.
"""

from __future__ import annotations

from pathlib import Path

import click


@click.command("composite")
@click.argument(
    "slice_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
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
    slice_dir: Path,
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
    Composite one slice folder (e.g. data/auto_scan-.../y0018.00cm) into
    composite.png/.npy using the commanded frame positions.
    """

    from composite import CompositeConfig, composite_slice

    outputs = composite_slice(
        slice_dir,
        CompositeConfig(
            PixelSize_um=pixel_size_um,
            Downsample=downsample,
            FlipX=flip_x,
            FlipZ=flip_z,
            Transpose=transpose,
            SubtractBackground=subtract_background,
            IncludeDarkFrames=include_dark,
            Colormap=colormap,
        ),
        output_stem=output_stem,
    )

    click.echo(f"Composite image: {outputs['png']}")
    click.echo(f"Composite array: {outputs['npy']}")
    click.echo(f"Metadata:        {outputs['meta']}")
