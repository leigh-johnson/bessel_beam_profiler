"""
Position-based compositing of one slice's X-Z raster into a single image.

Gantry scans record the commanded machine position of every frame, so
each frame is simply PLACED at its known coordinates and overlaps are
averaged. This works on frames image registration cannot handle
(mostly-dark, featureless, or empty proof-of-darkness perimeter frames
from the adaptive raster).

Geometry: machine X (horizontal transverse) maps to composite columns
(+X right), machine Z (vertical) to rows (+Z up). Default image-axis
mapping assumes array columns ↔ +X and array row 0 ↔ +Z; if seams in the
composite look shifted or mirrored, fix with --flip-x / --flip-z /
--transpose (the composite itself is the diagnostic — arcs should be
continuous across frame boundaries).

Pixel scale: the BFS-PGE-31S4M has 3.45 um pixels at full resolution, so
1 mm = ~290 px; frames are mean-pooled by --downsample (default 8 ->
27.6 um/px) to keep the canvas manageable.

Outputs (written into the slice folder): composite.npy (float32 mean
counts), composite.png (inferno, robust scaling), composite_meta.json
(extent in mm, scale, options, frames used).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)


class CompositeError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompositeConfig:
    PixelSize_um: float = 3.45
    Downsample: int = 8
    # Verified 2026-07-22 from seam continuity of a real 36-frame ring
    # composite: this camera is mounted 180 deg rotated relative to the
    # machine axes (both flips), no transpose.
    FlipX: bool = True
    FlipZ: bool = True
    Transpose: bool = False
    SubtractBackground: bool = True
    ScanKinds: tuple[str, ...] = ("AutoBeamStack",)
    Colormap: str = "inferno"

    # The adaptive raster's proof-of-darkness perimeter frames (labeled in
    # raster_metadata.json Cells[].AnySignal=false) stay in the DATASET —
    # they document why growth stopped — but by default they are excluded
    # from the composite, which then crops to the lit region.
    IncludeDarkFrames: bool = False


def load_scan_records(slice_dir: Path, scan_kinds) -> list[dict]:
    manifest = slice_dir / "frames.jsonl"

    if not manifest.exists():
        raise CompositeError(f"No frames.jsonl in {slice_dir}")

    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    return [r for r in records if r["Extra"].get("ScanKind") in scan_kinds]


def resolve_frame_path(record_path: str, slice_dir: Path) -> Path:
    """
    Manifest paths may be relative to the directory the scan RAN from;
    fall back to filename lookup in (or next to) the slice folder.
    """

    candidates = [
        Path(record_path),
        slice_dir / Path(record_path).name,
        slice_dir.parent / Path(record_path).parent.name / Path(record_path).name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise CompositeError(
        f"Frame file not found: {record_path} (also tried {candidates[1:]})"
    )


def load_dark_frame_names(slice_dir: Path):
    """
    Filenames of the slice's no-signal frames, from the adaptive raster's
    per-cell labels (raster_metadata.json Cells[].AnySignal). Returns None
    when no labels exist (fixed-mode or legacy scans): nothing to exclude.
    """

    metadata_path = slice_dir / "raster_metadata.json"

    if not metadata_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text())
    cells = metadata.get("Cells")

    if not cells:
        return None

    dark: set[str] = set()

    for cell in cells:
        if not cell.get("AnySignal"):
            for path in cell.get("Paths", []):
                if path:
                    dark.add(Path(path).name)

    return dark


def orient(arr: np.ndarray, config: CompositeConfig) -> np.ndarray:
    """Map the raw array into composite orientation (cols=+X, row0=+Z)."""

    if config.Transpose:
        arr = arr.T
    if config.FlipX:
        arr = arr[:, ::-1]
    if config.FlipZ:
        arr = arr[::-1, :]
    return arr


def mean_pool(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr.astype(np.float32)

    rows = (arr.shape[0] // factor) * factor
    cols = (arr.shape[1] // factor) * factor
    trimmed = arr[:rows, :cols].astype(np.float32)

    return trimmed.reshape(
        rows // factor, factor, cols // factor, factor
    ).mean(axis=(1, 3))


def load_background(slice_dir: Path, config: CompositeConfig):
    """Mean of the slice's background frames (via background_reference.json)."""

    reference_path = slice_dir / "background_reference.json"

    if not reference_path.exists():
        logger.warning(
            f"No background_reference.json in {slice_dir}; compositing "
            "without background subtraction."
        )
        return None

    reference = json.loads(reference_path.read_text())
    arrays = []

    for path in reference.get("BackgroundPaths", []):
        if path is None:
            continue
        try:
            arrays.append(np.load(resolve_frame_path(path, slice_dir)))
        except CompositeError as ex:
            logger.warning(f"Background frame unavailable ({ex}); skipping it.")

    if not arrays:
        logger.warning(
            "No background frames could be loaded; compositing without "
            "background subtraction."
        )
        return None

    background = np.mean(np.stack(arrays), axis=0)
    logger.info(
        f"Background: {len(arrays)} frame(s), mean level "
        f"{float(background.mean()):.2f} counts "
        f"(exposure {reference.get('BackgroundExposure_us')} us, "
        f"reused={reference.get('Reused')})"
    )
    return background


def composite_slice(
    slice_dir: Path,
    config: CompositeConfig = CompositeConfig(),
    output_stem: str = "composite",
) -> dict:
    slice_dir = Path(slice_dir)
    records = load_scan_records(slice_dir, config.ScanKinds)

    if not records:
        raise CompositeError(
            f"No frames with ScanKind in {config.ScanKinds} in {slice_dir}"
        )

    dark_excluded = 0

    if not config.IncludeDarkFrames:
        # Dark frames are labeled by filename ('-dark' suffix) on newer
        # scans; older runs are covered by the raster metadata labels.
        dark_names = load_dark_frame_names(slice_dir) or set()

        def is_dark(record) -> bool:
            path = Path(record["Path"])
            return path.stem.endswith("-dark") or path.name in dark_names

        lit_records = [r for r in records if not is_dark(r)]
        dark_excluded = len(records) - len(lit_records)

        if dark_excluded and not lit_records:
            raise CompositeError(
                "Every frame in this slice is labeled dark — nothing to "
                "composite (use IncludeDarkFrames/--include-dark to force)."
            )

        if dark_excluded:
            logger.info(
                f"Excluding {dark_excluded} labeled proof-of-darkness "
                "frame(s); composite crops to the lit region "
                "(--include-dark to keep them)."
            )
            records = lit_records

    background = (
        load_background(slice_dir, config) if config.SubtractBackground else None
    )
    if background is not None:
        background = mean_pool(orient(background, config), config.Downsample)

    mm_per_px = config.PixelSize_um / 1000.0 * config.Downsample

    # First pass: frame geometry from the first frame (all frames share it).
    first = np.load(resolve_frame_path(records[0]["Path"], slice_dir))
    frame_ds = mean_pool(orient(first, config), config.Downsample)
    frame_rows, frame_cols = frame_ds.shape
    frame_w_mm = frame_cols * mm_per_px
    frame_h_mm = frame_rows * mm_per_px

    xs = [r["GantryPosition_mm"]["x_mm"] for r in records]
    zs = [r["GantryPosition_mm"]["z_mm"] for r in records]

    x_left_mm = min(xs) - frame_w_mm / 2.0
    z_top_mm = max(zs) + frame_h_mm / 2.0
    canvas_cols = int(round((max(xs) - min(xs) + frame_w_mm) / mm_per_px))
    canvas_rows = int(round((max(zs) - min(zs) + frame_h_mm) / mm_per_px))

    total = np.zeros((canvas_rows, canvas_cols), dtype=np.float64)
    count = np.zeros((canvas_rows, canvas_cols), dtype=np.int32)

    for record in records:
        arr = np.load(resolve_frame_path(record["Path"], slice_dir))
        frame = mean_pool(orient(arr, config), config.Downsample)

        if background is not None:
            frame = np.clip(frame - background, 0.0, None)

        x_mm = record["GantryPosition_mm"]["x_mm"]
        z_mm = record["GantryPosition_mm"]["z_mm"]

        col0 = int(round((x_mm - frame_w_mm / 2.0 - x_left_mm) / mm_per_px))
        row0 = int(round((z_top_mm - (z_mm + frame_h_mm / 2.0)) / mm_per_px))

        total[row0:row0 + frame.shape[0], col0:col0 + frame.shape[1]] += frame
        count[row0:row0 + frame.shape[0], col0:col0 + frame.shape[1]] += 1

    covered = count > 0
    composite = np.zeros_like(total, dtype=np.float32)
    composite[covered] = (total[covered] / count[covered]).astype(np.float32)

    # -- outputs --------------------------------------------------------

    npy_path = slice_dir / f"{output_stem}.npy"
    np.save(npy_path, composite)

    png_path = slice_dir / f"{output_stem}.png"
    _save_png(png_path, composite, covered, config.Colormap)

    meta = {
        "Frames": len(records),
        "PixelSize_um": config.PixelSize_um,
        "Downsample": config.Downsample,
        "mm_per_px": mm_per_px,
        "FlipX": config.FlipX,
        "FlipZ": config.FlipZ,
        "Transpose": config.Transpose,
        "BackgroundSubtracted": background is not None,
        "DarkFramesExcluded": dark_excluded,
        "Extent_mm": {
            "XMin": x_left_mm,
            "XMax": x_left_mm + canvas_cols * mm_per_px,
            "ZMin": z_top_mm - canvas_rows * mm_per_px,
            "ZMax": z_top_mm,
        },
        "CanvasShape": [canvas_rows, canvas_cols],
        "PeakMean_counts": float(composite.max()),
        "MachineY_mm": records[0]["GantryPosition_mm"]["y_mm"],
        "BeamY_mm": records[0]["Extra"].get("BeamY_mm"),
    }

    meta_path = slice_dir / f"{output_stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    logger.info(
        f"Composite: {canvas_cols}x{canvas_rows} px covering "
        f"X {meta['Extent_mm']['XMin']:.1f}..{meta['Extent_mm']['XMax']:.1f} mm, "
        f"Z {meta['Extent_mm']['ZMin']:.1f}..{meta['Extent_mm']['ZMax']:.1f} mm "
        f"({mm_per_px * 1000:.1f} um/px, peak {composite.max():.1f} counts)"
    )

    return {"npy": npy_path, "png": png_path, "meta": meta_path, "metadata": meta}


def _save_png(path: Path, composite: np.ndarray, covered: np.ndarray, colormap: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = composite[covered]
    vmax = float(np.percentile(values, 99.9)) if values.size else 1.0

    plt.imsave(
        path,
        composite,
        cmap=colormap,
        vmin=0.0,
        vmax=max(vmax, 1.0),
        origin="upper",
    )
