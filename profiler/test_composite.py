"""
Tests for position-based slice compositing: placement geometry, overlap
averaging, path fallback, background subtraction, and orientation flags.
"""

import json

import numpy as np
import pytest

from composite import (
    CompositeConfig,
    CompositeError,
    composite_slice,
    mean_pool,
    orient,
)


# Synthetic sensor: 12 rows x 16 cols at 0.5 mm/px -> 8 mm wide, 6 mm tall.
PIXEL_UM = 500.0

GEOMETRY = CompositeConfig(
    PixelSize_um=PIXEL_UM,
    Downsample=1,
    FlipX=False,
    FlipZ=False,
    SubtractBackground=False,
)


def write_slice(tmp_path, frames, backgrounds=None):
    """
    frames: list of (x_mm, z_mm, array). Writes npy files + frames.jsonl
    with manifest paths pointing at a BOGUS directory, so resolution must
    fall back to the slice folder (mirrors scans run from another cwd).
    """

    slice_dir = tmp_path / "y0018.00cm"
    slice_dir.mkdir()

    records = []

    for idx, (x_mm, z_mm, arr) in enumerate(frames):
        name = f"frame_{idx:03d}.npy"
        np.save(slice_dir / name, arr)
        records.append(
            {
                "Path": f"data/some/other/cwd/{name}",
                "GantryPosition_mm": {"x_mm": x_mm, "y_mm": 10.0, "z_mm": z_mm},
                "Extra": {"ScanKind": "AutoBeamStack"},
            }
        )

    (slice_dir / "frames.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )

    if backgrounds is not None:
        paths = []
        for idx, arr in enumerate(backgrounds):
            name = f"bg_{idx:03d}.npy"
            np.save(slice_dir / name, arr)
            paths.append(f"data/some/other/cwd/{name}")

        (slice_dir / "background_reference.json").write_text(
            json.dumps(
                {
                    "Reused": False,
                    "BackgroundExposure_us": 1000.0,
                    "BackgroundPaths": paths,
                }
            )
        )

    return slice_dir


def test_two_frame_overlap_averages_and_geometry(tmp_path):
    a = np.full((12, 16), 10, dtype=np.uint8)
    b = np.full((12, 16), 30, dtype=np.uint8)

    # 5 mm apart in X with 8 mm-wide frames -> 3 mm (6 px) overlap.
    slice_dir = write_slice(tmp_path, [(0.0, 0.0, a), (5.0, 0.0, b)])

    outputs = composite_slice(slice_dir, GEOMETRY)
    composite = np.load(outputs["npy"])

    assert composite.shape == (12, 26)  # 13 mm x 6 mm at 0.5 mm/px
    assert composite[6, 2] == pytest.approx(10.0)   # left-only region
    assert composite[6, 23] == pytest.approx(30.0)  # right-only region
    assert composite[6, 13] == pytest.approx(20.0)  # overlap average

    meta = json.loads(outputs["meta"].read_text())
    assert meta["Extent_mm"]["XMin"] == pytest.approx(-4.0)
    assert meta["Extent_mm"]["XMax"] == pytest.approx(9.0)
    assert meta["Frames"] == 2
    assert outputs["png"].exists()


def test_z_maps_to_rows_with_plus_z_up(tmp_path):
    low = np.full((12, 16), 10, dtype=np.uint8)
    high = np.full((12, 16), 30, dtype=np.uint8)

    # Frames 6 mm apart in Z (no overlap): high-Z frame must be on TOP.
    slice_dir = write_slice(tmp_path, [(0.0, 0.0, low), (0.0, 6.0, high)])

    composite = np.load(composite_slice(slice_dir, GEOMETRY)["npy"])

    assert composite.shape == (24, 16)
    assert composite[2, 8] == pytest.approx(30.0)   # top rows = high Z
    assert composite[20, 8] == pytest.approx(10.0)  # bottom rows = low Z


def test_background_subtraction_clips_at_zero(tmp_path):
    frame = np.full((12, 16), 10, dtype=np.uint8)
    slice_dir = write_slice(
        tmp_path,
        [(0.0, 0.0, frame)],
        backgrounds=[
            np.full((12, 16), 4, dtype=np.uint8),
            np.full((12, 16), 6, dtype=np.uint8),  # mean background = 5
        ],
    )

    config = CompositeConfig(
        PixelSize_um=PIXEL_UM,
        Downsample=1,
        FlipX=False,
        FlipZ=False,
        SubtractBackground=True,
    )
    composite = np.load(composite_slice(slice_dir, config)["npy"])

    assert composite[6, 8] == pytest.approx(5.0)  # 10 - mean(4, 6)


def test_dark_labeled_files_are_excluded_and_canvas_crops(tmp_path):
    lit = np.full((12, 16), 30, dtype=np.uint8)
    dark = np.full((12, 16), 1, dtype=np.uint8)

    slice_dir = tmp_path / "y0018.00cm"
    slice_dir.mkdir()

    np.save(slice_dir / "frame_lit.npy", lit)
    np.save(slice_dir / "frame_perimeter-dark.npy", dark)

    records = [
        {
            "Path": f"elsewhere/{name}",
            "GantryPosition_mm": {"x_mm": x, "y_mm": 10.0, "z_mm": 0.0},
            "Extra": {"ScanKind": "AutoBeamStack"},
        }
        for name, x in (("frame_lit.npy", 0.0), ("frame_perimeter-dark.npy", 5.0))
    ]
    (slice_dir / "frames.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )

    outputs = composite_slice(slice_dir, GEOMETRY)
    composite = np.load(outputs["npy"])

    # Only the lit frame: canvas is one frame wide, not two.
    assert composite.shape == (12, 16)
    meta = json.loads(outputs["meta"].read_text())
    assert meta["DarkFramesExcluded"] == 1
    assert meta["Frames"] == 1

    # --include-dark restores the full fence.
    import dataclasses

    full = composite_slice(
        slice_dir,
        dataclasses.replace(GEOMETRY, IncludeDarkFrames=True),
        output_stem="composite_full",
    )
    assert np.load(full["npy"]).shape == (12, 26)


def test_metadata_labeled_dark_frames_excluded_for_legacy_runs(tmp_path):
    # Runs from before filename labeling: raster_metadata Cells carry the
    # AnySignal=false labels under the ORIGINAL filenames.
    lit = np.full((12, 16), 30, dtype=np.uint8)
    dark = np.full((12, 16), 1, dtype=np.uint8)

    slice_dir = write_slice(tmp_path, [(0.0, 0.0, lit), (5.0, 0.0, dark)])

    (slice_dir / "raster_metadata.json").write_text(
        json.dumps(
            {
                "Cells": [
                    {"AnySignal": True, "Paths": ["elsewhere/frame_000.npy"]},
                    {"AnySignal": False, "Paths": ["elsewhere/frame_001.npy"]},
                ]
            }
        )
    )

    outputs = composite_slice(slice_dir, GEOMETRY)

    assert np.load(outputs["npy"]).shape == (12, 16)
    assert json.loads(outputs["meta"].read_text())["DarkFramesExcluded"] == 1


def test_orient_flags():
    arr = np.zeros((2, 3))
    arr[0, 0] = 1.0  # marker at top-left

    flipped_x = orient(arr, CompositeConfig(FlipX=True, FlipZ=False))
    assert flipped_x[0, 2] == 1.0

    flipped_z = orient(arr, CompositeConfig(FlipX=False, FlipZ=True))
    assert flipped_z[1, 0] == 1.0

    transposed = orient(
        arr, CompositeConfig(FlipX=False, FlipZ=False, Transpose=True)
    )
    assert transposed.shape == (3, 2)


def test_mean_pool_reduces_and_averages():
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)

    pooled = mean_pool(arr, 2)

    assert pooled.shape == (2, 2)
    assert pooled[0, 0] == pytest.approx(np.mean([0, 1, 4, 5]))


def test_missing_manifest_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(CompositeError, match="frames.jsonl"):
        composite_slice(empty, GEOMETRY)
