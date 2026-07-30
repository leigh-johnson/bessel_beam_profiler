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


# ---------------------------------------------------------------------------
# CLI: run-directory batching via --match-pattern
# ---------------------------------------------------------------------------


def _write_named_slice(run_dir, name):
    slice_dir = run_dir / name
    slice_dir.mkdir(parents=True)
    arr = np.full((12, 16), 20, dtype=np.uint8)
    np.save(slice_dir / "frame_000.npy", arr)
    (slice_dir / "frames.jsonl").write_text(
        json.dumps(
            {
                "Path": f"bogus/frame_000.npy",
                "GantryPosition_mm": {"x_mm": 0.0, "y_mm": 10.0, "z_mm": 0.0},
                "Extra": {"ScanKind": "AutoBeamStack"},
            }
        )
        + "\n"
    )
    return slice_dir


def _composite_cli_args(extra):
    return extra + [
        "--pixel-size-um", str(PIXEL_UM),
        "--downsample", "1",
        "--no-flip-x", "--no-flip-z", "--no-subtract",
    ]


def test_cli_composites_every_matching_slice_in_a_run_dir(tmp_path):
    from click.testing import CliRunner

    from composite_cli import composite as composite_command

    run_dir = tmp_path / "auto_scan-2026-07-28_15-18-35"
    for name in ("y0029.60cm", "y0030.10cm"):
        _write_named_slice(run_dir, name)
    (run_dir / "not_a_slice").mkdir()  # must be ignored by the pattern

    result = CliRunner().invoke(
        composite_command, _composite_cli_args([str(run_dir)])
    )

    assert result.exit_code == 0, result.output
    assert (run_dir / "y0029.60cm" / "composite.png").exists()
    assert (run_dir / "y0030.10cm" / "composite.png").exists()
    assert not (run_dir / "not_a_slice" / "composite.png").exists()
    assert "2/2 slices composited" in result.output


def test_cli_single_slice_dir_still_works(tmp_path):
    from click.testing import CliRunner

    from composite_cli import composite as composite_command

    slice_dir = _write_named_slice(tmp_path, "y0018.00cm")

    result = CliRunner().invoke(
        composite_command, _composite_cli_args([str(slice_dir)])
    )

    assert result.exit_code == 0, result.output
    assert (slice_dir / "composite.png").exists()
    assert "Composite array:" in result.output


def test_cli_reports_failing_slice_and_continues(tmp_path):
    from click.testing import CliRunner

    from composite_cli import composite as composite_command

    run_dir = tmp_path / "auto_scan-run"
    _write_named_slice(run_dir, "y0001.00cm")
    (run_dir / "y0002.00cm").mkdir()  # matches pattern but has no frames

    result = CliRunner().invoke(
        composite_command, _composite_cli_args([str(run_dir)])
    )

    assert result.exit_code != 0
    assert (run_dir / "y0001.00cm" / "composite.png").exists()
    assert "FAILED" in result.output
    assert "1/2 slices composited" in result.output
    assert "1 slice(s) failed" in result.output


# ---------------------------------------------------------------------------
# CLI: multiple run dirs, glob patterns, --commit
# ---------------------------------------------------------------------------


def _invoke_composite(args):
    from click.testing import CliRunner

    from composite_cli import composite as composite_command

    return CliRunner().invoke(composite_command, _composite_cli_args(args))


def _git(cwd, *args):
    import subprocess

    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )


def _init_git_repo(path):
    _git(path, "init", "--initial-branch=main")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")


def test_cli_composites_multiple_run_dirs(tmp_path):
    run_a = tmp_path / "auto_scan-2026-07-29_13-36-52"
    run_b = tmp_path / "auto_scan-2026-07-29_14-01-15"
    _write_named_slice(run_a, "y0001.00cm")
    _write_named_slice(run_b, "y0002.00cm")

    result = _invoke_composite([str(run_a), str(run_b)])

    assert result.exit_code == 0, result.output
    assert (run_a / "y0001.00cm" / "composite.png").exists()
    assert (run_b / "y0002.00cm" / "composite.png").exists()
    assert f"=== {run_a} ===" in result.output
    assert f"=== {run_b} ===" in result.output


def test_cli_quoted_glob_pattern_expands_to_run_dirs(tmp_path):
    # A pattern the shell did NOT expand (quoted) must be globbed by the
    # CLI itself, sorted, and each match composited.
    run_a = tmp_path / "auto_scan-2026-07-29_13-36-52"
    run_b = tmp_path / "auto_scan-2026-07-29_14-01-15"
    _write_named_slice(run_a, "y0001.00cm")
    _write_named_slice(run_b, "y0002.00cm")
    (tmp_path / "auto_scan-notes.txt").write_text("not a dir")

    result = _invoke_composite([str(tmp_path / "auto_scan-2026-07-29_*")])

    assert result.exit_code == 0, result.output
    assert (run_a / "y0001.00cm" / "composite.png").exists()
    assert (run_b / "y0002.00cm" / "composite.png").exists()
    # Sorted expansion: earlier timestamp reported first.
    assert result.output.index(run_a.name) < result.output.index(run_b.name)


def test_cli_pattern_matching_nothing_is_a_usage_error(tmp_path):
    result = _invoke_composite([str(tmp_path / "auto_scan-1999-*")])

    assert result.exit_code != 0
    assert "matches no directories" in result.output


def test_cli_commit_makes_one_commit_per_run_dir(tmp_path, monkeypatch):
    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    for run, slc in (
        ("auto_scan-2026-07-29_13-36-52", "y0001.00cm"),
        ("auto_scan-2026-07-29_14-01-15", "y0002.00cm"),
    ):
        _write_named_slice(tmp_path / run, slc)

    result = _invoke_composite(["auto_scan-2026-07-29_*", "--commit"])

    assert result.exit_code == 0, result.output
    subjects = _git(
        tmp_path, "log", "--format=%s"
    ).stdout.strip().splitlines()
    # One commit per run, committed in expansion (timestamp) order.
    assert subjects == [
        "composite: auto_scan-2026-07-29_14-01-15/",
        "composite: auto_scan-2026-07-29_13-36-52/",
    ]

    # Each commit contains only its own run directory.
    files = _git(
        tmp_path, "show", "--name-only", "--format=", "HEAD"
    ).stdout
    assert "auto_scan-2026-07-29_14-01-15/" in files
    assert "auto_scan-2026-07-29_13-36-52/" not in files

    # Re-running is a no-op: nothing new, no third commit, exit 0.
    rerun = _invoke_composite(["auto_scan-2026-07-29_*", "--commit"])
    assert rerun.exit_code == 0, rerun.output
    assert "nothing new to commit" in rerun.output
    assert (
        len(_git(tmp_path, "log", "--format=%s").stdout.strip().splitlines())
        == 2
    )


def test_cli_commit_leaves_unrelated_staged_work_alone(tmp_path, monkeypatch):
    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    _write_named_slice(tmp_path / "auto_scan-run", "y0001.00cm")
    (tmp_path / "notes.md").write_text("work in progress")
    _git(tmp_path, "add", "notes.md")

    result = _invoke_composite(["auto_scan-run", "--commit"])

    assert result.exit_code == 0, result.output
    committed = _git(
        tmp_path, "show", "--name-only", "--format=", "HEAD"
    ).stdout
    assert "notes.md" not in committed
    # notes.md is still staged, untouched.
    assert "notes.md" in _git(tmp_path, "diff", "--cached", "--name-only").stdout


def test_cli_commit_skipped_when_a_slice_fails(tmp_path, monkeypatch):
    import subprocess

    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    run_dir = tmp_path / "auto_scan-run"
    _write_named_slice(run_dir, "y0001.00cm")
    (run_dir / "y0002.00cm").mkdir()  # matches pattern but has no frames

    result = _invoke_composite(["auto_scan-run", "--commit"])

    assert result.exit_code != 0
    assert "NOT committed" in result.output
    with pytest.raises(subprocess.CalledProcessError):
        _git(tmp_path, "rev-parse", "HEAD")  # no commit was created
