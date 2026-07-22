"""
Tests for the adaptive raster growth logic against a synthetic beam field.

The fake "camera" renders a disk-shaped beam onto a small sensor at any
requested position, so tests can verify coverage guarantees (every lattice
cell that would contain signal gets captured) independent of hardware.
No PySpin involvement anywhere.
"""

import types

import numpy as np
import pytest

from adaptive_raster import (
    AdaptiveRasterConfig,
    AdaptiveRasterError,
    AdaptiveRasterRunner,
    _machine_side_to_array_strip,
    border_strips,
)


# Synthetic sensor: 12 rows x 16 cols at 0.5 mm/px -> 8 mm wide, 6 mm tall.
ROWS, COLS = 12, 16
PX_MM = 0.5

BEAM_VALUE = 150
DARK_VALUE = 2

CENTER = (60.0, 80.0)
STEP = (5.0, 4.0)


def render_frame(x_mm, y_mm, beam_center, beam_radius_mm):
    """
    Frame at camera-center (x_mm, y_mm): array cols increase with machine
    +x, rows increase with machine +y (the identity mapping).
    """

    cols = x_mm + (np.arange(COLS) - (COLS - 1) / 2.0) * PX_MM
    rows = y_mm + (np.arange(ROWS) - (ROWS - 1) / 2.0) * PX_MM

    xx, yy = np.meshgrid(cols, rows)
    inside = (xx - beam_center[0]) ** 2 + (yy - beam_center[1]) ** 2 <= beam_radius_mm**2

    return np.where(inside, BEAM_VALUE, DARK_VALUE).astype(np.uint8)


class SyntheticCamera:
    def __init__(self, beam_center, beam_radius_mm):
        self.beam_center = beam_center
        self.beam_radius_mm = beam_radius_mm
        self.captured = []  # (x_mm, y_mm, i, j)

    def __call__(self, x_mm, y_mm, i, j):
        self.captured.append((x_mm, y_mm, i, j))
        arr = render_frame(x_mm, y_mm, self.beam_center, self.beam_radius_mm)
        record = types.SimpleNamespace(Path=f"cell_{i}_{j}.npy")
        return [record], arr


def make_config(**overrides):
    defaults = dict(
        CenterX_mm=CENTER[0],
        CenterY_mm=CENTER[1],
        StepX_mm=STEP[0],
        StepY_mm=STEP[1],
        XMin_mm=CENTER[0] - 4 * STEP[0],
        XMax_mm=CENTER[0] + 4 * STEP[0],
        YMin_mm=CENTER[1] - 4 * STEP[1],
        YMax_mm=CENTER[1] + 4 * STEP[1],
        SignalThreshold_counts=10.0,
        MinSignalPixels=4,
        BorderStripFraction=0.15,
    )
    defaults.update(overrides)
    return AdaptiveRasterConfig(**defaults)


def run_raster(beam_center, beam_radius_mm, echo=None, **config_overrides):
    camera = SyntheticCamera(beam_center, beam_radius_mm)
    runner = AdaptiveRasterRunner(
        make_config(**config_overrides), camera, echo_fn=echo
    )
    return runner.run(), camera


def lattice_cells_with_signal(config, beam_center, beam_radius_mm):
    """Ground truth: every capped lattice cell whose frame contains signal."""

    cells = set()

    i_range = range(
        -int((config.CenterX_mm - config.XMin_mm) // config.StepX_mm),
        int((config.XMax_mm - config.CenterX_mm) // config.StepX_mm) + 1,
    )
    j_range = range(
        -int((config.CenterY_mm - config.YMin_mm) // config.StepY_mm),
        int((config.YMax_mm - config.CenterY_mm) // config.StepY_mm) + 1,
    )

    for i in i_range:
        for j in j_range:
            x = config.CenterX_mm + i * config.StepX_mm
            y = config.CenterY_mm + j * config.StepY_mm
            arr = render_frame(x, y, beam_center, beam_radius_mm)

            if np.sum(arr > config.SignalThreshold_counts) >= config.MinSignalPixels:
                cells.add((i, j))

    return cells


# ---------------------------------------------------------------------------
# Core behavior
# ---------------------------------------------------------------------------


def test_beam_fitting_in_one_frame_takes_exactly_one_frame():
    echoes = []
    result, camera = run_raster(CENTER, beam_radius_mm=1.5, echo=echoes.append)

    assert len(camera.captured) == 1
    assert camera.captured[0] == (CENTER[0], CENTER[1], 0, 0)

    meta = result.Metadata
    assert meta["BeamFitsInSingleFrame"] is True
    assert meta["CellsCaptured"] == 1
    assert meta["GridShape"] == [1, 1]
    assert meta["EdgeStops"] == {s: "dark" for s in ("+x", "-x", "+y", "-y")}
    assert meta["TruncatedSides"] == []
    assert any("single camera frame" in message for message in echoes)


def test_medium_beam_grows_until_edges_dark_and_covers_all_signal_cells():
    radius = 6.0
    result, camera = run_raster(CENTER, radius)

    captured = {(i, j) for (_, _, i, j) in camera.captured}
    required = lattice_cells_with_signal(make_config(), CENTER, radius)

    # Coverage guarantee: every cell that would contain signal was captured.
    assert required <= captured
    # And it did NOT fall back to the full fixed grid.
    meta = result.Metadata
    assert meta["CellsCaptured"] == len(camera.captured)
    assert meta["CellsCaptured"] < meta["FixedGridCells"]
    assert meta["BeamFitsInSingleFrame"] is False
    assert meta["TruncatedSides"] == []
    assert set(meta["EdgeStops"].values()) == {"dark"}


def test_off_center_beam_grows_asymmetrically_with_coverage():
    beam_center = (CENTER[0] + 5.0, CENTER[1])  # shifted +x
    radius = 4.0
    result, camera = run_raster(beam_center, radius)

    captured = {(i, j) for (_, _, i, j) in camera.captured}
    required = lattice_cells_with_signal(make_config(), beam_center, radius)

    assert required <= captured
    assert (2, 0) in captured  # beam reaches x=69 -> cell at x=70 has signal

    rect = result.Metadata["FinalRect_mm"]
    # Grew farther toward +x than -x.
    assert rect["XMax"] - CENTER[0] > CENTER[0] - rect["XMin"]


def test_huge_beam_stops_at_cap_and_reports_truncation():
    echoes = []
    result, camera = run_raster(CENTER, beam_radius_mm=100.0, echo=echoes.append)

    meta = result.Metadata
    assert sorted(meta["TruncatedSides"]) == ["+x", "+y", "-x", "-y"] or set(
        meta["TruncatedSides"]
    ) == {"+x", "-x", "+y", "-y"}
    assert all(reason == "cap" for reason in meta["EdgeStops"].values())
    # Grew to exactly the full capped lattice (9 x 9).
    assert meta["GridShape"] == [9, 9]
    assert meta["CellsCaptured"] == meta["FixedGridCells"] == 81
    assert any("TRUNCATED" in message for message in echoes)


def test_each_cell_captured_exactly_once():
    _, camera = run_raster(CENTER, beam_radius_mm=6.0)

    cells = [(i, j) for (_, _, i, j) in camera.captured]
    assert len(cells) == len(set(cells))


def test_metadata_records_cells_and_threshold():
    result, _ = run_raster(CENTER, beam_radius_mm=1.5)
    meta = result.Metadata

    assert meta["RasterMode"] == "adaptive"
    assert meta["SignalThreshold_counts"] == 10.0
    assert meta["Step_mm"] == [5.0, 4.0]
    assert len(meta["Cells"]) == meta["CellsCaptured"]

    cell = meta["Cells"][0]
    assert cell["I"] == 0 and cell["J"] == 0
    assert cell["AnySignal"] is True
    assert cell["Paths"] == ["cell_0_0.npy"]
    assert set(cell["BorderSignal"]) == {"+x", "-x", "+y", "-y"}


def test_warns_when_min_signal_pixels_exceeds_strip_size():
    echoes = []
    run_raster(CENTER, beam_radius_mm=6.0, echo=echoes.append, MinSignalPixels=10_000)

    assert any("NEVER grow" in message for message in echoes)


def test_rejects_unknown_border_test_mode():
    with pytest.raises(AdaptiveRasterError, match="BorderTest"):
        AdaptiveRasterRunner(
            make_config(BorderTest="sideways"), SyntheticCamera(CENTER, 1.0)
        )


def test_center_outside_caps_is_an_error():
    runner = AdaptiveRasterRunner(
        make_config(XMin_mm=CENTER[0] + 10.0), SyntheticCamera(CENTER, 1.0)
    )

    with pytest.raises(AdaptiveRasterError, match="outside the configured"):
        runner.run()


# ---------------------------------------------------------------------------
# Directional border test
# ---------------------------------------------------------------------------


def test_directional_mode_grows_only_toward_the_beam():
    beam_center = (CENTER[0] + 5.0, CENTER[1])
    radius = 4.0

    result, camera = run_raster(
        beam_center, radius, BorderTest="directional"
    )

    captured = {(i, j) for (_, _, i, j) in camera.captured}
    required = lattice_cells_with_signal(make_config(), beam_center, radius)

    assert required <= captured
    # Directional mode never probes the dark -x side beyond the seed column.
    assert all(i >= 0 for (i, j) in captured)


def test_machine_side_to_array_strip_mapping():
    identity = make_config(BorderTest="directional")
    assert _machine_side_to_array_strip("+x", identity) == "col1"
    assert _machine_side_to_array_strip("-x", identity) == "col0"
    assert _machine_side_to_array_strip("+y", identity) == "row1"
    assert _machine_side_to_array_strip("-y", identity) == "row0"

    flipped = make_config(BorderTest="directional", ImageFlipX=True)
    assert _machine_side_to_array_strip("+x", flipped) == "col0"
    assert _machine_side_to_array_strip("-x", flipped) == "col1"

    transposed = make_config(BorderTest="directional", ImageTranspose=True)
    assert _machine_side_to_array_strip("+x", transposed) == "row1"
    assert _machine_side_to_array_strip("+y", transposed) == "col1"


def test_border_strips_shapes():
    arr = np.zeros((12, 16))
    strips = border_strips(arr, 0.15)

    assert strips["row0"].shape == (2, 16)
    assert strips["row1"].shape == (2, 16)
    assert strips["col0"].shape == (12, 2)
    assert strips["col1"].shape == (12, 2)
