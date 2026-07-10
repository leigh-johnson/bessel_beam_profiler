import json

import numpy as np
import pytest

from scipy.special import j0

import stitcher


# ---------------------------------------------------------------------------
# Synthetic Bessel-beam fixtures
# ---------------------------------------------------------------------------


def make_bessel_scene(
    height=600,
    width=800,
    k_r=0.12,
    noise_rms=30.0,
    seed=7,
) -> np.ndarray:
    """
    A synthetic quasi-Bessel intensity pattern (J0^2 with a Gaussian
    envelope) plus shot-like noise, rendered as uint16 like Mono16 data.
    """

    rng = np.random.default_rng(seed)

    y, x = np.mgrid[0:height, 0:width]
    r = np.hypot(y - height / 2, x - width / 2)

    intensity = (j0(k_r * r) ** 2) * np.exp(-((r / (0.45 * width)) ** 2))
    scene = 3000.0 * intensity + rng.normal(0.0, noise_rms, size=(height, width))

    return np.clip(scene, 0, 65535).astype(np.uint16)


def cut_tiles(scene, tile_shape, corners):
    """Cut overlapping tiles whose top-left corners are `corners`."""

    tiles = []

    for (cy, cx) in corners:
        tiles.append(scene[cy : cy + tile_shape[0], cx : cx + tile_shape[1]].copy())

    return tiles


TILE_SHAPE = (256, 256)

# Simulates cranking the stage mostly along +x with a little y wobble.
CORNERS = [(150, 100), (160, 220), (148, 340), (155, 460)]


@pytest.fixture(scope="module")
def scene():
    return make_bessel_scene()


@pytest.fixture(scope="module")
def tiles(scene):
    return cut_tiles(scene, TILE_SHAPE, CORNERS)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_phase_correlation_recovers_known_shift(tiles):
    shift = stitcher.phase_correlation_shift(tiles[0], tiles[1])

    # Camera moved +120 px in x, +10 px in y between these tiles.
    assert shift.dy_px == pytest.approx(10.0, abs=1.0)
    assert shift.dx_px == pytest.approx(120.0, abs=1.0)
    assert shift.Confidence > 0.05


def test_phase_correlation_negative_and_zero_shift(scene):
    a = scene[100:356, 300:556]
    b = scene[80:336, 250:506]  # moved up and left

    shift = stitcher.phase_correlation_shift(a, b)

    assert shift.dy_px == pytest.approx(-20.0, abs=1.0)
    assert shift.dx_px == pytest.approx(-50.0, abs=1.0)

    identity = stitcher.phase_correlation_shift(a, a)
    assert identity.dy_px == pytest.approx(0.0, abs=0.5)
    assert identity.dx_px == pytest.approx(0.0, abs=0.5)


def test_phase_correlation_shift_larger_than_half_frame(scene):
    """
    Circular FFT correlation is ambiguous modulo the frame size; overlap
    validation must pick the right branch when the stage moves more than
    half the sensor width between saves.
    """

    a = scene[230 : 230 + 240, 40 : 40 + 240]
    b = scene[222 : 222 + 240, 190 : 190 + 240]  # dx = +150 > 240/2

    shift = stitcher.phase_correlation_shift(a, b)

    assert shift.dy_px == pytest.approx(-8.0, abs=1.0)
    assert shift.dx_px == pytest.approx(150.0, abs=1.0)
    assert shift.OverlapNCC > 0.9


def test_phase_correlation_ignores_fixed_pattern_noise():
    """
    Anything glued to the sensor (dust shadows, fixed-pattern noise) is
    identical in both frames and puts a spurious correlation peak at exactly
    (0, 0). On a smooth beam with a small stage step that peak can beat the
    true-motion peak — this is what broke the 2026-07-09 manual scans, where
    long runs of pairwise shifts collapsed to zero. Overlap validation of
    the top peaks must recover the real shift anyway.
    """

    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(3)

    scene = make_bessel_scene(noise_rms=0.0).astype(np.float64)

    # Dust-like static sensor pattern, identical in both frames.
    tile = 384
    fixed_pattern = gaussian_filter(rng.normal(size=(tile, tile)), sigma=1.0)
    fixed_pattern *= 20.0 / fixed_pattern.std()

    # A small stage step (~6% of the frame), like a manual micrometer scan.
    a = scene[100 : 100 + tile, 100 : 100 + tile] + fixed_pattern
    b = scene[105 : 105 + tile, 125 : 125 + tile] + fixed_pattern

    # The spurious (0, 0) peak really is the global maximum here: trusting
    # the single argmax (num_peaks=1) locks onto the fixed pattern.
    argmax_only = stitcher.phase_correlation_shift(a, b, num_peaks=1)
    assert abs(argmax_only.dy_px) < 1.0 and abs(argmax_only.dx_px) < 1.0

    # Validating the top peaks against the actual pixels recovers the
    # true stage motion.
    shift = stitcher.phase_correlation_shift(a, b)

    assert shift.dy_px == pytest.approx(5.0, abs=1.0)
    assert shift.dx_px == pytest.approx(25.0, abs=1.0)
    assert shift.OverlapNCC > 0.9


def test_phase_correlation_rejects_mismatched_shapes():
    with pytest.raises(stitcher.StitchError, match="identical shapes"):
        stitcher.phase_correlation_shift(
            np.zeros((10, 10)), np.zeros((12, 10))
        )


def test_opencv_fallback_if_available():
    """
    ORB feature matching is unreliable on self-similar concentric Bessel
    rings (which is exactly why phase correlation is the default) — so the
    OpenCV path is validated on a speckle-textured scene instead.
    """

    cv2 = pytest.importorskip("cv2")  # noqa: F841

    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(11)
    speckle = gaussian_filter(rng.normal(size=(600, 800)), sigma=3.0)
    speckle -= speckle.min()
    scene = (speckle / speckle.max() * 4000.0).astype(np.uint16)

    a = scene[150 : 150 + 256, 100 : 100 + 256]
    b = scene[160 : 160 + 256, 220 : 220 + 256]

    shift = stitcher.opencv_feature_shift(a, b)

    assert shift.Method == "OpenCV-ORB"
    assert shift.dy_px == pytest.approx(10.0, abs=2.0)
    assert shift.dx_px == pytest.approx(120.0, abs=2.0)


def test_estimate_shift_unknown_method(tiles):
    with pytest.raises(stitcher.StitchError, match="Unknown stitch method"):
        stitcher.estimate_shift(
            tiles[0], tiles[1], stitcher.StitchConfig(Method="patchmatch")
        )


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------


def test_stitch_frames_recovers_scene_geometry(tiles):
    result = stitcher.stitch_frames(tiles, stitcher.StitchConfig(Method="phase"))

    assert len(result.OffsetsPx) == len(tiles)
    assert len(result.PairwiseShifts) == len(tiles) - 1

    # Cumulative offsets should match the true corner displacements.
    true_offsets = [
        (cy - CORNERS[0][0], cx - CORNERS[0][1]) for (cy, cx) in CORNERS
    ]

    for (est_dy, est_dx), (true_dy, true_dx) in zip(result.OffsetsPx, true_offsets):
        assert est_dy == pytest.approx(true_dy, abs=1.0)
        assert est_dx == pytest.approx(true_dx, abs=1.0)

    # Canvas spans the union of tiles: x range 100..460+256 -> 616 wide.
    expected_h = (max(c[0] for c in CORNERS) - min(c[0] for c in CORNERS)) + TILE_SHAPE[0]
    expected_w = (max(c[1] for c in CORNERS) - min(c[1] for c in CORNERS)) + TILE_SHAPE[1]

    assert result.Composite.shape == (expected_h, expected_w)

    # The union of the tiles covers nearly the whole canvas (the canvas is
    # the bounding box of the union, so corners can be uncovered when the
    # stage wobbles in y).
    assert np.mean(result.Coverage > 0) > 0.95

    # Uncovered pixels must be exactly zero, not NaN/garbage.
    assert np.all(result.Composite[result.Coverage == 0] == 0.0)


def test_stitch_composite_matches_scene(scene, tiles):
    """The blended composite should closely reproduce the original scene."""

    result = stitcher.stitch_frames(tiles, stitcher.StitchConfig(Method="phase"))

    min_cy = min(c[0] for c in CORNERS)
    min_cx = min(c[1] for c in CORNERS)

    h, w = result.Composite.shape
    reference = scene[min_cy : min_cy + h, min_cx : min_cx + w].astype(np.float64)

    # Normalized RMS error small compared to the beam's dynamic range
    # (evaluated only where at least one tile contributed).
    covered = result.Coverage > 0
    rms = np.sqrt(np.mean((result.Composite[covered] - reference[covered]) ** 2))
    assert rms < 0.05 * reference.max()


def test_stitch_frames_requires_frames():
    with pytest.raises(stitcher.StitchError, match="No frames"):
        stitcher.stitch_frames([])


def test_stitch_single_frame_passthrough(tiles):
    result = stitcher.stitch_frames([tiles[0]])

    assert result.OffsetsPx == [(0.0, 0.0)]
    assert result.Composite.shape == tiles[0].shape
    np.testing.assert_allclose(result.Composite, tiles[0], rtol=1e-5)


# ---------------------------------------------------------------------------
# Run-directory integration
# ---------------------------------------------------------------------------


def make_fake_run_dir(tmp_path, tiles, with_manifest=True):
    run_dir = tmp_path / "2026-07-07-testrun"
    run_dir.mkdir()

    lines = []

    for i, tile in enumerate(tiles):
        path = run_dir / f"manual-stage-shot{i:04d}.npy"
        np.save(path, tile)
        lines.append(json.dumps({"Path": str(path), "ShotIndex": i}))

    if with_manifest:
        (run_dir / "frames.jsonl").write_text("\n".join(lines) + "\n")

    return run_dir


def test_stitch_run_dir_writes_outputs(tmp_path, tiles):
    run_dir = make_fake_run_dir(tmp_path, tiles)

    outputs = stitcher.stitch_run_dir(
        run_dir, stitcher.StitchConfig(Method="phase")
    )

    assert outputs["npy"].exists()
    assert outputs["png"].exists()
    assert outputs["offsets"].exists()

    composite = np.load(outputs["npy"])
    assert composite.ndim == 2
    assert composite.dtype == np.float32

    payload = json.loads(outputs["offsets"].read_text())
    assert len(payload["OffsetsPx"]) == len(tiles)
    assert len(payload["Frames"]) == len(tiles)
    assert payload["Config"]["Method"] == "phase"


def test_stitch_run_dir_without_manifest(tmp_path, tiles):
    run_dir = make_fake_run_dir(tmp_path, tiles, with_manifest=False)

    outputs = stitcher.stitch_run_dir(run_dir)
    assert outputs["png"].exists()


def test_stitch_run_dir_ignores_previous_composite(tmp_path, tiles):
    run_dir = make_fake_run_dir(tmp_path, tiles)

    stitcher.stitch_run_dir(run_dir)
    outputs = stitcher.stitch_run_dir(run_dir)  # re-stitch must not ingest composite.npy

    payload = json.loads(outputs["offsets"].read_text())
    assert len(payload["Frames"]) == len(tiles)


def test_stitch_run_dir_empty_raises(tmp_path):
    run_dir = tmp_path / "empty-run"
    run_dir.mkdir()

    with pytest.raises(stitcher.StitchError, match="No .npy frames"):
        stitcher.stitch_run_dir(run_dir)