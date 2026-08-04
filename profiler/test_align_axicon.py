"""
Tests for the axicon alignment tool: pure geometry/fit units plus the
full patrol session against a synthetic ring rendered by a fake camera
that answers with whatever the "gantry" is pointed at — no PySpin, no
hardware.

Rig geometry: a slightly elliptical annulus in the machine X-Z plane
with an azimuthal brightness modulation (the input-beam-decenter
signature), rendered at a fat 50 um "pixel" so a 256x192 fake sensor
spans 12.8 x 9.6 mm and tests stay fast.
"""

import importlib
import json
import math
import sys
import types

import numpy as np
import pytest

from conftest import FakeCamera, FakeCameraSettings, FakeNode


@pytest.fixture()
def modules(monkeypatch, fake_pyspin):
    fake_camera_settings_module = types.ModuleType("camera_settings")
    fake_camera_settings_module.FLIRCameraSettings = object
    fake_camera_settings_module.PixelFormatName = str
    monkeypatch.setitem(sys.modules, "camera_settings", fake_camera_settings_module)

    for name in ("align_axicon", "align_preview"):
        sys.modules.pop(name, None)

    return types.SimpleNamespace(
        dataset_writer=importlib.import_module("dataset_writer"),
        coordinates=importlib.import_module("coordinates"),
        align=importlib.import_module("align_axicon"),
        preview=importlib.import_module("align_preview"),
    )


# ---------------------------------------------------------------------------
# Synthetic ring rig
# ---------------------------------------------------------------------------


SENSOR_ROWS = 192
SENSOR_COLS = 256
PIXEL_UM = 50.0  # fat fake pixels: 256 px * 50 um = 12.8 mm wide frames

RING = dict(
    center=(60.0, -60.0),
    semi_major=4.6,
    semi_minor=4.4,
    tilt_deg=30.0,
    width_mm=0.35,
    modulation=0.30,
    bright_angle_deg=90.0,  # brightest at +Z (top), dimmest at the bottom
    amplitude=150.0,
    reference_exposure_us=10_000.0,
    background=5.0,
)

LIMITS_KW = dict(
    x_min_mm=0.0, x_max_mm=120.0,
    y_min_mm=0.0, y_max_mm=160.0,
    z_min_mm=-127.0, z_max_mm=3.0,
)


class RingScene:
    """Renders the machine-frame view of the ring at a gantry position."""

    def __init__(self, center_fn=None, scale_fn=None, **overrides):
        self.params = {**RING, **overrides}
        # center_fn(machine_y) -> (cx, cz): lets tests give the ring a
        # pointing tilt along the beam.
        self.center_fn = center_fn or (lambda y: self.params["center"])
        # scale_fn(machine_y) -> multiplier on the ring size: lets tests
        # model a diverging cone (radius growing along the beam).
        self.scale_fn = scale_fn or (lambda y: 1.0)

    def render(self, x_c, z_c, machine_y, exposure_us):
        p = self.params
        cx, cz = self.center_fn(machine_y)
        scale = self.scale_fn(machine_y)

        mm_per_px = PIXEL_UM / 1000.0
        width_mm = SENSOR_COLS * mm_per_px
        height_mm = SENSOR_ROWS * mm_per_px

        x = x_c - width_mm / 2.0 + (np.arange(SENSOR_COLS) + 0.5) * mm_per_px
        z = z_c + height_mm / 2.0 - (np.arange(SENSOR_ROWS) + 0.5) * mm_per_px
        X, Z = np.meshgrid(x, z)

        dx = X - cx
        dz = Z - cz
        phi = np.arctan2(dz, dx)
        rho = np.hypot(dx, dz)

        tilt = math.radians(p["tilt_deg"])
        a, b = p["semi_major"] * scale, p["semi_minor"] * scale
        phi_rot = phi - tilt
        ring_radius = a * b / np.sqrt(
            (b * np.cos(phi_rot)) ** 2 + (a * np.sin(phi_rot)) ** 2
        )

        amplitude = p["amplitude"] * (
            1.0
            + p["modulation"]
            * np.cos(phi - math.radians(p["bright_angle_deg"]))
        )
        intensity = amplitude * np.exp(
            -(((rho - ring_radius) / p["width_mm"]) ** 2)
        )

        counts = p["background"] + intensity * (
            exposure_us / p["reference_exposure_us"]
        )
        machine_frame = np.clip(counts, 0, 255).astype(np.uint8)

        # The real camera is mounted 180 deg rotated vs the machine axes
        # (composite.py FlipX+FlipZ, verified 2026-07-22): the RAW frame
        # is the machine view rotated 180, which orient() undoes.
        return machine_frame[::-1, ::-1]


class RingStage:
    """Stage fake that remembers where the gantry currently is."""

    def __init__(self):
        self.position = None
        self.moved_to = []

    def move_to_scan_point(self, point, signals):
        self.position = point.GantryPosition_mm
        self.moved_to.append(point.GantryPosition_mm)
        signals.MovementStarted.set()

    def wait_until_motion_complete(self, point, timeout_s, signals):
        signals.MovementComplete.set()


class RingCamera(FakeCamera):
    """Fake camera that images the scene at the stage's position."""

    def __init__(self, scene: RingScene, stage: RingStage):
        super().__init__(images=())
        self.scene = scene
        self.stage = stage
        self.Height = FakeNode(SENSOR_ROWS)
        self.Width = FakeNode(SENSOR_COLS)
        self.ExposureTime.SetValue(10_000.0)

    def GetNextImage(self, timeout_ms):
        from conftest import FakeImage

        position = self.stage.position
        assert position is not None, "grab before any move"
        arr = self.scene.render(
            position.x_mm,
            position.z_mm,
            position.y_mm,
            float(self.ExposureTime.GetValue()),
        )
        return FakeImage(arr)


def align_config(modules, **overrides):
    defaults = dict(
        MachineY_mm=20.0,
        Stations=8,
        PixelSize_um=PIXEL_UM,
        Downsample=4,
        SurveyDX_mm=3.0,
        MinSignalPixels=20,
        MaxRingShift_mm=3.0,
    )
    defaults.update(overrides)
    return modules.align.AlignConfig(**defaults)


def make_session(modules, tmp_path, scene=None, **config_overrides):
    scene = scene or RingScene()
    stage = RingStage()
    camera = RingCamera(scene, stage)

    writer = modules.dataset_writer.FLIRDatasetWriter(
        camera_index=0,
        cam=camera,
        camera_settings=FakeCameraSettings(),
        config=modules.dataset_writer.DatasetWriterConfig(
            JobType="align",
            DatasetRoot=tmp_path,
            TriggerArmDelay_s=0.0,
        ),
        stage_controller=stage,
    )
    writer.prepare_run()

    limits = modules.coordinates.Bounds3D(**LIMITS_KW)
    config = align_config(modules, **config_overrides)
    session = modules.align.AxiconAlignSession(writer, config, limits)
    return session, writer, scene


# ---------------------------------------------------------------------------
# Pure geometry units
# ---------------------------------------------------------------------------


def test_frame_axes_center_pixel_is_at_commanded_position(modules):
    x, z = modules.align.frame_axes_mm((10, 10), 60.0, -80.0, 0.2)

    # Even-sized frame: the position sits between the two middle pixels.
    assert x[4] < 60.0 < x[5]
    assert z[5] < -80.0 < z[4]  # row 0 is +Z
    assert np.isclose(x[1] - x[0], 0.2)
    assert np.isclose(z[0] - z[1], 0.2)


def test_fit_circle_recovers_known_circle(modules):
    theta = np.linspace(0.3, 5.5, 40)
    points = np.column_stack(
        (12.0 + 4.5 * np.cos(theta), -7.0 + 4.5 * np.sin(theta))
    )

    fit = modules.align.fit_circle(points)

    assert fit is not None
    assert np.isclose(fit.CenterX_mm, 12.0, atol=1e-6)
    assert np.isclose(fit.CenterZ_mm, -7.0, atol=1e-6)
    assert np.isclose(fit.Radius_mm, 4.5, atol=1e-6)
    assert fit.RMS_mm < 1e-9


def test_fit_ellipse_recovers_axes_and_angle(modules):
    tilt = math.radians(25.0)
    t = np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False)
    ex = 5.0 * np.cos(t)
    ez = 3.0 * np.sin(t)
    points = np.column_stack(
        (
            10.0 + ex * np.cos(tilt) - ez * np.sin(tilt),
            -4.0 + ex * np.sin(tilt) + ez * np.cos(tilt),
        )
    )

    fit = modules.align.fit_ellipse(points)

    assert fit is not None
    assert np.isclose(fit.CenterX_mm, 10.0, atol=1e-6)
    assert np.isclose(fit.CenterZ_mm, -4.0, atol=1e-6)
    assert np.isclose(fit.SemiMajor_mm, 5.0, atol=1e-6)
    assert np.isclose(fit.SemiMinor_mm, 3.0, atol=1e-6)
    assert np.isclose(fit.MajorAxisAngle_deg, 25.0, atol=1e-3)


def test_fit_ellipse_rejects_degenerate_input(modules):
    line = np.column_stack((np.linspace(0, 10, 20), np.linspace(0, 10, 20)))
    assert modules.align.fit_ellipse(line) is None


def test_azimuthal_uniformity_finds_bright_and_dim_sectors(modules):
    angles = np.arange(0.0, 360.0, 2.0)
    intensities = 100.0 * (1.0 + 0.3 * np.cos(np.radians(angles - 90.0)))

    metrics = modules.align.azimuthal_uniformity(angles, intensities)

    assert metrics is not None
    assert abs(metrics.BrightestAngle_deg - 90.0) <= 15.0
    assert abs(metrics.DimmestAngle_deg - 270.0) <= 15.0
    assert metrics.CoverageFraction == 1.0
    assert 0.5 < metrics.MinMaxRatio < 0.6  # 0.7/1.3 ~ 0.538


def test_solve_ring_from_chords_recovers_circle(modules):
    cx, cz, r = 10.0, -5.0, 6.0

    # Thin ring: two short lit intervals per column. The envelope chord
    # spans them plus the 0.05 half-width, so r reads ~0.05 high.
    def intervals(x):
        h = math.sqrt(r * r - (x - cx) ** 2)
        return [
            (cz - h - 0.05, cz - h + 0.05),
            (cz + h - 0.05, cz + h + 0.05),
        ]

    solution = modules.align.solve_ring_from_chords(
        9.0, intervals(9.0), 12.0, intervals(12.0)
    )

    assert solution is not None
    sx, sz, sr = solution
    assert np.isclose(sx, cx, atol=0.1)
    assert np.isclose(sz, cz, atol=1e-6)
    assert np.isclose(sr, r, atol=0.1)


def test_solve_ring_from_chords_handles_broad_annulus_single_intervals(modules):
    # Hardware regression (2026-07-27, axicon-2 band): a BROAD annulus
    # with a small hole gives ONE lit interval on columns that miss the
    # hole. That must read as a chord of the outer boundary, not as a
    # tangent (the old h=0 reading skewed the solve by ~7 mm).
    cx, cz, outer = 80.0, -86.5, 10.0

    def envelope(x):
        h = math.sqrt(outer * outer - (x - cx) ** 2)
        return [(cz - h, cz + h)]  # continuous light, hole missed

    solution = modules.align.solve_ring_from_chords(
        75.0, envelope(75.0), 70.0, envelope(70.0)
    )

    assert solution is not None
    sx, sz, sr = solution
    assert np.isclose(sx, cx, atol=1e-6)
    assert np.isclose(sz, cz, atol=1e-6)
    assert np.isclose(sr, outer, atol=1e-6)


def test_background_threshold_finds_dim_arc_in_filled_frame(modules):
    # A dim broad band filling most of the frame: the median IS signal,
    # so the fallback threshold blinds itself — the measured off-axis
    # background must recover the arc.
    config = align_config(modules)
    frame = np.full((48, 64), 10.0, dtype=np.float32)
    frame[:, :40] = 90.0  # dim band covering >60% of the frame

    blind = modules.align.extract_arc(
        frame, 60.0, -60.0, (55.0, -60.0), config
    )
    seen = modules.align.extract_arc(
        frame, 60.0, -60.0, (55.0, -60.0), config, background_level=12.0
    )

    assert blind is None
    assert seen is not None
    assert seen.LitPixels >= 48 * 40


def test_extract_arc_points_land_on_the_ring(modules):
    scene = RingScene()
    config = align_config(modules)
    cx, cz = RING["center"]

    # Frame centered on the ring's right edge, looking at a vertical arc.
    x_c, z_c = cx + RING["semi_major"], cz
    raw = scene.render(x_c, z_c, 20.0, 10_000.0)
    oriented = modules.align.orient(raw, config.composite_config())
    frame = modules.align.mean_pool(oriented, config.Downsample)

    arc = modules.align.extract_arc(frame, x_c, z_c, (cx, cz), config)

    assert arc is not None
    radii = np.hypot(arc.Points_mm[:, 0] - cx, arc.Points_mm[:, 1] - cz)
    ring_radii = np.linspace(RING["semi_minor"], RING["semi_major"], 2)
    assert radii.min() > ring_radii.min() - 0.5
    assert radii.max() < ring_radii.max() + 0.5


def test_compose_canvas_averages_overlap(modules):
    a = np.full((4, 4), 10.0, dtype=np.float32)
    b = np.full((4, 4), 30.0, dtype=np.float32)

    composed = modules.align.compose_canvas([(a, 0.0, 0.0), (b, 0.0, 0.0)], 1.0)

    assert composed is not None
    canvas, extent = composed
    assert canvas.shape == (4, 4)
    assert np.allclose(canvas, 20.0)
    assert extent == (-2.0, 2.0, -2.0, 2.0)


def test_fit_center_fixed_radius_recovers_center_from_short_arc(modules):
    # A 30-degree arc: a free-radius circle fit would be ill-conditioned,
    # the fixed-radius fit must still nail the center.
    theta = np.radians(np.linspace(75.0, 105.0, 20))
    points = np.column_stack(
        (60.0 + 4.5 * np.cos(theta), -60.0 + 4.5 * np.sin(theta))
    )

    fit = modules.align.fit_center_fixed_radius(
        points, 4.5, initial_center=(59.4, -60.7)
    )

    assert fit is not None
    cx, cz, rms = fit
    assert abs(cx - 60.0) < 1e-3
    assert abs(cz + 60.0) < 1e-3
    assert rms < 1e-6


def test_extract_arc_measures_ring_width(modules):
    scene = RingScene()
    config = align_config(modules)
    cx, cz = RING["center"]

    x_c, z_c = cx + RING["semi_major"], cz
    raw = scene.render(x_c, z_c, 20.0, 10_000.0)
    oriented = modules.align.orient(raw, config.composite_config())
    frame = modules.align.mean_pool(oriented, config.Downsample)

    arc = modules.align.extract_arc(frame, x_c, z_c, (cx, cz), config)

    assert arc is not None
    # Gaussian ring: FWHM = 2*sqrt(ln 2) * width_mm ~ 1.665 * 0.35 = 0.58.
    expected = 2.0 * math.sqrt(math.log(2.0)) * RING["width_mm"]
    assert abs(arc.WidthFWHM_mm - expected) < 0.2


# ---------------------------------------------------------------------------
# Session end-to-end against the rig
# ---------------------------------------------------------------------------


def run_cycles(session, n):
    results = []

    def on_cycle(result, frames):
        results.append((result, frames))
        return len(results) < n

    session.run(on_cycle, max_cycles=n)
    return results


def test_patrol_recovers_ring_center_radius_and_metrics(modules, tmp_path):
    session, writer, scene = make_session(modules, tmp_path)

    results = run_cycles(session, 3)

    assert len(results) == 3
    final, frames = results[-1]
    assert not final.Lost
    assert frames, "cycle should deliver preview frames"

    cx, cz = RING["center"]
    circle = final.Circle
    assert circle is not None
    assert abs(circle.CenterX_mm - cx) < 0.3
    assert abs(circle.CenterZ_mm - cz) < 0.3
    assert 4.1 < circle.Radius_mm < 4.9

    ellipse = final.Ellipse
    assert ellipse is not None
    expected_ratio = RING["semi_minor"] / RING["semi_major"]
    assert abs(ellipse.axis_ratio - expected_ratio) < 0.03

    uniformity = final.Uniformity
    assert uniformity is not None
    assert uniformity.CoverageFraction > 0.8
    assert abs(uniformity.BrightestAngle_deg - RING["bright_angle_deg"]) <= 30.0
    dim_delta = abs(uniformity.DimmestAngle_deg - 270.0)
    assert min(dim_delta, 360.0 - dim_delta) <= 45.0


def test_reference_offset_reads_zero_then_tracks_center_shift(modules, tmp_path):
    session, writer, scene = make_session(modules, tmp_path)

    run_cycles(session, 1)
    session.set_reference_here()

    # Nudge the ring: simulates turning an adjuster between cycles.
    scene.params["center"] = (60.4, -60.0)

    (result, _), = run_cycles(session, 1)

    assert result.Offset_mm is not None
    assert abs(result.Offset_mm[0] - 0.4) < 0.1
    assert abs(result.Offset_mm[1]) < 0.1


def test_bootstrap_tightens_survey_dx_for_compact_beam(modules, tmp_path):
    """
    Focused-Bessel-region regression (hardware 2026-08-03): the whole
    pattern is < 1 mm in radius, so BOTH second survey columns at the
    configured +/-SurveyDX miss it entirely — the survey must retry
    with a spacing scaled to the extent column 1 measured, instead of
    dying with "only one usable survey column".
    """

    scene = RingScene(
        center=(60.0, -60.0),
        semi_major=0.45,
        semi_minor=0.43,
        width_mm=0.18,
    )
    session, writer, scene = make_session(
        modules,
        tmp_path,
        scene=scene,
        Downsample=2,  # 0.1 mm/px: resolve the small ring
        MinSignalPixels=10,
        SurveyDX_mm=3.0,  # steps clear over the ~0.8 mm-radius spot
    )

    estimate = session.bootstrap(session.config.MachineY_mm)

    assert abs(estimate.CenterX_mm - 60.0) < 0.4
    assert abs(estimate.CenterZ_mm - (-60.0)) < 0.4
    assert estimate.Radius_mm < 1.5


def test_reference_snapshot_name_encodes_coords(modules):
    align = modules.align

    # No reference yet (r pressed before the first fit): nothing to save.
    assert align.reference_snapshot_name({}) is None

    single = align.reference_snapshot_name({20.0: (60.1234, -59.8764)})
    assert single == "preview_r=X60.123_Z-59.876.png"

    # Two-plane patrol: one Y-labelled group per plane, sorted by Y.
    double = align.reference_snapshot_name(
        {130.0: (60.4, -59.9), 10.0: (60.1234, -59.8764)}
    )
    assert double == (
        "preview_r=Y10_X60.123_Z-59.876__Y130_X60.400_Z-59.900.png"
    )


def test_reference_snapshot_name_tracks_session_reference(modules, tmp_path):
    session, writer, scene = make_session(modules, tmp_path)

    assert modules.align.reference_snapshot_name(session.references) is None

    run_cycles(session, 1)
    session.set_reference_here()

    name = modules.align.reference_snapshot_name(session.references)
    assert name is not None
    assert name.startswith("preview_r=") and name.endswith(".png")

    cx, cz = session.references[session.config.MachineY_mm]
    assert f"X{cx:.3f}" in name
    assert f"Z{cz:.3f}" in name


def test_two_plane_patrol_reports_pointing_tilt(modules, tmp_path):
    slope = 0.01  # mm of center-X per mm of Y -> 10 mrad

    def center_fn(machine_y):
        return (60.0 + slope * (machine_y - 20.0), -60.0)

    session, writer, scene = make_session(
        modules,
        tmp_path,
        scene=RingScene(center_fn=center_fn),
        MachineY2_mm=60.0,
        BeamDirectionSign=1,
    )

    results = run_cycles(session, 4)

    planes = [r.MachineY_mm for r, _ in results]
    assert planes == [20.0, 60.0, 20.0, 60.0]

    tilt = results[-1][0].Tilt
    assert tilt is not None
    assert abs(tilt.TiltX_mrad - 10.0) < 1.0
    assert abs(tilt.TiltZ_mrad) < 1.0


def test_diverging_cone_two_planes_reports_cone_angle(modules, tmp_path):
    # After axicon 1 the annulus grows along the beam: ~20%/40 mm here
    # (r 4.5 -> 5.4 between Y20 and Y60 => +22.5 mrad cone with the
    # beam along +Y). Per-plane ring-diameter priors must both accept,
    # and the cone readout must recover the spread rate.
    def scale_fn(machine_y):
        return 1.0 + 0.005 * (machine_y - 20.0)

    session, writer, scene = make_session(
        modules,
        tmp_path,
        scene=RingScene(scale_fn=scale_fn),
        MachineY2_mm=60.0,
        BeamDirectionSign=1,
        RingDiameter_mm=9.0,
        RingDiameter2_mm=10.8,
    )

    results = run_cycles(session, 4)

    r1 = session.estimates[20.0].Radius_mm
    r2 = session.estimates[60.0].Radius_mm
    assert r2 > r1 + 0.5  # the cone genuinely diverges

    tilt = results[-1][0].Tilt
    assert tilt is not None
    expected_cone = (r2 - r1) / 40.0 * 1000.0
    assert tilt.Cone_mrad is not None
    assert abs(tilt.Cone_mrad - expected_cone) < 1.0
    assert abs(tilt.Cone_mrad - 22.5) < 5.0


def test_sane_radius_uses_per_plane_prior(modules, tmp_path):
    session, writer, scene = make_session(
        modules,
        tmp_path,
        MachineY2_mm=60.0,
        RingDiameter_mm=9.0,
        RingDiameter2_mm=30.0,
    )

    # 4.5 mm radius: sane vs the 9 mm-diameter prior at plane 1, but
    # far below the 30 mm prior at plane 2 -> replaced by that prior.
    assert session._sane_radius(4.5, 20.0) == 4.5
    assert session._sane_radius(4.5, 60.0) == 15.0


def test_bootstrap_raises_when_beam_is_off(modules, tmp_path):
    session, writer, scene = make_session(
        modules, tmp_path, scene=RingScene(amplitude=0.0)
    )

    with pytest.raises(modules.align.AlignError):
        session.bootstrap(20.0)


def test_cycle_reports_lost_when_beam_disappears(modules, tmp_path):
    session, writer, scene = make_session(modules, tmp_path)
    run_cycles(session, 1)

    scene.params["amplitude"] = 0.0
    (result, _), = run_cycles(session, 1)

    assert result.Lost
    assert result.Circle is None


def test_exposure_servo_halves_on_saturation(modules, tmp_path):
    # 4x the amplitude: frames saturate at the bootstrap exposure.
    session, writer, scene = make_session(
        modules, tmp_path, scene=RingScene(amplitude=600.0)
    )

    run_cycles(session, 1)
    exposure_after_first = session.inner._current_exposure_us

    run_cycles(session, 1)

    # The servo (or the bootstrap calibration) must have pulled exposure
    # down between cycles rather than letting stations stay saturated.
    assert session.inner._current_exposure_us <= exposure_after_first


def test_cycle_log_is_valid_jsonl(modules, tmp_path):
    session, writer, scene = make_session(modules, tmp_path)
    (result, _), = run_cycles(session, 1)

    log_path = tmp_path / "alignment_log.jsonl"
    modules.align.append_cycle_log(log_path, result)

    payload = json.loads(log_path.read_text().splitlines()[0])
    assert payload["Index"] == 0
    assert payload["Circle"]["Radius_mm"] == pytest.approx(
        result.Circle.Radius_mm
    )
    assert "Points_mm" not in payload
    assert payload["NPoints"] > 0


def test_disk_cover_images_the_center_without_polluting_the_fit(modules, tmp_path):
    session, writer, scene = make_session(modules, tmp_path, CoverMode="disk")

    (result, frames), = run_cycles(session, 1)

    cx, cz = RING["center"]

    # Interior stations exist: half-radius ring + one at the center.
    fill = [s for s in result.Stations if s.Role == "fill"]
    assert len(fill) == 8 // 2 + 1
    # One fill station sits at the (bootstrap-accuracy) ring center.
    assert any(
        abs(s.X_mm - cx) < 2.0 and abs(s.Z_mm - cz) < 2.0 for s in fill
    )

    # The composite now covers the ring center.
    composed = modules.align.compose_canvas(
        frames, align_config(modules).mm_per_px()
    )
    assert composed is not None
    canvas, (x_min, x_max, z_min, z_max) = composed
    assert x_min < cx < x_max
    assert z_min < cz < z_max

    # The center PIXEL is real data; un-imaged pixels are NaN (rendered
    # gray by the preview), never zero pretending to be dark signal.
    mm_per_px = align_config(modules).mm_per_px()
    row = int((z_max - cz) / mm_per_px)
    col = int((cx - x_min) / mm_per_px)
    assert np.isfinite(canvas[row, col])
    assert np.isnan(canvas[0, 0])  # canvas corner: never visited

    # Fill frames must not pollute the ring geometry.
    assert result.Circle is not None
    assert abs(result.Circle.CenterX_mm - cx) < 0.3
    assert abs(result.Circle.CenterZ_mm - cz) < 0.3
    assert 4.1 < result.Circle.Radius_mm < 4.9


def test_broad_band_bootstrap_and_first_lap_converge(modules, tmp_path):
    # Wide band (2 mm 1/e half-width on a 4.5 mm radius: lit from ~2 to
    # ~7 mm, small dark hole) — the regime that broke the thin-ring
    # chord logic on hardware. The first lap must land on the true
    # center by adopting the fit unclamped.
    session, writer, scene = make_session(
        modules, tmp_path, scene=RingScene(width_mm=2.0)
    )

    results = run_cycles(session, 2)

    cx, cz = RING["center"]
    first, _ = results[0]
    assert first.Circle is not None

    # Unclamped adoption: cycle 0's post-update estimate IS its fit.
    assert first.Estimate.CenterX_mm == pytest.approx(
        first.Circle.CenterX_mm
    )

    final, _ = results[-1]
    assert abs(final.Circle.CenterX_mm - cx) < 0.4
    assert abs(final.Circle.CenterZ_mm - cz) < 0.4


def test_max_exposure_caps_bootstrap_and_servo(modules, tmp_path):
    # A very dim scene calibrates to a long exposure; the cap must clamp
    # it and keep the ring tracked (background-referenced thresholds
    # make dim arcs detectable).
    scene = RingScene(amplitude=40.0)
    session, writer, _ = make_session(
        modules, tmp_path, scene=scene, MaxExposure_us=30_000.0
    )

    (result, _), = run_cycles(session, 1)

    assert session.inner._current_exposure_us <= 30_000.0
    assert result.Circle is not None


def test_bootstrap_captures_offaxis_background(modules, tmp_path):
    session, writer, scene = make_session(modules, tmp_path)

    run_cycles(session, 1)

    assert session._background is not None
    # The rig's ambient is 5 counts; p99 of a dark corner frame ~ 5.
    assert session._background["P99"] < 15.0
    assert session.background_level() is not None


# ---------------------------------------------------------------------------
# Park-and-stream mode
# ---------------------------------------------------------------------------


def test_stream_parks_orbits_once_and_tracks_center_shift(modules, tmp_path):
    session, writer, scene = make_session(modules, tmp_path)

    cycles = []
    samples = []

    def on_cycle(result, frames):
        cycles.append(result)

    def on_frame(sample, frame):
        samples.append(sample)
        if sample.Index == 0:
            session.set_reference_here()
        if sample.Index == 2:
            # Turn an adjuster: the ring shifts mid-stream.
            scene.params["center"] = (60.3, -60.0)
        return modules.align.STREAM_CONTINUE

    session.run_stream(on_frame, on_cycle=on_cycle, max_frames=8)

    assert len(cycles) == 1, "exactly one bootstrap orbit expected"
    assert len(samples) == 8
    assert all(not s.Lost for s in samples)

    # The parked fixed-radius fit reads ~zero offset before the nudge...
    settled_before = samples[1]
    assert settled_before.Offset_mm is not None
    assert abs(settled_before.Offset_mm[0]) < 0.15
    # ...and tracks the 0.3 mm X shift after it.
    settled_after = samples[-1]
    assert abs(settled_after.Offset_mm[0] - 0.3) < 0.15
    assert abs(settled_after.Offset_mm[1]) < 0.15

    # Width readout is live in stream mode.
    expected_fwhm = 2.0 * math.sqrt(math.log(2.0)) * RING["width_mm"]
    assert abs(samples[-1].WidthFWHM_mm - expected_fwhm) < 0.25


def test_stream_orbit_action_runs_a_fresh_lap(modules, tmp_path):
    session, writer, scene = make_session(modules, tmp_path)

    cycles = []

    def on_frame(sample, frame):
        if sample.Index == 1:
            return modules.align.STREAM_ORBIT
        return modules.align.STREAM_CONTINUE

    session.run_stream(
        on_frame, on_cycle=lambda r, f: cycles.append(r), max_frames=4
    )

    assert len(cycles) == 2  # bootstrap orbit + requested orbit


def test_stream_survives_beam_loss_without_raising(modules, tmp_path):
    session, writer, scene = make_session(modules, tmp_path)

    samples = []

    def on_frame(sample, frame):
        samples.append(sample)
        if sample.Index == 0:
            scene.params["amplitude"] = 0.0  # beam blocked mid-stream
        return modules.align.STREAM_CONTINUE

    # Loss triggers an orbit, the orbit triggers a re-find, the re-find
    # fails (beam off) — the stream must keep going, not crash.
    session.run_stream(on_frame, max_frames=10)

    assert len(samples) == 10
    assert samples[-1].Lost


# ---------------------------------------------------------------------------
# Preview (headless Agg backend)
# ---------------------------------------------------------------------------


def test_preview_renders_and_saves_headless(modules, tmp_path):
    session, writer, scene = make_session(modules, tmp_path)
    (result, frames), = run_cycles(session, 1)

    preview = modules.preview.AlignPreview(
        align_config(modules), display=False
    )
    try:
        preview.on_station(result.Stations[0], frames[0][0])
        preview.update(result, frames)
        saved = preview.save_png(tmp_path / "preview.png")
        assert saved is not None
        assert (tmp_path / "preview.png").stat().st_size > 0
    finally:
        preview.close()


def test_preview_stream_update_renders_headless(modules, tmp_path):
    session, writer, scene = make_session(modules, tmp_path)

    preview = modules.preview.AlignPreview(
        align_config(modules), display=False
    )
    captured = []

    def on_cycle(result, frames):
        preview.update(result, frames)

    def on_frame(sample, frame):
        preview.update_stream(sample, frame)
        captured.append(sample)
        return modules.align.STREAM_CONTINUE

    try:
        session.run_stream(on_frame, on_cycle=on_cycle, max_frames=2)
        saved = preview.save_png(tmp_path / "stream.png")
        assert saved is not None
        assert len(captured) == 2
    finally:
        preview.close()


# ---------------------------------------------------------------------------
# Core mode (axicon 3): live J0^2 fits, Y jog
# ---------------------------------------------------------------------------

CORE_K_SCENE = 6012.0   # rad/m -> first zero 400 um = 8 fake pixels


class CoreScene:
    """Bessel core fixed at machine (60, -60): A*J0(k r)^2 * envelope."""

    def __init__(self, k_per_m=CORE_K_SCENE, amplitude=150.0, background=5.0,
                 reference_exposure_us=10_000.0):
        self.params = dict(k=k_per_m, amplitude=amplitude,
                           background=background,
                           reference_exposure_us=reference_exposure_us)

    def render(self, x_c, z_c, machine_y, exposure_us):
        from scipy.special import j0

        p = self.params
        mm_per_px = PIXEL_UM / 1000.0
        width_mm = SENSOR_COLS * mm_per_px
        height_mm = SENSOR_ROWS * mm_per_px
        x = x_c - width_mm / 2.0 + (np.arange(SENSOR_COLS) + 0.5) * mm_per_px
        z = z_c + height_mm / 2.0 - (np.arange(SENSOR_ROWS) + 0.5) * mm_per_px
        X, Z = np.meshgrid(x, z)
        r_m = np.hypot(X - 60.0, Z + 60.0) * 1e-3
        intensity = (
            p["amplitude"]
            * j0(p["k"] * r_m) ** 2
            * np.exp(-((r_m / 3e-3) ** 2))
        )
        counts = p["background"] + intensity * (
            exposure_us / p["reference_exposure_us"]
        )
        machine_frame = np.clip(counts, 0, 255).astype(np.uint8)
        return machine_frame[::-1, ::-1]  # camera 180 deg vs machine axes


def core_config_overrides():
    return dict(
        CoreKrIdeal_per_m=CORE_K_SCENE * 0.95,  # scene sits +5.3% vs "ideal"
        CoreCropRadius_um=3000.0,
    )


def test_analyze_core_recovers_k_from_native_frame(modules):
    scene = CoreScene()
    config = align_config(modules, **core_config_overrides())
    raw = scene.render(60.0, -60.0, 20.0, 10_000.0)
    frame = modules.align.orient(raw, config.composite_config())

    result = modules.align.analyze_core(frame, 10_000.0, 5.0, config)

    assert result is not None
    assert not result["saturated"]
    assert abs(result["kr"] / CORE_K_SCENE - 1) < 0.02
    assert abs(result["kx"] / CORE_K_SCENE - 1) < 0.05
    assert abs(result["kz"] / CORE_K_SCENE - 1) < 0.05
    assert result["rms_over_a"] < 0.06
    # Centroid lands on the core (machine 60,-60 = frame center here).
    assert abs(result["centroid_col"] - (SENSOR_COLS - 1) / 2) < 2.0
    assert abs(result["centroid_row"] - (SENSOR_ROWS - 1) / 2) < 2.0


def test_run_core_streams_fits_and_jogs_y(modules, tmp_path):
    session, writer, scene = make_session(
        modules, tmp_path, scene=CoreScene(), **core_config_overrides()
    )

    samples = []

    def on_frame(sample, frame):
        samples.append(sample)
        if sample.Index == 1:
            return modules.align.CORE_JOG_UP
        if sample.Index == 3:
            return modules.align.CORE_JOG_DOWN
        if sample.Index >= 5:
            return modules.align.CORE_STOP
        return modules.align.CORE_CONTINUE

    session.run_core(on_frame)

    assert len(samples) == 6
    assert all(not s.Lost for s in samples)
    # Jogs: +10 mm after index 1, -10 mm after index 3 (default step).
    assert samples[1].MachineY_mm == 20.0
    assert samples[2].MachineY_mm == 30.0
    assert samples[4].MachineY_mm == 20.0
    # Fits track the scene k at every frame.
    for s in samples:
        assert s.Kr_per_m is not None
        assert abs(s.Kr_per_m / CORE_K_SCENE - 1) < 0.02
        assert s.XChord is not None and s.Radial is not None


def test_run_core_jog_clamps_to_machine_envelope(modules, tmp_path):
    session, writer, scene = make_session(
        modules, tmp_path, scene=CoreScene(),
        MachineY_mm=155.0, **core_config_overrides()
    )

    seen = []

    def on_frame(sample, frame):
        seen.append(sample.MachineY_mm)
        if sample.Index == 0:
            return modules.align.CORE_JOG_UP   # 165 > y_max 160 -> clamp
        return modules.align.CORE_STOP

    session.run_core(on_frame)

    assert seen == [155.0, 160.0]


def test_core_preview_renders_and_saves_headless(modules, tmp_path):
    session, writer, scene = make_session(
        modules, tmp_path, scene=CoreScene(), **core_config_overrides()
    )
    preview = modules.preview.CorePreview(
        align_config(modules, **core_config_overrides()), display=False
    )

    captured = []

    def on_frame(sample, frame):
        preview.update_core(sample, frame)
        captured.append(sample)
        return (modules.align.CORE_STOP if sample.Index >= 1
                else modules.align.CORE_CONTINUE)

    try:
        session.run_core(on_frame)
        assert preview.save_png(tmp_path / "core.png") is not None
        assert (tmp_path / "core.png").stat().st_size > 0
        assert len(captured) == 2
    finally:
        preview.close()


# ---------------------------------------------------------------------------
# Free-stream mode (live view + manual jogging, no bootstrap)
# ---------------------------------------------------------------------------


def test_run_free_streams_and_jogs_all_axes(modules, tmp_path):
    session, writer, scene = make_session(modules, tmp_path)

    samples = []
    script = {
        0: modules.align.free_jog("x", +1),
        1: modules.align.free_jog("y", -1),
        2: modules.align.free_jog("z", +1),
        3: modules.align.FREE_STEP_UP,
        4: modules.align.free_jog("x", -1),
    }

    def on_frame(sample, frame):
        samples.append(sample)
        if sample.Index >= 5:
            return modules.align.FREE_STOP
        return script.get(sample.Index, modules.align.FREE_CONTINUE)

    session.run_free(on_frame, auto_exposure=False)

    positions = [
        (s.MachineX_mm, s.MachineY_mm, s.MachineZ_mm) for s in samples
    ]
    # Defaults: X/Z envelope centers (60, -62), Y from config (20).
    assert positions == [
        (60.0, 20.0, -62.0),
        (70.0, 20.0, -62.0),   # x +10
        (70.0, 10.0, -62.0),   # y -10
        (70.0, 10.0, -52.0),   # z +10
        (70.0, 10.0, -52.0),   # step doubled: no motion
        (50.0, 10.0, -52.0),   # x -20 with the doubled step
    ]
    assert samples[3].JogStep_mm == 10.0
    assert samples[4].JogStep_mm == 20.0
    assert not any(s.NoFrame for s in samples)

    # The gantry really was commanded to each position.
    last = writer.stage_controller.moved_to[-1]
    assert (last.x_mm, last.y_mm, last.z_mm) == (50.0, 10.0, -52.0)


def test_run_free_clamps_jogs_and_manual_exposure(modules, tmp_path):
    session, writer, scene = make_session(modules, tmp_path)

    samples = []

    def on_frame(sample, frame):
        samples.append(sample)
        if sample.Index == 0:
            return modules.align.free_jog("x", +1)  # 115+10 -> clamp 120
        if sample.Index == 1:
            return modules.align.FREE_EXP_DOWN
        if sample.Index == 2:
            return modules.align.FREE_EXP_AUTO
        return modules.align.FREE_STOP

    session.run_free(
        on_frame, start_x_mm=115.0, exposure_us=9000.0, auto_exposure=False
    )

    assert samples[1].MachineX_mm == 120.0  # clamped to the envelope
    assert samples[2].Exposure_us == pytest.approx(6000.0)  # 9000 / 1.5
    assert samples[2].AutoExposure is False
    assert samples[3].AutoExposure is True  # a toggled it back on


def test_run_free_auto_exposure_halves_on_saturation(modules, tmp_path):
    session, writer, scene = make_session(modules, tmp_path)

    samples = []

    def on_frame(sample, frame):
        samples.append(sample)
        return (modules.align.FREE_STOP if sample.Index >= 1
                else modules.align.FREE_CONTINUE)

    session.run_free(
        on_frame, start_x_mm=60.0, start_z_mm=-62.0, exposure_us=40_000.0
    )

    assert samples[0].Saturated
    assert samples[0].Exposure_us == pytest.approx(40_000.0)
    assert samples[1].Exposure_us == pytest.approx(20_000.0)


def test_free_preview_renders_saves_and_queues_key_actions(
    modules, tmp_path
):
    session, writer, scene = make_session(modules, tmp_path)
    preview = modules.preview.FreePreview(
        align_config(modules), display=False
    )

    def on_frame(sample, frame):
        preview.update_free(sample, frame)
        return (modules.align.FREE_STOP if sample.Index >= 1
                else modules.align.FREE_CONTINUE)

    class FakeEvent:
        def __init__(self, key):
            self.key = key

    try:
        session.run_free(on_frame)
        assert preview.save_png(tmp_path / "free.png") is not None
        assert (tmp_path / "free.png").stat().st_size > 0

        # Window keys queue the shared FREE_KEY_ACTIONS; unknown keys
        # are ignored; q flips the quit flag instead of queueing.
        for key in ("left", ".", "x", "q"):
            preview._on_key(FakeEvent(key))
        assert preview.pending_actions == [
            modules.align.free_jog("x", -1),
            modules.align.free_jog("y", +1),
        ]
        assert preview.quit_requested
        assert preview.pop_action() == modules.align.free_jog("x", -1)
        assert preview.pop_action() == modules.align.free_jog("y", +1)
        assert preview.pop_action() is None
    finally:
        preview.close()
