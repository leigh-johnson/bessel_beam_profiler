"""
Tests for per-slice subfolder output in the dataset writer and for the
AutoScanSession workflow (per-Y calibration, backgrounds, X-Z raster),
using the shared fake camera scaffolding — no gantry, no PySpin.

Coordinate convention: X horizontal transverse, Y beam propagation,
Z vertical. Slices are X-Z rasters stepped along machine Y.
"""

import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from conftest import FakeCamera, FakeCameraSettings, FakeImage


@pytest.fixture()
def modules(monkeypatch, fake_pyspin):
    fake_camera_settings_module = types.ModuleType("camera_settings")
    fake_camera_settings_module.FLIRCameraSettings = object
    fake_camera_settings_module.PixelFormatName = str
    monkeypatch.setitem(sys.modules, "camera_settings", fake_camera_settings_module)

    return types.SimpleNamespace(
        dataset_writer=importlib.import_module("dataset_writer"),
        coordinates=importlib.import_module("coordinates"),
        auto_scan=importlib.import_module("auto_scan"),
    )


class FastStageController:
    def __init__(self):
        self.moved_to = []

    def move_to_scan_point(self, point, signals):
        self.moved_to.append(point.GantryPosition_mm)
        signals.MovementStarted.set()

    def wait_until_motion_complete(self, point, timeout_s, signals):
        signals.MovementComplete.set()


def make_writer(modules, tmp_path, images, job_type="auto_scan", **config_kwargs):
    writer = modules.dataset_writer.FLIRDatasetWriter(
        camera_index=0,
        cam=FakeCamera(images=images),
        camera_settings=FakeCameraSettings(),
        config=modules.dataset_writer.DatasetWriterConfig(
            JobType=job_type,
            DatasetRoot=tmp_path,
            TriggerArmDelay_s=0.0,  # keep the fake-camera tests fast
            **config_kwargs,
        ),
        stage_controller=FastStageController(),
    )
    writer.prepare_run()
    return writer


def make_point(modules, *, machine_y_mm=20.0, beam_y_mm=1000.0, metadata=None):
    return modules.coordinates.ScanPoint(
        PlacementID="placement-01",
        GantryPosition_mm=modules.coordinates.Vec3D(60.0, machine_y_mm, -80.0),
        TablePosition_mm=modules.coordinates.Vec3D(60.0, beam_y_mm, -80.0),
        NShots=1,
        Metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Dataset writer: subfolders, per-subfolder manifests, file tags
# ---------------------------------------------------------------------------


def test_explicit_subfolder_groups_frames_and_manifests(modules, tmp_path):
    arr = np.full((2, 2), 7, dtype=np.uint8)
    writer = make_writer(modules, tmp_path, images=[FakeImage(arr)])

    point = make_point(modules, metadata={"Subfolder": "y0100.00cm"})
    records = writer.acquire_scan([point])

    saved = modules.dataset_writer.Path(records[0].Path)
    assert saved.parent == writer.run_dir / "y0100.00cm"
    assert saved.exists()

    # Both the run-level and the subfolder-level manifest record the frame.
    root_lines = (writer.run_dir / "frames.jsonl").read_text().splitlines()
    sub_lines = (
        writer.run_dir / "y0100.00cm" / "frames.jsonl"
    ).read_text().splitlines()

    assert len(root_lines) == 1
    assert len(sub_lines) == 1
    assert json.loads(root_lines[0]) == json.loads(sub_lines[0])


def test_group_by_y_config_flag_names_subfolder_from_table_y(modules, tmp_path):
    arr = np.zeros((2, 2), dtype=np.uint8)
    writer = make_writer(
        modules,
        tmp_path,
        images=[FakeImage(arr)],
        GroupByYSubfolder=True,
    )

    records = writer.acquire_scan([make_point(modules, beam_y_mm=1000.0)])

    assert (
        modules.dataset_writer.Path(records[0].Path).parent
        == writer.run_dir / "y_p001000.000mm"
    )


def test_no_subfolder_by_default(modules, tmp_path):
    arr = np.zeros((2, 2), dtype=np.uint8)
    writer = make_writer(modules, tmp_path, images=[FakeImage(arr)])

    records = writer.acquire_scan([make_point(modules)])

    assert modules.dataset_writer.Path(records[0].Path).parent == writer.run_dir


def test_filename_records_beam_y_from_table_position(modules, tmp_path):
    arr = np.zeros((2, 2), dtype=np.uint8)
    writer = make_writer(modules, tmp_path, images=[FakeImage(arr)])

    records = writer.acquire_scan([make_point(modules, beam_y_mm=1000.0)])

    assert "tableyp001000.000mm" in records[0].Path


def test_file_tag_keeps_same_position_frames_unique(modules, tmp_path):
    arr = np.zeros((2, 2), dtype=np.uint8)
    writer = make_writer(
        modules, tmp_path, images=[FakeImage(arr), FakeImage(arr)]
    )

    record_a = writer.acquire_at_current_position(
        make_point(modules, metadata={"FileTag": "exp000100.0us"})
    )[0]
    record_b = writer.acquire_at_current_position(
        make_point(modules, metadata={"FileTag": "exp000200.0us"})
    )[0]

    assert record_a.Path != record_b.Path
    assert "exp000100.0us" in record_a.Path


# ---------------------------------------------------------------------------
# AutoScanSession: full placement workflow against the fake camera
# ---------------------------------------------------------------------------


def make_session(
    modules,
    tmp_path,
    n_images=32,
    background_mode="ladder",
    background_x_mm=None,
    background_z_mm=None,
    background_exposure_change=0.0,  # most tests: capture at every slice
    raster_mode="fixed",
    min_signal_pixels=50,
    find_beam=False,  # legacy tests pre-date the sweep; explicit tests opt in
    images=None,
):
    if images is None:
        # Mid-brightness frames: instantly inside the calibration window.
        images = [
            FakeImage(np.full((4, 4), 150, dtype=np.uint8))
            for _ in range(n_images)
        ]
    writer = make_writer(modules, tmp_path, images=images)

    config = modules.auto_scan.AutoScanConfig(
        PlacementID="placement-01",
        MeasuredSensorY_mm=1000.0,  # 100 cm after axicon3 at YStart
        YStart_machine_mm=20.0,
        YStop_machine_mm=30.0,
        YStep_mm=10.0,  # -> machine Y 20, 30 -> beam y 100 cm, 101 cm
        BeamDirectionSign=1,  # tests use +Y = downstream for simple sums
        X=modules.coordinates.AxisRange(start_mm=55.0, stop_mm=60.0, step_mm=5.0),
        Z=modules.coordinates.AxisRange(start_mm=-80.0, stop_mm=-80.0, step_mm=5.0),
        NShots=1,
        RasterMode=raster_mode,
        MinSignalPixels=min_signal_pixels,
        FindBeam=find_beam,
        BackgroundMode=background_mode,
        BackgroundX_mm=background_x_mm,
        BackgroundZ_mm=background_z_mm,
        BackgroundExposureChangeFraction=background_exposure_change,
        BackgroundExposures_us=(100.0, 1000.0),
        BackgroundShots=1,
    )

    pauses = []
    session = modules.auto_scan.AutoScanSession(
        writer,
        config,
        pause_fn=pauses.append,
    )
    return session, writer, pauses


LIMITS_KW = dict(
    x_min_mm=0.0, x_max_mm=120.0,
    y_min_mm=0.0, y_max_mm=160.0,
    z_min_mm=-127.0, z_max_mm=3.0,
)

# Calibration point is (X 57.5, Z -80); the farthest machine-limit X/Z
# corner is (120, 3).
EXPECTED_BG_CORNER = (120.0, 3.0)


def load_manifest(writer):
    return [
        json.loads(line)
        for line in (writer.run_dir / "frames.jsonl").read_text().splitlines()
    ]


def test_ladder_run_layout(modules, tmp_path):
    session, writer, pauses = make_session(modules, tmp_path)
    limits = modules.coordinates.Bounds3D(**LIMITS_KW)

    records = session.run(limits)

    # 2 ladder rungs x 1 shot + 2 Y-slices x (2 X-Z points x 1 shot)
    assert len(records) == 2 + 4

    run_dir = writer.run_dir

    # Ladder backgrounds: one file per rung, tagged by exposure.
    background_files = sorted(
        p.name for p in (run_dir / "background").glob("*.npy")
    )
    assert len(background_files) == 2
    assert any("exp0000100.0us" in name for name in background_files)
    assert any("exp0001000.0us" in name for name in background_files)

    # Per-slice folders named from the distance along the beam: measured
    # 1000 mm at machine Y=20, so Y=20 -> y0100.00cm and Y=30 -> y0101.00cm.
    for y_name in ("y0100.00cm", "y0101.00cm"):
        slice_dir = run_dir / y_name
        assert slice_dir.is_dir(), f"missing {y_name}"
        assert len(list(slice_dir.glob("*.npy"))) == 2
        assert (slice_dir / "frames.jsonl").exists()

        calibration = json.loads(
            (slice_dir / "calibration_result.json").read_text()
        )
        assert calibration["Converged"] is True
        assert calibration["MeasuredFrom"] == "axicon3"

    assert (run_dir / "auto_scan_setup.json").exists()
    assert len(pauses) == 2
    assert "BLOCK" in pauses[0]
    assert "UNBLOCK" in pauses[1]


def test_scan_frames_record_exposure_scan_kind_and_machine_y(modules, tmp_path):
    session, writer, _ = make_session(modules, tmp_path)
    limits = modules.coordinates.Bounds3D(**LIMITS_KW)

    session.run(limits)

    manifest = load_manifest(writer)
    scans = [r for r in manifest if r["Extra"]["ScanKind"] == "AutoBeamStack"]

    assert len(scans) == 4

    for record in scans:
        assert record["Extra"]["Exposure_us"] > 0
        assert record["Extra"]["MeasuredFrom"] == "axicon3"
        assert record["Extra"]["Subfolder"] in ("y0100.00cm", "y0101.00cm")
        assert record["Extra"]["MachineY_mm"] in (20.0, 30.0)
        # Machine == table except Y (anchored to the optic).
        assert record["GantryPosition_mm"]["x_mm"] == record["TablePosition_mm"]["x_mm"]
        assert record["GantryPosition_mm"]["z_mm"] == record["TablePosition_mm"]["z_mm"]
        assert record["TablePosition_mm"]["y_mm"] == pytest.approx(
            1000.0 + (record["GantryPosition_mm"]["y_mm"] - 20.0)
        )


def test_fixed_raster_writes_raster_metadata(modules, tmp_path):
    session, writer, _ = make_session(modules, tmp_path, background_mode="none")
    limits = modules.coordinates.Bounds3D(**LIMITS_KW)

    session.run(limits)

    meta = json.loads(
        (writer.run_dir / "y0100.00cm" / "raster_metadata.json").read_text()
    )
    assert meta["RasterMode"] == "fixed"
    assert meta["GridShape"] == [2, 1]
    assert meta["CellsCaptured"] == 2
    assert meta["MachineY_mm"] == 20.0
    assert meta["BeamY_mm"] == 1000.0
    assert meta["FinalRect_mm"] == {
        "XMin": 55.0, "XMax": 60.0, "ZMin": -80.0, "ZMax": -80.0,
    }


# ---------------------------------------------------------------------------
# Off-axis backgrounds
# ---------------------------------------------------------------------------


def test_offaxis_background_position_defaults_to_farthest_corner(modules, tmp_path):
    session, _, _ = make_session(modules, tmp_path, background_mode="offaxis")
    limits = modules.coordinates.Bounds3D(**LIMITS_KW)

    assert session.background_xz(limits) == EXPECTED_BG_CORNER


def test_offaxis_background_position_honors_explicit_config(modules, tmp_path):
    session, _, _ = make_session(
        modules,
        tmp_path,
        background_mode="offaxis",
        background_x_mm=10.0,
        background_z_mm=-120.0,
    )
    limits = modules.coordinates.Bounds3D(**LIMITS_KW)

    assert session.background_xz(limits) == (10.0, -120.0)


def test_axis_range_supports_descending(modules):
    down = modules.coordinates.AxisRange(
        start_mm=130.0, stop_mm=115.0, step_mm=5.0
    )
    assert down.values() == [130.0, 125.0, 120.0, 115.0]

    # The step's sign is ignored; direction comes from start/stop.
    down_neg = modules.coordinates.AxisRange(
        start_mm=130.0, stop_mm=115.0, step_mm=-5.0
    )
    assert down_neg.values() == down.values()

    up = modules.coordinates.AxisRange(start_mm=10.0, stop_mm=20.0, step_mm=5.0)
    assert up.values() == [10.0, 15.0, 20.0]


def test_descending_y_scan_walks_downward_with_correct_beam_y(
    modules, tmp_path
):
    # --y-start 30 --y-stop 20: bootstrap near the optic (largest machine
    # Y on this rig), then walk downward. Beam-y bookkeeping is anchored
    # at Y-START, so folders still name the true distance from the optic.
    import dataclasses

    session, writer, pauses = make_session(
        modules, tmp_path, background_mode="none"
    )
    config = dataclasses.replace(
        session.config, YStart_machine_mm=30.0, YStop_machine_mm=20.0
    )
    session = modules.auto_scan.AutoScanSession(
        writer, config, pause_fn=pauses.append
    )
    limits = modules.coordinates.Bounds3D(**LIMITS_KW)

    session.run(limits)

    manifest = load_manifest(writer)
    scans = [r for r in manifest if r["Extra"]["ScanKind"] == "AutoBeamStack"]
    machine_ys = [r["GantryPosition_mm"]["y_mm"] for r in scans]

    # First slice at Y30 (the start), then Y20 — descending.
    assert machine_ys[0] == 30.0
    assert machine_ys[-1] == 20.0
    assert set(machine_ys) == {30.0, 20.0}

    # MeasuredSensorY=1000 mm at YStart=30, sign +1: beam y 100 cm at
    # Y30 and 99 cm at Y20.
    assert (writer.run_dir / "y0100.00cm").is_dir()
    assert (writer.run_dir / "y0099.00cm").is_dir()


def test_offaxis_run_puts_matched_backgrounds_in_each_slice_folder(modules, tmp_path):
    session, writer, pauses = make_session(
        modules, tmp_path, background_mode="offaxis"
    )
    limits = modules.coordinates.Bounds3D(**LIMITS_KW)

    records = session.run(limits)

    # 2 slices x (1 background shot + 2 X-Z points); no ladder, no prompts.
    assert len(records) == 6
    assert pauses == []
    assert not (writer.run_dir / "background").exists()

    manifest = load_manifest(writer)
    backgrounds = [r for r in manifest if r["Extra"]["ScanKind"] == "Background"]
    scans = [r for r in manifest if r["Extra"]["ScanKind"] == "AutoBeamStack"]

    assert len(backgrounds) == 2
    assert len(scans) == 4

    for y_name, machine_y in (("y0100.00cm", 20.0), ("y0101.00cm", 30.0)):
        slice_dir = writer.run_dir / y_name
        npy_files = sorted(p.name for p in slice_dir.glob("*.npy"))
        assert len(npy_files) == 3
        assert sum("background" in name for name in npy_files) == 1

        slice_backgrounds = [
            r for r in backgrounds if r["Extra"]["Subfolder"] == y_name
        ]
        assert len(slice_backgrounds) == 1
        background = slice_backgrounds[0]

        # Exact exposure match with this slice's calibration, taken at the
        # off-axis corner of the X-Z plane at this slice's machine Y.
        calibration = json.loads(
            (slice_dir / "calibration_result.json").read_text()
        )
        assert background["Extra"]["Exposure_us"] == calibration["FinalExposure_us"]
        assert background["Extra"]["BackgroundMode"] == "OffAxisAmbient"
        assert background["GantryPosition_mm"] == {
            "x_mm": EXPECTED_BG_CORNER[0],
            "y_mm": machine_y,
            "z_mm": EXPECTED_BG_CORNER[1],
        }

    setup = json.loads((writer.run_dir / "auto_scan_setup.json").read_text())
    assert setup["BackgroundMode"] == "offaxis"
    assert setup["BackgroundXZ_mm"] == list(EXPECTED_BG_CORNER)
    assert "BackgroundExposures_us" not in setup


def test_background_mode_none_skips_all_backgrounds(modules, tmp_path):
    session, writer, pauses = make_session(modules, tmp_path, background_mode="none")
    limits = modules.coordinates.Bounds3D(**LIMITS_KW)

    records = session.run(limits)

    assert len(records) == 4  # scans only
    assert pauses == []
    assert all(
        r["Extra"]["ScanKind"] == "AutoBeamStack" for r in load_manifest(writer)
    )


def test_background_reused_when_exposure_stable(modules, tmp_path):
    # Both slices calibrate to the same exposure -> the second slice reuses
    # the first slice's background instead of driving to the corner again.
    session, writer, _ = make_session(
        modules,
        tmp_path,
        background_mode="offaxis",
        background_exposure_change=0.10,
    )
    limits = modules.coordinates.Bounds3D(**LIMITS_KW)

    records = session.run(limits)

    backgrounds = [
        r
        for r in load_manifest(writer)
        if r["Extra"]["ScanKind"] == "Background"
    ]

    assert len(backgrounds) == 1  # captured for the first slice only
    assert len(records) == 1 + 4

    ref_first = json.loads(
        (writer.run_dir / "y0100.00cm" / "background_reference.json").read_text()
    )
    assert ref_first["Reused"] is False
    assert ref_first["ExposureChangeFraction"] is None
    assert ref_first["BackgroundSlice"] == "y0100.00cm"

    ref_second = json.loads(
        (writer.run_dir / "y0101.00cm" / "background_reference.json").read_text()
    )
    assert ref_second["Reused"] is True
    assert ref_second["ExposureChangeFraction"] == 0.0
    assert ref_second["BackgroundSlice"] == "y0100.00cm"
    assert ref_second["BackgroundMachineY_mm"] == 20.0
    assert ref_second["BackgroundExposure_us"] == ref_second["SliceExposure_us"]
    assert ref_second["BackgroundPaths"] == [backgrounds[0]["Path"]]
    # The reused background's frames live in the FIRST slice's folder.
    assert "y0100.00cm" in backgrounds[0]["Path"]


def test_background_recaptured_when_exposure_changes(modules, tmp_path):
    def bright():
        return FakeImage(np.full((4, 4), 150, dtype=np.uint8))

    def dim():
        return FakeImage(np.full((4, 4), 5, dtype=np.uint8))

    # Slice 1 calibrates in one frame (exposure stays at the 1000 us seed);
    # slice 2's first calibration frame is dim -> exposure jumps 8x -> the
    # >=10% change forces a fresh background.
    images = [
        bright(), bright(), bright(), bright(),   # slice 1: calib, bg, 2 scans
        dim(), bright(), bright(), bright(), bright(),  # slice 2: calib x2, bg, 2 scans
    ]

    session, writer, _ = make_session(
        modules,
        tmp_path,
        images=images,
        background_mode="offaxis",
        background_exposure_change=0.10,
    )
    limits = modules.coordinates.Bounds3D(**LIMITS_KW)

    session.run(limits)

    backgrounds = [
        r
        for r in load_manifest(writer)
        if r["Extra"]["ScanKind"] == "Background"
    ]
    assert len(backgrounds) == 2  # one per slice

    ref_second = json.loads(
        (writer.run_dir / "y0101.00cm" / "background_reference.json").read_text()
    )
    assert ref_second["Reused"] is False
    assert ref_second["ExposureChangeFraction"] == pytest.approx(7.0)
    assert ref_second["BackgroundSlice"] == "y0101.00cm"
    assert ref_second["BackgroundExposure_us"] == pytest.approx(8000.0)


# ---------------------------------------------------------------------------
# Adaptive raster integration
# ---------------------------------------------------------------------------


def test_adaptive_raster_single_frame_beam_end_to_end(modules, tmp_path):
    # Per slice: 1 calibration frame, 1 background frame, then the seed
    # raster frame whose beam blob sits in the frame center (dark borders)
    # -> the slice completes after ONE raster position.
    def blob_frame():
        arr = np.full((12, 16), 2, dtype=np.uint8)
        arr[4:8, 6:10] = 150
        return FakeImage(arr)

    def calibration_frame():
        return FakeImage(np.full((12, 16), 150, dtype=np.uint8))

    def dark_frame():
        return FakeImage(np.full((12, 16), 2, dtype=np.uint8))

    images = []
    for _ in range(2):  # two slices
        images.extend([calibration_frame(), dark_frame(), blob_frame()])

    session, writer, _ = make_session(
        modules,
        tmp_path,
        images=images,
        background_mode="offaxis",
        raster_mode="adaptive",
        min_signal_pixels=4,
    )
    limits = modules.coordinates.Bounds3D(**LIMITS_KW)

    records = session.run(limits)

    # Per slice: 1 background + 1 raster frame.
    assert len(records) == 4

    for y_name in ("y0100.00cm", "y0101.00cm"):
        meta = json.loads(
            (writer.run_dir / y_name / "raster_metadata.json").read_text()
        )
        assert meta["RasterMode"] == "adaptive"
        assert meta["BeamFitsInSingleFrame"] is True
        assert meta["CellsCaptured"] == 1
        assert meta["GridShape"] == [1, 1]
        assert meta["TruncatedSides"] == []
        assert meta["LatticeAxes"] == {"x": "machine X", "y": "machine Z"}
        # Threshold came from this slice's own off-axis background:
        # p99(2) + margin(8) = 10.
        assert meta["SignalThreshold_counts"] == 10.0
        assert "offaxis-background" in meta["SignalThresholdSource"]
        assert meta["Cells"][0]["AnySignal"] is True

    scans = [
        r
        for r in load_manifest(writer)
        if r["Extra"]["ScanKind"] == "AutoBeamStack"
    ]
    assert len(scans) == 2
    for record in scans:
        assert record["Extra"]["GridI"] == 0
        assert record["Extra"]["GridJ"] == 0
        # Seed frame sits at the calibration point in the X-Z plane.
        assert record["GantryPosition_mm"]["x_mm"] == 57.5
        assert record["GantryPosition_mm"]["z_mm"] == -80.0
        assert record["GantryPosition_mm"]["y_mm"] in (20.0, 30.0)


def test_dark_perimeter_frames_are_relabeled_on_disk(modules, tmp_path):
    # Seed frame has signal to every border -> one growth ring of 8 dark
    # cells -> stop. The 8 proof-of-darkness frames must be renamed with
    # the -dark suffix, with manifests and raster metadata following.
    images = [FakeImage(np.full((12, 16), 150, dtype=np.uint8))]  # calibration
    images.append(FakeImage(np.full((12, 16), 150, dtype=np.uint8)))  # seed
    images.extend(
        FakeImage(np.full((12, 16), 2, dtype=np.uint8)) for _ in range(8)
    )

    writer = make_writer(modules, tmp_path, images=images)

    config = modules.auto_scan.AutoScanConfig(
        PlacementID="placement-01",
        MeasuredSensorY_mm=1000.0,
        YStart_machine_mm=20.0,
        YStop_machine_mm=20.0,
        X=modules.coordinates.AxisRange(start_mm=50.0, stop_mm=70.0, step_mm=5.0),
        Z=modules.coordinates.AxisRange(start_mm=-84.0, stop_mm=-76.0, step_mm=4.0),
        RasterMode="adaptive",
        MinSignalPixels=4,
        FindBeam=False,
        BackgroundMode="none",
    )

    session = modules.auto_scan.AutoScanSession(writer, config)
    limits = modules.coordinates.Bounds3D(**LIMITS_KW)

    records = session.run(limits)

    slice_dir = writer.run_dir / "y0100.00cm"
    dark_files = sorted(slice_dir.glob("*-dark.npy"))
    lit_files = [
        p for p in slice_dir.glob("*.npy") if not p.stem.endswith("-dark")
    ]

    assert len(dark_files) == 8
    assert len(lit_files) == 1  # the seed

    # Manifests reference the renamed paths (no stale entries).
    for manifest in (writer.run_dir / "frames.jsonl", slice_dir / "frames.jsonl"):
        paths = [
            json.loads(line)["Path"] for line in manifest.read_text().splitlines()
        ]
        assert all(Path(p).exists() for p in map(Path, paths)), manifest
        assert sum(p.endswith("-dark.npy") for p in paths) == 8

    # Raster metadata cell paths follow the rename.
    meta = json.loads((slice_dir / "raster_metadata.json").read_text())
    for cell in meta["Cells"]:
        for path in cell["Paths"]:
            if not cell["AnySignal"]:
                assert path.endswith("-dark.npy")
            assert Path(path).exists()

    # In-memory records were remapped too.
    assert sum(r.Path.endswith("-dark.npy") for r in records) == 8


def test_calibration_seeds_next_slice_from_previous_exposure(modules, tmp_path):
    session, writer, _ = make_session(modules, tmp_path)
    limits = modules.coordinates.Bounds3D(**LIMITS_KW)

    session.run(limits)

    exposure_node = writer.cam.ExposureTime

    # Ladder rungs first (100, 1000), then one calibration set per slice;
    # the second slice's seed equals the first slice's converged exposure.
    calibration_sets = exposure_node.set_calls[2:]
    assert len(calibration_sets) == 2
    assert calibration_sets[0] == calibration_sets[1]


def test_follow_beam_moves_calibration_to_brightest_cell(modules, tmp_path):
    # Slice 1's two raster points have different brightness; slice 2's
    # calibration move must target the brighter cell's X/Z, not the
    # configured center.
    def frame(value):
        return FakeImage(np.full((4, 4), value, dtype=np.uint8))

    images = [
        frame(150),  # slice 1 calibration (converges at seed exposure)
        frame(40),   # slice 1 scan point (55, -80): dim
        frame(150),  # slice 1 scan point (60, -80): bright
        frame(150),  # slice 2 calibration
        frame(150), frame(150),  # slice 2 scan points
    ]

    session, writer, _ = make_session(
        modules, tmp_path, images=images, background_mode="none"
    )
    limits = modules.coordinates.Bounds3D(**LIMITS_KW)

    session.run(limits)

    moved = writer.stage_controller.moved_to

    # Slice 1 calibrates at the configured center (57.5); slice 2 at the
    # brightest slice-1 cell (60, -80).
    slice1_calibration = moved[0]
    slice2_calibration = moved[3]

    assert (slice1_calibration.x_mm, slice1_calibration.z_mm) == (57.5, -80.0)
    assert slice1_calibration.y_mm == 20.0
    assert (slice2_calibration.x_mm, slice2_calibration.z_mm) == (60.0, -80.0)
    assert slice2_calibration.y_mm == 30.0

    # The followed point is recorded in the slice's calibration JSON.
    calibration_2 = json.loads(
        (writer.run_dir / "y0101.00cm" / "calibration_result.json").read_text()
    )
    assert calibration_2["CalibrationX_mm"] == 60.0
    assert calibration_2["CalibrationZ_mm"] == -80.0


def test_follow_beam_disabled_keeps_configured_point(modules, tmp_path):
    def frame(value):
        return FakeImage(np.full((4, 4), value, dtype=np.uint8))

    images = [
        frame(150), frame(40), frame(150),
        frame(150), frame(150), frame(150),
    ]

    import dataclasses

    session, writer, _ = make_session(
        modules, tmp_path, images=images, background_mode="none"
    )
    session.config = dataclasses.replace(session.config, FollowBeam=False)
    limits = modules.coordinates.Bounds3D(**LIMITS_KW)

    session.run(limits)

    moved = writer.stage_controller.moved_to
    assert (moved[3].x_mm, moved[3].z_mm) == (57.5, -80.0)


def make_find_beam_session(modules, tmp_path, images):
    """Session with a 3-stop Z sweep range (-90, -85, -80) and FindBeam on."""

    writer = make_writer(modules, tmp_path, images=images)

    config = modules.auto_scan.AutoScanConfig(
        PlacementID="placement-01",
        MeasuredSensorY_mm=1000.0,
        YStart_machine_mm=20.0,
        YStop_machine_mm=20.0,
        X=modules.coordinates.AxisRange(start_mm=55.0, stop_mm=60.0, step_mm=5.0),
        Z=modules.coordinates.AxisRange(start_mm=-90.0, stop_mm=-80.0, step_mm=5.0),
        FindBeam=True,
        FindBeamStepZ_mm=5.0,
        BackgroundMode="none",
    )

    return modules.auto_scan.AutoScanSession(writer, config), writer


def blob_image():
    arr = np.full((12, 16), 2, dtype=np.uint8)
    arr[4:8, 6:10] = 150  # max 150, median 2 -> contrast 148
    return FakeImage(arr)


def flat_image(value):
    return FakeImage(np.full((12, 16), value, dtype=np.uint8))


def test_find_beam_sweeps_far_end_first_and_seeds_at_contrast(modules, tmp_path):
    # Stops swept: -90 (flat) then -85 (blob) -> hit; -80 never visited.
    session, writer = make_find_beam_session(
        modules, tmp_path, images=[flat_image(2), blob_image()]
    )

    assert session.find_beam(20.0) is True
    assert session._calibration_xz == (57.5, -85.0)

    moved_z = [move.z_mm for move in writer.stage_controller.moved_to]
    assert moved_z == [-90.0, -85.0]  # far extremum first


def test_find_beam_rejects_flat_ambient_and_escalates_exposure(modules, tmp_path):
    # Bright but FLAT frames (ambient at long exposure) must never count as
    # the beam: 3 attempts x 3 stops, all rejected.
    session, writer = make_find_beam_session(
        modules, tmp_path, images=[flat_image(150) for _ in range(9)]
    )

    assert session.find_beam(20.0) is False
    assert session._calibration_xz == (57.5, -85.0)  # unchanged (caps center)

    # Exposure escalated x8 per attempt: 10ms, 80ms, 640ms.
    assert writer.cam.ExposureTime.set_calls == [10000.0, 80000.0, 640000.0]


def test_find_beam_engages_only_without_explicit_calibration_point(modules, tmp_path):
    session, _, _ = make_session(modules, tmp_path, find_beam=True)
    assert session._need_find_beam is True  # no explicit point given

    import dataclasses

    explicit_writer = make_writer(
        modules, tmp_path, images=[], job_type="auto_scan2"
    )
    explicit = modules.auto_scan.AutoScanSession(
        explicit_writer,
        dataclasses.replace(
            session.config, CalibrationX_mm=59.0, CalibrationZ_mm=-76.0
        ),
    )
    assert explicit._need_find_beam is False


def test_beam_direction_sign_flips_beam_y(modules, tmp_path):
    config = modules.auto_scan.AutoScanConfig(
        PlacementID="p",
        MeasuredSensorY_mm=1000.0,
        YStart_machine_mm=20.0,
        BeamDirectionSign=-1,
    )

    # Machine +Y moves TOWARD the optic: beam y decreases as Y increases.
    assert config.beam_y_mm(20.0) == 1000.0
    assert config.beam_y_mm(30.0) == 990.0


def test_beam_direction_default_matches_hardware_verification(modules, tmp_path):
    # Verified on the rig 2026-07-22: machine +Y points TOWARD the optic.
    config = modules.auto_scan.AutoScanConfig(
        PlacementID="p", MeasuredSensorY_mm=1000.0
    )

    assert config.BeamDirectionSign == -1


# ---------------------------------------------------------------------------
# Camera reconnect: exposure restoration
# ---------------------------------------------------------------------------


def test_session_wires_restore_state_hook(modules, tmp_path):
    session, writer, _ = make_session(modules, tmp_path)

    assert writer.RestoreState == session._restore_camera_state


def test_restore_reapplies_last_deliberate_exposure(modules, tmp_path):
    session, writer, _ = make_session(modules, tmp_path)

    # No exposure deliberately set yet: restore must be a no-op.
    writer.cam.ExposureTime.SetValue(25.0)
    session._restore_camera_state()
    assert writer.cam.ExposureTime.GetValue() == 25.0

    # After a deliberate set (as calibration / find-beam / ladder do)...
    session._set_exposure(1234.0)
    assert writer.cam.ExposureTime.GetValue() == 1234.0

    # ...a reconnect reverts the camera to base settings...
    writer.cam.ExposureTime.SetValue(1000.0)

    # ...and the hook restores the scan's actual exposure.
    session._restore_camera_state()
    assert writer.cam.ExposureTime.GetValue() == 1234.0
