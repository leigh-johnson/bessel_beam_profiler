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
        X=modules.coordinates.AxisRange(start_mm=55.0, stop_mm=60.0, step_mm=5.0),
        Z=modules.coordinates.AxisRange(start_mm=-80.0, stop_mm=-80.0, step_mm=5.0),
        NShots=1,
        RasterMode=raster_mode,
        MinSignalPixels=min_signal_pixels,
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
