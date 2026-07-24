import importlib
import json
import sys
import types

import numpy as np
import pytest

from conftest import FakeCamera, FakeCameraSettings, FakeImage


MODULE_NAME = "dataset_writer"


class FastStageController:
    def __init__(self):
        self.moved_to = []
        self.waited_for = []

    def move_to_scan_point(self, point, signals):
        self.moved_to.append(point.GantryPosition_mm)
        signals.MovementStarted.set()

    def wait_until_motion_complete(self, point, timeout_s, signals):
        self.waited_for.append((point.GantryPosition_mm, timeout_s))
        signals.MovementComplete.set()


class NeverCompletesStageController:
    def move_to_scan_point(self, point, signals):
        signals.MovementStarted.set()

    def wait_until_motion_complete(self, point, timeout_s, signals):
        pass


@pytest.fixture()
def dataset_writer_module(monkeypatch, fake_pyspin):
    fake_camera_settings_module = types.ModuleType("camera_settings")
    fake_camera_settings_module.FLIRCameraSettings = object
    monkeypatch.setitem(sys.modules, "camera_settings", fake_camera_settings_module)

    return importlib.import_module(MODULE_NAME)


@pytest.fixture()
def coordinates_module():
    return importlib.import_module("coordinates")


def make_point(coordinates_module, *, nshots=1):
    return coordinates_module.ScanPoint(
        PlacementID="p 000/bad",
        GantryPosition_mm=coordinates_module.Vec3D(x_mm=1.0, y_mm=2.0, z_mm=3.0),
        TablePosition_mm=coordinates_module.Vec3D(x_mm=101.0, y_mm=202.0, z_mm=303.0),
        NShots=nshots,
        Metadata={"note": "unit test"},
    )


def test_prepare_run_writes_settings_and_metadata(dataset_writer_module, tmp_path):
    cam = FakeCamera(images=[])
    settings = FakeCameraSettings()

    config = dataset_writer_module.DatasetWriterConfig(
        DatasetRoot=tmp_path,
        JobType="unit_test",
    )

    writer = dataset_writer_module.FLIRDatasetWriter(
        camera_index=0,
        cam=cam,
        camera_settings=settings,
        config=config,
        stage_controller=FastStageController(),
    )

    run_dir = writer.prepare_run()

    assert run_dir.exists()
    assert settings.apply_calls == [(cam, True)]
    assert (run_dir / "camera_settings.json").exists()

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["RunDir"] == str(run_dir)
    assert metadata["Config"]["DatasetRoot"] == str(tmp_path)
    assert "CoordinateConvention" in metadata
    assert "GantryPosition_mm" in metadata["CoordinateConvention"]
    assert "TablePosition_mm" in metadata["CoordinateConvention"]


def test_acquire_scan_writes_npy_manifest_and_coordinate_record(
    dataset_writer_module,
    coordinates_module,
    tmp_path,
):
    arr = np.array([[0, 1], [2, 3]], dtype=np.uint16)
    image = FakeImage(arr, frame_id=123)
    cam = FakeCamera(images=[image])
    settings = FakeCameraSettings()
    stage = FastStageController()

    writer = dataset_writer_module.FLIRDatasetWriter(
        camera_index=0,
        cam=cam,
        camera_settings=settings,
        config=dataset_writer_module.DatasetWriterConfig(
            DatasetRoot=tmp_path,
            JobType="unit_test",
            AcquisitionTimeout_ms=1234,
        ),
        stage_controller=stage,
    )
    writer.prepare_run()

    point = make_point(coordinates_module, nshots=1)
    records = writer.acquire_scan([point])

    assert len(records) == 1
    record = records[0]

    assert record.PlacementID == point.PlacementID
    assert record.GantryPosition_mm == point.GantryPosition_mm
    assert record.TablePosition_mm == point.TablePosition_mm
    assert record.Shape == arr.shape
    assert record.DType == "uint16"
    assert record.Min == 0
    assert record.Max == 3
    assert record.SaturatedPixelCount == 0
    assert record.Extra["note"] == "unit test"
    assert record.Extra["FrameID"] == 123

    saved_path = dataset_writer_module.Path(record.Path)
    assert saved_path.exists()
    np.testing.assert_array_equal(np.load(saved_path), arr)

    manifest_lines = (writer.run_dir / "frames.jsonl").read_text().splitlines()
    assert len(manifest_lines) == 1

    manifest_record = json.loads(manifest_lines[0])
    assert manifest_record["PlacementID"] == point.PlacementID
    assert manifest_record["GantryPosition_mm"] == {
        "x_mm": 1.0,
        "y_mm": 2.0,
        "z_mm": 3.0,
    }
    assert manifest_record["TablePosition_mm"] == {
        "x_mm": 101.0,
        "y_mm": 202.0,
        "z_mm": 303.0,
    }
    assert manifest_record["Extra"]["note"] == "unit test"
    assert manifest_record["Extra"]["FrameID"] == 123

    assert stage.moved_to == [point.GantryPosition_mm]
    assert stage.waited_for == [(point.GantryPosition_mm, 30.0)]
    assert cam.events == ["begin", "get:1234", "end"]
    assert cam.node_map.GetNode("TriggerSoftware").execute_count == 1
    assert image.released is True

    # Node configuration is delegated to FLIRCameraSettings.apply during prepare_run.
    assert settings.apply_calls == [(cam, True)]


def test_acquire_static_writes_multiple_frames_without_stage_motion(
    dataset_writer_module,
    tmp_path,
):
    arr0 = np.array([[0, 1], [2, 3]], dtype=np.uint16)
    arr1 = np.array([[4, 5], [6, 7]], dtype=np.uint16)

    image0 = FakeImage(arr0, frame_id=100)
    image1 = FakeImage(arr1, frame_id=101)
    cam = FakeCamera(images=[image0, image1])
    settings = FakeCameraSettings()
    stage = FastStageController()

    writer = dataset_writer_module.FLIRDatasetWriter(
        camera_index=0,
        cam=cam,
        camera_settings=settings,
        config=dataset_writer_module.DatasetWriterConfig(
            DatasetRoot=tmp_path,
            JobType="static",
            AcquisitionTimeout_ms=1234,
        ),
        stage_controller=stage,
    )
    writer.prepare_run()

    records = writer.acquire_static(
        nshots=2,
        metadata={"note": "camera fixed on table; no gantry yet"},
    )

    assert len(records) == 2
    assert stage.moved_to == []
    assert stage.waited_for == []
    assert cam.events == ["begin", "get:1234", "get:1234", "end"]
    assert cam.node_map.GetNode("TriggerSoftware").execute_count == 2

    assert records[0].PlacementID == "static-camera"
    assert records[0].GantryPosition_mm.x_mm == 0.0
    assert records[0].GantryPosition_mm.y_mm == 0.0
    assert records[0].GantryPosition_mm.z_mm == 0.0
    assert records[0].Extra["ScanKind"] == "Static"
    assert records[0].Extra["note"] == "camera fixed on table; no gantry yet"
    assert records[0].Extra["FrameID"] == 100

    assert records[1].ShotIndex == 1
    assert records[1].Extra["FrameID"] == 101

    np.testing.assert_array_equal(np.load(records[0].Path), arr0)
    np.testing.assert_array_equal(np.load(records[1].Path), arr1)

    manifest_lines = (writer.run_dir / "frames.jsonl").read_text().splitlines()
    assert len(manifest_lines) == 2

    first_manifest_record = json.loads(manifest_lines[0])
    assert first_manifest_record["PlacementID"] == "static-camera"
    assert first_manifest_record["GantryPosition_mm"] == {
        "x_mm": 0.0,
        "y_mm": 0.0,
        "z_mm": 0.0,
    }
    assert first_manifest_record["TablePosition_mm"] == {
        "x_mm": 0.0,
        "y_mm": 0.0,
        "z_mm": 0.0,
    }
    assert first_manifest_record["Extra"]["ScanKind"] == "Static"
    assert first_manifest_record["Extra"]["FrameID"] == 100

    # Node configuration is delegated to FLIRCameraSettings.apply during prepare_run.
    assert settings.apply_calls == [(cam, True)]


def test_acquire_one_frame_refuses_before_movement_complete(
    dataset_writer_module,
    coordinates_module,
    tmp_path,
):
    cam = FakeCamera(images=[FakeImage(np.zeros((2, 2), dtype=np.uint16))])

    writer = dataset_writer_module.FLIRDatasetWriter(
        camera_index=0,
        cam=cam,
        camera_settings=FakeCameraSettings(),
        config=dataset_writer_module.DatasetWriterConfig(JobType="unit_test", DatasetRoot=tmp_path),
        stage_controller=FastStageController(),
    )

    with pytest.raises(dataset_writer_module.DatasetWriterError, match="MovementComplete"):
        writer._acquire_one_frame(make_point(coordinates_module), shot_idx=0)

    assert cam.events == []
    assert cam.node_map.GetNode("TriggerSoftware").execute_count == 0


def test_dropped_trigger_timeout_is_retried(
    dataset_writer_module,
    coordinates_module,
    fake_pyspin,
    tmp_path,
):
    """A -1011 GetNextImage timeout re-triggers instead of failing the scan."""

    arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    image = FakeImage(arr, frame_id=7)

    class DroppedTriggerCamera(FakeCamera):
        def __init__(self):
            super().__init__(images=[image])
            self.failures_remaining = 1

        def GetNextImage(self, timeout_ms):
            if self.failures_remaining:
                self.failures_remaining -= 1
                raise fake_pyspin.SpinnakerException(
                    "Spinnaker: Failed waiting for EventData on "
                    "NEW_BUFFER_DATA event. (GenTL error code: -1011) [-1011]"
                )
            return super().GetNextImage(timeout_ms)

    cam = DroppedTriggerCamera()

    writer = dataset_writer_module.FLIRDatasetWriter(
        camera_index=0,
        cam=cam,
        camera_settings=FakeCameraSettings(),
        config=dataset_writer_module.DatasetWriterConfig(
            JobType="unit_test", DatasetRoot=tmp_path
        ),
        stage_controller=FastStageController(),
    )
    writer.prepare_run()

    point = make_point(coordinates_module, nshots=1)
    records = writer.acquire_scan([point])

    assert len(records) == 1
    assert records[0].Extra["FrameID"] == 7
    # One dropped trigger + one successful re-trigger.
    assert cam.node_map.GetNode("TriggerSoftware").execute_count == 2


def test_persistent_trigger_timeout_still_raises(
    dataset_writer_module,
    coordinates_module,
    fake_pyspin,
    tmp_path,
):
    class DeadCamera(FakeCamera):
        def GetNextImage(self, timeout_ms):
            raise fake_pyspin.SpinnakerException("(GenTL error code: -1011) [-1011]")

    writer = dataset_writer_module.FLIRDatasetWriter(
        camera_index=0,
        cam=DeadCamera(),
        camera_settings=FakeCameraSettings(),
        config=dataset_writer_module.DatasetWriterConfig(
            JobType="unit_test", DatasetRoot=tmp_path
        ),
        stage_controller=FastStageController(),
    )
    writer.prepare_run()

    with pytest.raises(dataset_writer_module.DatasetWriterError, match="-1011"):
        writer.acquire_scan([make_point(coordinates_module, nshots=1)])


def test_incomplete_image_releases_and_ends_acquisition(
    dataset_writer_module,
    coordinates_module,
    tmp_path,
):
    image = FakeImage(
        np.zeros((2, 2), dtype=np.uint16),
        incomplete=True,
        status=3,
    )
    cam = FakeCamera(images=[image])

    writer = dataset_writer_module.FLIRDatasetWriter(
        camera_index=0,
        cam=cam,
        camera_settings=FakeCameraSettings(),
        config=dataset_writer_module.DatasetWriterConfig(JobType="unit_test", DatasetRoot=tmp_path),
        stage_controller=FastStageController(),
    )
    writer.signals.MovementComplete.set()

    with pytest.raises(
        dataset_writer_module.DatasetWriterError,
        match="Image incomplete; image status = 3",
    ):
        writer._acquire_point_frames(make_point(coordinates_module))

    assert image.released is True
    assert cam.events == ["begin", "get:2000", "end"]
    assert cam.node_map.GetNode("TriggerSoftware").execute_count == 1

# ---------------------------------------------------------------------------
# Camera bus removal (-1024): reconnect and retry the shot
# ---------------------------------------------------------------------------


def make_removal_camera(fake_pyspin, images, failures=1):
    """A camera that reports a bus removal for the first `failures` grabs."""

    class RemovedCamera(FakeCamera):
        def __init__(self):
            super().__init__(images=images)
            self.failures_remaining = failures

        def GetNextImage(self, timeout_ms):
            if self.failures_remaining:
                self.failures_remaining -= 1
                raise fake_pyspin.SpinnakerException(
                    "Spinnaker: Camera has been removed from the list and "
                    "is no longer valid. [-1024]"
                )
            return super().GetNextImage(timeout_ms)

    return RemovedCamera()


def test_camera_removal_reconnects_and_retries_shot(
    dataset_writer_module,
    coordinates_module,
    fake_pyspin,
    monkeypatch,
    tmp_path,
):
    """One -1024 mid-shot: reconnect, restore state, retry — scan survives."""

    arr = np.array([[9, 9], [9, 9]], dtype=np.uint8)
    cam = make_removal_camera(fake_pyspin, [FakeImage(arr, frame_id=42)])

    writer = dataset_writer_module.FLIRDatasetWriter(
        camera_index=0,
        cam=cam,
        camera_settings=FakeCameraSettings(),
        config=dataset_writer_module.DatasetWriterConfig(
            JobType="unit_test", DatasetRoot=tmp_path, TriggerArmDelay_s=0.0
        ),
        stage_controller=FastStageController(),
    )
    writer.prepare_run()

    reopens = []
    monkeypatch.setattr(writer, "reopen", lambda: reopens.append(True))

    restores = []
    writer.RestoreState = lambda: restores.append(True)

    records = writer.acquire_scan([make_point(coordinates_module, nshots=1)])

    assert len(records) == 1
    assert records[0].Extra["FrameID"] == 42
    assert reopens == [True]
    assert restores == [True]  # RestoreState ran after the reopen
    # begin -> removal (no get event logged for the failed grab) -> end ->
    # (reopen) -> begin -> retry get -> end
    assert cam.events == ["begin", "end", "begin", "get:2000", "end"]


def test_persistent_camera_removal_still_raises(
    dataset_writer_module,
    coordinates_module,
    fake_pyspin,
    monkeypatch,
    tmp_path,
):
    """The removal retry happens exactly once; a dead bus still fails."""

    cam = make_removal_camera(fake_pyspin, images=[], failures=99)

    writer = dataset_writer_module.FLIRDatasetWriter(
        camera_index=0,
        cam=cam,
        camera_settings=FakeCameraSettings(),
        config=dataset_writer_module.DatasetWriterConfig(
            JobType="unit_test", DatasetRoot=tmp_path, TriggerArmDelay_s=0.0
        ),
        stage_controller=FastStageController(),
    )
    writer.prepare_run()

    monkeypatch.setattr(writer, "reopen", lambda: None)

    with pytest.raises(dataset_writer_module.DatasetWriterError, match="-1024"):
        writer.acquire_scan([make_point(coordinates_module, nshots=1)])


def test_restore_state_failure_is_logged_not_raised(
    dataset_writer_module,
    monkeypatch,
    caplog,
    tmp_path,
):
    import logging

    writer = dataset_writer_module.FLIRDatasetWriter(
        camera_index=0,
        cam=FakeCamera(images=[]),
        camera_settings=FakeCameraSettings(),
        config=dataset_writer_module.DatasetWriterConfig(
            JobType="unit_test", DatasetRoot=tmp_path
        ),
        stage_controller=FastStageController(),
    )

    monkeypatch.setattr(writer, "reopen", lambda: None)

    def bad_restore():
        raise RuntimeError("exposure node gone")

    writer.RestoreState = bad_restore

    with caplog.at_level(logging.WARNING, logger="dataset_writer"):
        writer.reconnect()  # must not raise

    assert any(
        "RestoreState" in r.message and "exposure node gone" in r.message
        for r in caplog.records
    )


def test_reopen_unavailable_for_injected_test_camera(
    dataset_writer_module, tmp_path
):
    """Guards against a fake-camera test silently 'reconnecting'."""

    writer = dataset_writer_module.FLIRDatasetWriter(
        camera_index=0,
        cam=FakeCamera(images=[]),
        camera_settings=FakeCameraSettings(),
        config=dataset_writer_module.DatasetWriterConfig(
            JobType="unit_test", DatasetRoot=tmp_path
        ),
        stage_controller=FastStageController(),
    )

    with pytest.raises(Exception, match="injected test camera"):
        writer.reopen()
