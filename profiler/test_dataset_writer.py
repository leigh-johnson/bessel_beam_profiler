import importlib
import json
import sys
import types

import numpy as np
import pytest


MODULE_NAME = "dataset_writer"


class FakeEnumEntry:
    def __init__(self, name, value=1, *, readable=True):
        self.name = name
        self.value = value
        self.readable = readable

    def GetValue(self):
        return self.value

    def GetSymbolic(self):
        return self.name


class FakeEnumNode:
    def __init__(self, *, readable=True, writable=True):
        self.readable = readable
        self.writable = writable
        self.available = True
        self.current_value = None
        self.requested_entries = []

    def GetEntryByName(self, name):
        self.requested_entries.append(name)
        return FakeEnumEntry(name=name, value=hash(name) % 1000)

    def SetIntValue(self, value):
        self.current_value = value


class FakeNodeMap:
    def __init__(self):
        self.acquisition_mode = FakeEnumNode()

    def GetNode(self, name):
        if name == "AcquisitionMode":
            return self.acquisition_mode
        return None


class FakeImage:
    def __init__(self, array, *, incomplete=False, status=0):
        self.array = np.array(array, copy=True)
        self.incomplete = incomplete
        self.status = status
        self.released = False

    def IsIncomplete(self):
        return self.incomplete

    def GetImageStatus(self):
        return self.status

    def GetNDArray(self):
        return self.array

    def Release(self):
        self.released = True


class FakeCamera:
    def __init__(self, images):
        self.images = list(images)
        self.node_map = FakeNodeMap()
        self.events = []

    def GetNodeMap(self):
        return self.node_map

    def BeginAcquisition(self):
        self.events.append("begin")

    def GetNextImage(self, timeout_ms):
        self.events.append(f"get:{timeout_ms}")
        if not self.images:
            raise RuntimeError("No fake images left")
        return self.images.pop(0)

    def EndAcquisition(self):
        self.events.append("end")


class FakeCameraSettings:
    def __init__(self):
        self.apply_calls = []
        self.saved_paths = []

    def apply(self, cam, strict=True):
        self.apply_calls.append((cam, strict))

    def to_json_file(self, path):
        self.saved_paths.append(path)
        path.write_text(json.dumps({"CameraModel": "fake"}, indent=2) + "\n")


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


def make_fake_pyspin():
    fake = types.SimpleNamespace()

    class SpinnakerException(Exception):
        pass

    fake.SpinnakerException = SpinnakerException
    fake.Camera = object
    fake.CEnumerationPtr = lambda node: node
    fake.IsReadable = lambda node: node is not None and getattr(node, "readable", True)
    fake.IsWritable = lambda node: node is not None and getattr(node, "writable", True)
    return fake


@pytest.fixture()
def dataset_writer_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "PySpin", make_fake_pyspin())

    fake_camera_settings_module = types.ModuleType("camera_settings")
    fake_camera_settings_module.FLIRCameraSettings = object
    monkeypatch.setitem(sys.modules, "camera_settings", fake_camera_settings_module)

    sys.modules.pop(MODULE_NAME, None)
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
        RunUUID="unit-test-run",
    )

    writer = dataset_writer_module.FLIRDatasetWriter(
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
    image = FakeImage(arr)
    cam = FakeCamera(images=[image])
    settings = FakeCameraSettings()
    stage = FastStageController()

    writer = dataset_writer_module.FLIRDatasetWriter(
        cam=cam,
        camera_settings=settings,
        config=dataset_writer_module.DatasetWriterConfig(
            DatasetRoot=tmp_path,
            RunUUID="unit-test-run",
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

    assert stage.moved_to == [point.GantryPosition_mm]
    assert cam.events == ["begin", "get:1234", "end"]
    assert image.released is True


def test_acquire_one_frame_refuses_before_movement_complete(
    dataset_writer_module,
    coordinates_module,
    tmp_path,
):
    cam = FakeCamera(images=[FakeImage(np.zeros((2, 2), dtype=np.uint16))])

    writer = dataset_writer_module.FLIRDatasetWriter(
        cam=cam,
        camera_settings=FakeCameraSettings(),
        config=dataset_writer_module.DatasetWriterConfig(DatasetRoot=tmp_path),
        stage_controller=FastStageController(),
    )

    with pytest.raises(dataset_writer_module.DatasetWriterError, match="MovementComplete"):
        writer._acquire_one_frame(make_point(coordinates_module), shot_idx=0)

    assert cam.events == []


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
        cam=cam,
        camera_settings=FakeCameraSettings(),
        config=dataset_writer_module.DatasetWriterConfig(DatasetRoot=tmp_path),
        stage_controller=FastStageController(),
    )
    writer.signals.MovementComplete.set()

    with pytest.raises(dataset_writer_module.DatasetWriterError, match="Image incomplete; image status = 3"):
        writer._acquire_one_frame(make_point(coordinates_module), shot_idx=0)

    assert image.released is True
    assert cam.events == ["begin", "get:2000", "end"]