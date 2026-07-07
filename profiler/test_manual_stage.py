import importlib
import json
import sys
import types

import numpy as np
import pytest

# Reuse the fake PySpin / camera scaffolding from the dataset writer tests.
from test_dataset_writer import (
    FakeCamera,
    FakeCameraSettings,
    FakeImage,
    make_fake_pyspin,
)


@pytest.fixture()
def modules(monkeypatch):
    monkeypatch.setitem(sys.modules, "PySpin", make_fake_pyspin())

    fake_camera_settings_module = types.ModuleType("camera_settings")
    fake_camera_settings_module.FLIRCameraSettings = object
    monkeypatch.setitem(sys.modules, "camera_settings", fake_camera_settings_module)

    for name in ("dataset_writer", "manual_stage"):
        sys.modules.pop(name, None)

    dataset_writer = importlib.import_module("dataset_writer")
    manual_stage = importlib.import_module("manual_stage")

    return dataset_writer, manual_stage


def make_writer(dataset_writer, tmp_path, images):
    writer = dataset_writer.FLIRDatasetWriter(
        camera_index=0,
        cam=FakeCamera(images=images),
        camera_settings=FakeCameraSettings(),
        config=dataset_writer.DatasetWriterConfig(
            DatasetRoot=tmp_path,
            RunUUID="manual-test-run",
        ),
    )
    writer.prepare_run()
    return writer


def make_session(manual_stage, writer, **overrides):
    defaults = dict(
        SensorZ_mm=65.0,
        SensorZReference="front face of axicon #1",
        Metadata={"note": "manual unit test"},
    )
    defaults.update(overrides)

    return manual_stage.ManualStageSession(
        writer, manual_stage.ManualStageConfig(**defaults)
    )


class KeyEvent:
    def __init__(self, key):
        self.key = key


def test_save_frame_array_writes_npy_and_manifest(modules, tmp_path):
    dataset_writer, manual_stage = modules

    writer = make_writer(dataset_writer, tmp_path, images=[])
    session = make_session(manual_stage, writer)

    arr = np.arange(12, dtype=np.uint16).reshape(3, 4)
    session._last_frame = arr

    record = session.save_current_frame()

    assert record is not None
    assert session.move_index == 1
    assert session.saved_records == [record]

    # z-position of the sensor is captured in table coordinates + metadata.
    assert record.TablePosition_mm.z_mm == 65.0
    assert record.Extra["ScanKind"] == "ManualStage"
    assert record.Extra["MoveIndex"] == 0
    assert record.Extra["SensorZ_mm"] == 65.0
    assert record.Extra["SensorZReference"] == "front face of axicon #1"
    assert record.Extra["note"] == "manual unit test"

    np.testing.assert_array_equal(np.load(record.Path), arr)

    manifest_lines = (writer.run_dir / "frames.jsonl").read_text().splitlines()
    assert len(manifest_lines) == 1

    manifest_record = json.loads(manifest_lines[0])
    assert manifest_record["TablePosition_mm"]["z_mm"] == 65.0
    assert manifest_record["Extra"]["MoveIndex"] == 0


def test_each_save_gets_unique_filename_and_move_index(modules, tmp_path):
    dataset_writer, manual_stage = modules

    writer = make_writer(dataset_writer, tmp_path, images=[])
    session = make_session(manual_stage, writer)

    session._last_frame = np.zeros((2, 2), dtype=np.uint16)
    first = session.save_current_frame()

    session._last_frame = np.ones((2, 2), dtype=np.uint16)
    second = session.save_current_frame()

    assert first.Path != second.Path
    assert first.Extra["MoveIndex"] == 0
    assert second.Extra["MoveIndex"] == 1
    assert second.ShotIndex == 1


def test_save_without_frame_is_a_noop(modules, tmp_path):
    dataset_writer, manual_stage = modules

    writer = make_writer(dataset_writer, tmp_path, images=[])
    session = make_session(manual_stage, writer)

    assert session.save_current_frame() is None
    assert session.move_index == 0
    assert not (writer.run_dir / "frames.jsonl").exists()


def test_grab_frame_triggers_and_releases(modules, tmp_path):
    dataset_writer, manual_stage = modules

    arr = np.array([[5, 6], [7, 8]], dtype=np.uint16)
    image = FakeImage(arr)

    writer = make_writer(dataset_writer, tmp_path, images=[image])
    session = make_session(manual_stage, writer)

    frame = session.grab_frame()

    np.testing.assert_array_equal(frame, arr)
    assert image.released is True
    assert writer.cam.node_map.GetNode("TriggerSoftware").execute_count == 1

    # The returned frame must be a detached copy, not the camera buffer.
    frame[0, 0] = 999
    assert image.array[0, 0] == 5


def test_grab_frame_skips_incomplete_images(modules, tmp_path):
    dataset_writer, manual_stage = modules

    image = FakeImage(np.zeros((2, 2), dtype=np.uint16), incomplete=True)

    writer = make_writer(dataset_writer, tmp_path, images=[image])
    session = make_session(manual_stage, writer)

    assert session.grab_frame() is None
    assert image.released is True


def test_key_handling_saves_and_quits(modules, tmp_path):
    dataset_writer, manual_stage = modules

    writer = make_writer(dataset_writer, tmp_path, images=[])
    session = make_session(manual_stage, writer)
    session._last_frame = np.zeros((2, 2), dtype=np.uint16)

    session._on_key(KeyEvent(" "))
    session._on_key(KeyEvent("s"))
    assert session.move_index == 2

    assert session._done is False
    session._on_key(KeyEvent("q"))
    assert session._done is True

    # Unrelated keys do nothing.
    session._done = False
    session._on_key(KeyEvent("x"))
    assert session._done is False
    assert session.move_index == 2


def test_run_loop_headless_save_and_quit(modules, tmp_path, monkeypatch):
    """
    Drive the full run() loop with the Agg backend: save on the first
    iteration, quit on the second, and confirm acquisition begin/end pairing.
    """

    import matplotlib

    matplotlib.use("Agg", force=True)

    dataset_writer, manual_stage = modules

    frames = [
        FakeImage(np.full((4, 4), fill, dtype=np.uint16))
        for fill in (10, 20, 30, 40)
    ]

    writer = make_writer(dataset_writer, tmp_path, images=frames)
    session = make_session(manual_stage, writer, PreviewInterval_s=0.01)

    # Agg windows never receive real key events; inject them via plt.pause.
    import matplotlib.pyplot as plt

    pause_count = {"n": 0}

    def fake_pause(_interval):
        pause_count["n"] += 1

        if pause_count["n"] == 1:
            session._on_key(KeyEvent(" "))
        else:
            session._on_key(KeyEvent("q"))

    monkeypatch.setattr(plt, "pause", fake_pause)

    records = session.run()

    assert len(records) == 1
    assert records[0].Extra["MoveIndex"] == 0

    events = writer.cam.events
    assert events[0] == "begin"
    assert events[-1] == "end"