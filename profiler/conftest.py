"""
Shared fakes and fixtures for the profiler test suite.

The fake PySpin module and camera scaffolding here are a superset of what
every test file needs, so the fakes cannot drift apart per-file again.
"""

import json
import sys
import types

import numpy as np
import pytest


# Production modules that bind PySpin (directly or via imports) at import
# time. They must be re-imported per test so each test's fake PySpin is the
# one they see; a cached module keeps whichever fake was installed first.
PYSPIN_DEPENDENT_MODULES = (
    "align_axicon",
    "align_preview",
    "auto_scan",
    "calibration",
    "camera_base",
    "camera_settings",
    "dataset_writer",
    "headless_calibration",
    "manual_stage",
)


@pytest.fixture(autouse=True)
def _isolate_profiler_modules():
    for name in PYSPIN_DEPENDENT_MODULES:
        sys.modules.pop(name, None)
    yield
    for name in PYSPIN_DEPENDENT_MODULES:
        sys.modules.pop(name, None)


@pytest.fixture()
def fake_pyspin(monkeypatch):
    fake = make_fake_pyspin()
    monkeypatch.setitem(sys.modules, "PySpin", fake)
    return fake


class FakeNode:
    def __init__(
        self,
        value=None,
        *,
        minimum=float("-inf"),
        maximum=float("inf"),
        available=True,
        readable=True,
        writable=True,
    ):
        self.value = value
        self.minimum = minimum
        self.maximum = maximum
        self.available = available
        self.readable = readable
        self.writable = writable
        self.set_calls = []

    def SetValue(self, value):
        self.value = value
        self.set_calls.append(value)

    def GetValue(self):
        return self.value

    def GetMin(self):
        return self.minimum

    def GetMax(self):
        return self.maximum


class FakeEnumEntry:
    def __init__(self, name, value=1, readable=True):
        self.name = name
        self.value = value
        self.readable = readable
        self.available = True

    def GetValue(self):
        return self.value

    def GetSymbolic(self):
        return self.name


class FakeEnumNode:
    def __init__(self, readable=True, writable=True):
        self.readable = readable
        self.writable = writable
        self.available = True
        self.current_value = None
        self.requested_entries = []
        self.set_values = []

    # Older assertions use the entries_requested spelling.
    @property
    def entries_requested(self):
        return self.requested_entries

    def GetEntryByName(self, name):
        self.requested_entries.append(name)
        return FakeEnumEntry(name, value=len(self.requested_entries))

    def SetValue(self, value):
        self.set_values.append(value)
        self.current_value = value

    def SetIntValue(self, value):
        self.set_values.append(value)
        self.current_value = value

    def GetCurrentEntry(self):
        return FakeEnumEntry("Current")


class FakeCommandNode:
    available = True
    readable = True
    writable = True

    def __init__(self):
        self.execute_count = 0

    def Execute(self):
        self.execute_count += 1


class FakeNodeMap:
    def __init__(self, nodes):
        self.nodes = nodes

    def GetNode(self, name):
        return self.nodes.get(name)


class FakeImage:
    def __init__(self, array, incomplete=False, status=0, frame_id=1):
        self.array = np.array(array, copy=True)
        self.incomplete = incomplete
        self.status = status
        self.frame_id = frame_id
        self.released = False

    # Older assertions use the arr spelling.
    @property
    def arr(self):
        return self.array

    def IsIncomplete(self):
        return self.incomplete

    def GetImageStatus(self):
        return self.status

    def GetNDArray(self):
        return self.array

    def GetFrameID(self):
        return self.frame_id

    def Save(self, path):
        self.saved_paths = getattr(self, "saved_paths", [])
        self.saved_paths.append(path)

    def Release(self):
        self.released = True


class FakeCamera:
    def __init__(self, images=(), model_name="BFS-PGE-31S4M"):
        self.images = list(images)
        self.events = []

        # QuickSpin-style nodes used by real FLIRCameraSettings.apply(...)
        self.PixelFormat = FakeNode()
        self.ExposureAuto = FakeNode()
        self.ExposureMode = FakeNode()
        self.ExposureTime = FakeNode(minimum=25.0, maximum=1_000_000.0)

        self.AcquisitionFrameRatePersistence = FakeNode(True)
        self.AcquisitionMode = FakeEnumNode()
        self.AcquisitionFrameRateEnable = FakeNode(True)
        self.AcquisitionFrameRate = FakeNode(1.0, minimum=0.0, maximum=1_000_000.0)

        self.GainAuto = FakeNode()
        self.Gain = FakeNode(minimum=0.0, maximum=48.0)

        self.GammaEnable = FakeNode()
        self.Gamma = FakeNode(minimum=0.25, maximum=4.0)

        self.BlackLevelClampingEnable = FakeNode(False)

        self.BlackLevelSelector = FakeNode()
        self.BlackLevel = FakeNode(minimum=0.0, maximum=100.0)

        self.BalanceWhiteAuto = FakeNode()
        self.BalanceRatioSelector = FakeNode()
        self.BalanceRatio = FakeNode(minimum=0.0, maximum=10.0)
        self.GevSCPSPacketSize = FakeNode(1500, minimum=0, maximum=10_000_000)
        self.DeviceLinkThroughputLimit = FakeNode(
            10_000_000,
            minimum=0,
            maximum=100_000_000,
        )
        self.TriggerMode = FakeEnumNode()
        self.TriggerSource = FakeEnumNode()

        self.stream_buffer_mode = FakeEnumNode()
        self.stream_buffer_count_mode = FakeEnumNode()
        self.stream_buffer_count_manual = FakeNode(10, minimum=1, maximum=10_000)

        self.node_map = FakeNodeMap(
            {
                "AcquisitionMode": self.AcquisitionMode,
                "TriggerMode": self.TriggerMode,
                "TriggerSource": self.TriggerSource,
                "TriggerSoftware": FakeCommandNode(),
            }
        )
        self.stream_node_map = FakeNodeMap(
            {
                "StreamBufferHandlingMode": self.stream_buffer_mode,
                "StreamBufferCountMode": self.stream_buffer_count_mode,
                "StreamBufferCountManual": self.stream_buffer_count_manual,
            }
        )
        self.tl_device_node_map = FakeNodeMap(
            {
                "DeviceModelName": FakeNode(model_name),
            }
        )

    def GetNodeMap(self):
        return self.node_map

    def GetTLStreamNodeMap(self):
        return self.stream_node_map

    def GetTLDeviceNodeMap(self):
        return self.tl_device_node_map

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


def make_fake_pyspin():
    fake = types.SimpleNamespace()

    class SpinnakerException(Exception):
        pass

    class FakeCameraList:
        def __init__(self, cameras=None):
            self.cameras = list(cameras or [])

        def GetSize(self):
            return len(self.cameras)

        def GetByIndex(self, index):
            return self.cameras[index]

        def Clear(self):
            self.cameras.clear()

    class FakeSystem:
        def __init__(self, cameras=None):
            self.cameras = FakeCameraList(cameras)

        def GetCameras(self):
            return self.cameras

        def ReleaseInstance(self):
            pass

    fake.FakeCameraList = FakeCameraList
    fake.FakeSystem = FakeSystem
    fake.SpinnakerException = SpinnakerException
    fake.Camera = object
    fake.System = types.SimpleNamespace(GetInstance=lambda: FakeSystem())
    fake.CameraList = object

    fake.IsAvailable = lambda node: node is not None and getattr(node, "available", True)
    fake.IsReadable = lambda node: node is not None and getattr(node, "readable", True)
    fake.IsWritable = lambda node: node is not None and getattr(node, "writable", True)

    fake.CEnumerationPtr = lambda node: node
    fake.CStringPtr = lambda node: node
    fake.CFloatPtr = lambda node: node
    fake.CBooleanPtr = lambda node: node
    fake.CIntegerPtr = lambda node: node
    fake.CCommandPtr = lambda node: node

    # QuickSpin enum constants used by FLIRCameraSettings.apply(...)
    fake.PixelFormat_Mono8 = "PixelFormat_Mono8"
    fake.PixelFormat_Mono12Packed = "PixelFormat_Mono12Packed"
    fake.PixelFormat_Mono16 = "PixelFormat_Mono16"

    fake.ExposureAuto_Off = "ExposureAuto_Off"
    fake.ExposureAuto_Once = "ExposureAuto_Once"
    fake.ExposureAuto_Continuous = "ExposureAuto_Continuous"

    fake.ExposureMode_Timed = "ExposureMode_Timed"
    fake.ExposureMode_TriggerWidth = "ExposureMode_TriggerWidth"

    fake.GainAuto_Off = "GainAuto_Off"
    fake.GainAuto_Once = "GainAuto_Once"
    fake.GainAuto_Continuous = "GainAuto_Continuous"

    fake.BlackLevelSelector_All = "BlackLevelSelector_All"

    fake.BalanceWhiteAuto_Off = "BalanceWhiteAuto_Off"
    fake.BalanceWhiteAuto_Once = "BalanceWhiteAuto_Once"
    fake.BalanceWhiteAuto_Continuous = "BalanceWhiteAuto_Continuous"

    fake.BalanceRatioSelector_Blue = "BalanceRatioSelector_Blue"
    fake.BalanceRatioSelector_Red = "BalanceRatioSelector_Red"

    fake.TriggerMode_Off = "TriggerMode_Off"
    fake.TriggerMode_On = "TriggerMode_On"
    fake.TriggerSource_Software = "TriggerSource_Software"

    return fake
