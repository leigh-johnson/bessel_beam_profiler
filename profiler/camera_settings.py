from __future__ import annotations # convert type hints to strings at runtime

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal, Optional
import json

import PySpin


AutoMode = Literal["Off", "Once", "Continuous"]
ExposureModeName = Literal["Timed", "TriggerWidth"]
PixelFormatName = Literal[
    "Mono8",
    "Mono10p",
    "Mono12p",
    "Mono12Packed",
    "Mono16",
]


AUTO_MODES = {"Off", "Once", "Continuous"}
EXPOSURE_MODES = {"Timed", "TriggerWidth"}
PIXEL_FORMATS = {"Mono8", "Mono10p", "Mono12p", "Mono12Packed", "Mono16"}
STREAM_BUFFER_HANDLING_MODES = {"NewestFirst", "OldestFirst", "OldestFirstOverwrite", "NewestOnly"}
ACQUISITION_MODES = {"SingleFrame", "MultiFrame", "Continuous"}

class CameraSettingError(RuntimeError):
    pass


@dataclass(frozen=True)
class FLIRCameraSettings:
    """
    These are camera software settings saved to JSON files for reproducible beam profiling experiments.
    Intended for use with the Spinnaker Python QuickSpin API.

    Example usage:

        1. Initialize settings and write to JSON file:
            ```
            settings = FLIRCameraSettings(
                    CameraModel="BFS-PGE-31S4M",
                    PixelFormat="Mono16",
                    ExposureAuto="Off",
                    ExposureMode="Timed",
                    ExposureTime=500.0,   # microseconds
                    GainAuto="Off",
                    Gain=0.0,
                    GammaEnable=False,
                )

            settings.to_json_file("beam_scan_camera_settings.json")
            ```
        2. Apply settings to a connected camera:
            ```
            system = PySpin.System.GetInstance()
            cam_list = system.GetCameras()

            cam = cam_list.GetByIndex(0)
            cam.Init()

            try:
                settings = FLIRCameraSettings.from_json_file("beam_scan_camera_settings.json")
                warnings = settings.apply(cam, strict=True)

                # Now safe to begin acquisition.
                cam.BeginAcquisition()
                # acquire images...
                cam.EndAcquisition()

            finally:
                cam.DeInit()
                del cam
                cam_list.Clear()
                system.ReleaseInstance()
            ```
    """
    # This is the Spinnaker GenAPI AcquisitionMode enum name.
    # For software-triggered acquisition, FLIR's buffer-handling example uses
    # Continuous mode, then captures exactly one frame per TriggerSoftware.Execute().
    AcquisitionMode: str = "Continuous"
    AcquisitionFrameRateEnable: bool = True
    AcquisitionFrameRate: float = 3.0 # frames per second
    AcquisitionFrameRatePersistence: bool = True

    # Black level / DC offset
    BlackLevelClampingEnable: Optional[bool] = False
    BlackLevelSelector: Optional[str] = "All"
    BlackLevel: Optional[float] = None

    # White balance / color channel ratios for RGB-Bayer sensors.
    # I don't think we need these for monochrome cameras, but included here for completeness since we have a few color cameras in the lab. 
    # Leave these as None for BFS-PGE-31S4M.
    BalanceWhiteAuto: Optional[AutoMode] = None
    BalanceRatioBlue: Optional[float] = None
    BalanceRatioRed: Optional[float] = None

    CameraModel: str = "" # e.g. BFS-PGE-31S4M

    # Mono8: quick alignment. Mono16 for final beam profiling / fit fn.
    PixelFormat: Optional[PixelFormatName] = "Mono8"

    # Exposure
    ExposureAuto: Optional[AutoMode] = "Off"
    ExposureMode: Optional[ExposureModeName] = "Timed"
    ExposureTime: Optional[float] = 1000  # microseconds

    # Packet size for GEV cameras. 
    # 1500 bytes is a good default for Wifi networks.
    # For wired networks, you can try 9000 bytes (jumbo frames) if your network supports it.
    GevSCPSPacketSize: Optional[int] = 1500
    
    # Gain
    GainAuto: Optional[AutoMode] = "Off"
    Gain: Optional[float] = 0.0  # dB

    # Gamma
    GammaEnable: Optional[bool] = False
    Gamma: Optional[float] = 1.0

    StreamBufferCountManual: Optional[int] = 10
    StreamBufferHandlingMode: Optional[str] = "NewestOnly"
    StreamBufferCountMode: Optional[str] = "Manual"
    DeviceLinkThroughputLimit: Optional[int] = 10_000_000 # bits

    TriggerSource: str = "Software"


    def __post_init__(self) -> None:
        # validate settings
        _check_choice("ExposureAuto", self.ExposureAuto, AUTO_MODES)
        _check_choice("GainAuto", self.GainAuto, AUTO_MODES)
        _check_choice("BalanceWhiteAuto", self.BalanceWhiteAuto, AUTO_MODES)
        _check_choice("ExposureMode", self.ExposureMode, EXPOSURE_MODES)
        _check_choice("PixelFormat", self.PixelFormat, PIXEL_FORMATS)
        _check_choice("StreamBufferHandlingMode", self.StreamBufferHandlingMode, STREAM_BUFFER_HANDLING_MODES)
        _check_choice("AcquisitionMode", self.AcquisitionMode, ACQUISITION_MODES)


        if self.ExposureTime is not None and self.ExposureTime <= 0:
            raise ValueError("ExposureTime must be positive, in microseconds.")

        if self.Gain is not None and self.Gain < 0:
            raise ValueError("Gain should be nonnegative, in dB.")

        if self.Gamma is not None and self.Gamma <= 0:
            raise ValueError("Gamma must be positive.")
        
        if self.AcquisitionFrameRate is not None and self.AcquisitionFrameRate <= 0:
            raise ValueError("AcquisitionFrameRate must be positive, in frames per second.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json_file(self, path: str | Path, *, indent: int = 2) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=indent) + "\n")

    @classmethod
    def from_json_file(cls, path: str | Path) -> "FLIRCameraSettings":
        path = Path(path)
        return cls(**json.loads(path.read_text()))

    @classmethod
    def from_camera(cls, cam: PySpin.Camera) -> "FLIRCameraSettings":
        """
        Read current settings from a connected, initialized camera.

        Assumes:
            cam.Init() has already been called.
        """

        return cls(
            AcquisitionMode=_get_enum(cam, "AcquisitionMode", default="Continuous"),
            AcquisitionFrameRateEnable=_get_bool(cam, "AcquisitionFrameRateEnable", default=True),
            AcquisitionFrameRate=_get_float(cam, "AcquisitionFrameRate", default=1.0),
            AcquisitionFrameRatePersistence=_get_bool(cam, "AcquisitionFrameRatePersistence", default=True),
            CameraModel=_read_tl_string(cam, "DeviceModelName", default=""),
            PixelFormat=_get_enum(cam, "PixelFormat", default=None),
            ExposureAuto=_get_enum(cam, "ExposureAuto", default=None),
            ExposureMode=_get_enum(cam, "ExposureMode", default=None),
            ExposureTime=_get_float(cam, "ExposureTime", default=None),
            GainAuto=_get_enum(cam, "GainAuto", default=None),
            Gain=_get_float(cam, "Gain", default=None),
            GammaEnable=_get_bool(cam, "GammaEnable", default=None),
            Gamma=_get_float(cam, "Gamma", default=None),
            GevSCPSPacketSize=_get_float(cam, "GevSCPSPacketSize", default=1500),
            BlackLevel=_get_float(cam, "BlackLevel", default=None),
            BlackLevelClampingEnable=_get_bool(cam, "BlackLevelClampingEnable", default=False),
            BlackLevelSelector=_get_enum(cam, "BlackLevelSelector", default=None),
            BalanceWhiteAuto=_get_enum(cam, "BalanceWhiteAuto", default=None),
            BalanceRatioBlue=_get_selected_float(
                cam,
                selector_feature="BalanceRatioSelector",
                selector_entry="Blue",
                value_feature="BalanceRatio",
                default=None,
            ),
            BalanceRatioRed=_get_selected_float(
                cam,
                selector_feature="BalanceRatioSelector",
                selector_entry="Red",
                value_feature="BalanceRatio",
                default=None,
            ),
            StreamBufferCountManual=_get_integer(cam, "StreamBufferCountManual", default=10),
            StreamBufferHandlingMode=_get_enum(cam, "StreamBufferHandlingMode", default="NewestFirst"),
            StreamBufferCountMode=_get_enum(cam, "StreamBufferCountMode", default="Manual"),
            DeviceLinkThroughputLimit=_get_integer(cam, "DeviceLinkThroughputLimit", default=10_000_000),
        )

    def apply(
        self,
        cam: PySpin.Camera,
        *,
        strict: bool = True,
        validate_model: bool = True,
    ) -> list[str]:
        """
        Apply these settings to an initialized camera.

        Assumes:
            cam.Init() has already been called.
            cam.BeginAcquisition() has NOT been called yet.

        Returns:
            A list of warning messages when strict=False.

        In strict=True mode, unavailable or unwritable nodes raise errors.
        """

        messages: list[str] = []

        if validate_model and self.CameraModel:
            actual_model = _read_tl_string(cam, "DeviceModelName", default="")
            if actual_model and not _model_matches(self.CameraModel, actual_model):
                _skip_or_raise(
                    strict,
                    messages,
                    f"Camera model mismatch: config has {self.CameraModel!r}, "
                    f"connected camera reports {actual_model!r}.",
                )

        if self.PixelFormat is not None:
            _set_enum(cam, "PixelFormat", self.PixelFormat, strict, messages)

        # Exposure: manual beam profiling should almost always use Off + Timed.
        if self.ExposureAuto is not None:
            _set_enum(cam, "ExposureAuto", self.ExposureAuto, strict, messages)

        if self.ExposureAuto == "Off":
            if self.ExposureMode is not None:
                _set_enum(cam, "ExposureMode", self.ExposureMode, strict, messages)

            if self.ExposureTime is None:
                _skip_or_raise(
                    strict,
                    messages,
                    "ExposureAuto is Off, but ExposureTime is None.",
                )
            else:
                _set_float(cam, "ExposureTime", self.ExposureTime, strict, messages)
        else:
            if self.ExposureTime is not None:
                messages.append(
                    "ExposureTime was not applied because ExposureAuto is not Off."
                )

        # Acquisition mode & frame rate
        if self.AcquisitionFrameRatePersistence is not None:
            _set_bool(
                cam,
                "AcquisitionFrameRatePersistence",
                self.AcquisitionFrameRatePersistence,
                strict,
                messages,
            )
        if self.AcquisitionMode is not None:
            _set_enum(
                cam,
                "AcquisitionMode",
                self.AcquisitionMode,
                strict,
                messages,
            )
        if self.AcquisitionFrameRateEnable is not None:
            _set_bool(
                cam,
                "AcquisitionFrameRateEnable",
                self.AcquisitionFrameRateEnable,
                strict,
                messages,
            )
        if self.AcquisitionFrameRate is not None:
            _set_float(
                cam,
                "AcquisitionFrameRate",
                self.AcquisitionFrameRate,
                strict,
                messages,
            )

        # Gain
        if self.GainAuto is not None:
            _set_enum(cam, "GainAuto", self.GainAuto, strict, messages)

        if self.GainAuto == "Off" and self.Gain is not None:
            _set_float(cam, "Gain", self.Gain, strict, messages)

        # Gamma
        if self.GammaEnable is not None:
            _set_bool(cam, "GammaEnable", self.GammaEnable, strict, messages)

        if self.GammaEnable and self.Gamma is not None:
            _set_float(cam, "Gamma", self.Gamma, strict, messages)

        # Black level
        if self.BlackLevelClampingEnable is not None:
            _set_bool(
                cam,
                "BlackLevelClampingEnable",
                self.BlackLevelClampingEnable,
                strict,
                messages,
            )
        if self.BlackLevel is not None:
            if self.BlackLevelSelector is not None:
                _set_enum(
                    cam,
                    "BlackLevelSelector",
                    self.BlackLevelSelector,
                    strict,
                    messages,
                )
            _set_float(cam, "BlackLevel", self.BlackLevel, strict, messages)

        # White balance. Usually skip these entirely on monochrome cameras.
        if self.BalanceWhiteAuto is not None:
            _set_enum(cam, "BalanceWhiteAuto", self.BalanceWhiteAuto, strict, messages)

        if self.BalanceRatioBlue is not None:
            if self.BalanceWhiteAuto is None:
                _set_enum(cam, "BalanceWhiteAuto", "Off", strict, messages)
            _set_enum(cam, "BalanceRatioSelector", "Blue", strict, messages)
            _set_float(cam, "BalanceRatio", self.BalanceRatioBlue, strict, messages)

        if self.BalanceRatioRed is not None:
            if self.BalanceWhiteAuto is None:
                _set_enum(cam, "BalanceWhiteAuto", "Off", strict, messages)
            _set_enum(cam, "BalanceRatioSelector", "Red", strict, messages)
            _set_float(cam, "BalanceRatio", self.BalanceRatioRed, strict, messages)

        # Stream buffer settings
        # These are set using s_node_map = cam.GetTLStreamNodeMap()
        # So we can't use the QuickSpin cam object for these, and must use the GenAPI node map instead.

        # Retrieve Stream Parameters device nodemap
        s_node_map = cam.GetTLStreamNodeMap()

        handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferHandlingMode'))
        if not _read_writeable(handling_mode):
            _skip_or_raise(strict, messages, "StreamBufferHandlingMode is not available/writable.")
        else:
            handling_mode_entry = handling_mode.GetEntryByName(self.StreamBufferHandlingMode)
            handling_mode.SetIntValue(handling_mode_entry.GetValue())

        # Set stream buffer Count Mode to manual
        stream_buffer_count_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferCountMode'))
        if not _read_writeable(stream_buffer_count_mode):
            _skip_or_raise(strict, messages, "StreamBufferCountMode is not available/writable.")
        else:
            stream_buffer_count_mode_entry = stream_buffer_count_mode.GetEntryByName(self.StreamBufferCountMode)
            stream_buffer_count_mode.SetIntValue(stream_buffer_count_mode_entry.GetValue())

        # must be set after StreamBufferCountMode is set to Manual
        if self.StreamBufferCountMode == "Manual" and self.StreamBufferCountManual is not None:
            buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode('StreamBufferCountManual'))
            if not _read_writeable(buffer_count):
                _skip_or_raise(strict, messages, "StreamBufferCountManual is not available/writable.")
            else:
                buffer_count.SetValue(self.StreamBufferCountManual)

        if self.DeviceLinkThroughputLimit is not None:
            _set_integer(
                cam,
                "DeviceLinkThroughputLimit",
                self.DeviceLinkThroughputLimit,
                strict,
                messages,
            )
        if self.GevSCPSPacketSize is not None:
            _set_integer(
                cam,
                "GevSCPSPacketSize",
                self.GevSCPSPacketSize,
                strict,
                messages,
            )

        # set software trigger source
        if self.TriggerSource is not None:
            # Trigger must be Off before changing TriggerSource.
            _set_enum(cam, "TriggerMode", "Off", strict, messages)
            _set_enum(cam, "TriggerSource", self.TriggerSource, strict, messages)
            _set_enum(cam, "TriggerMode", "On", strict, messages)


        return messages


def _check_choice(name: str, value: Optional[str], allowed: set[str]) -> None:
    if value is not None and value not in allowed:
        allowed_str = ", ".join(sorted(allowed))
        raise ValueError(f"{name}={value!r} is invalid. Allowed: {allowed_str}")


def _model_matches(expected: str, actual: str) -> bool:
    """
    Flexible because some cameras report a longer name such as
    'Blackfly S BFS-PGE-31S4M' rather than only 'BFS-PGE-31S4M'.
    """
    return expected == actual or expected in actual or actual in expected


def _quickspin_node(cam: PySpin.Camera, feature: str) -> Any:
    return getattr(cam, feature, None)


def _is_available(node: Any) -> bool:
    return node is not None and PySpin.IsAvailable(node)


def _is_readable(node: Any) -> bool:
    return _is_available(node) and PySpin.IsReadable(node)


def _is_writable(node: Any) -> bool:
    return _is_available(node) and PySpin.IsWritable(node)


def _skip_or_raise(strict: bool, messages: list[str], message: str) -> None:
    if strict:
        raise CameraSettingError(message)
    messages.append(message)


def _set_enum(
    cam: PySpin.Camera,
    feature: str,
    entry_name: str,
    strict: bool,
    messages: list[str],
) -> None:
    """
    Set an enum node such as ExposureAuto='Off' using QuickSpin first,
    then fall back to GenAPI if needed.
    """

    # QuickSpin path, e.g. cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    node = _quickspin_node(cam, feature)
    enum_const_name = f"{feature}_{entry_name}"
    enum_value = getattr(PySpin, enum_const_name, None)

    if enum_value is not None and _is_writable(node):
        node.SetValue(enum_value)
        return

    # GenAPI fallback
    try:
        node_map = cam.GetNodeMap()
        enum_node = PySpin.CEnumerationPtr(node_map.GetNode(feature))

        if not PySpin.IsAvailable(enum_node) or not PySpin.IsWritable(enum_node):
            _skip_or_raise(strict, messages, f"{feature} is not available/writable.")
            return

        entry = enum_node.GetEntryByName(entry_name)
        if not PySpin.IsAvailable(entry) or not PySpin.IsReadable(entry):
            _skip_or_raise(
                strict,
                messages,
                f"{feature} has no readable enum entry {entry_name!r}.",
            )
            return

        enum_node.SetIntValue(entry.GetValue())

    except PySpin.SpinnakerException as ex:
        _skip_or_raise(strict, messages, f"Failed to set {feature}={entry_name}: {ex}")


def _set_float(
    cam: PySpin.Camera,
    feature: str,
    value: float,
    strict: bool,
    messages: list[str],
) -> None:
    node = _quickspin_node(cam, feature)

    if not _is_writable(node):
        _skip_or_raise(strict, messages, f"{feature} is not available/writable.")
        return

    try:
        if hasattr(node, "GetMin") and hasattr(node, "GetMax"):
            lo = float(node.GetMin())
            hi = float(node.GetMax())
            if not (lo <= float(value) <= hi):
                raise ValueError(
                    f"{feature}={value} is outside camera range [{lo}, {hi}]."
                )

        node.SetValue(float(value))

    except (PySpin.SpinnakerException, ValueError) as ex:
        _skip_or_raise(strict, messages, f"Failed to set {feature}={value}: {ex}")


def _set_bool(
    cam: PySpin.Camera,
    feature: str,
    value: bool,
    strict: bool,
    messages: list[str],
) -> None:
    node = _quickspin_node(cam, feature)

    if not _is_writable(node):
        _skip_or_raise(strict, messages, f"{feature} is not available/writable.")
        return

    try:
        node.SetValue(bool(value))
    except PySpin.SpinnakerException as ex:
        _skip_or_raise(strict, messages, f"Failed to set {feature}={value}: {ex}")

def _set_integer(
    cam: PySpin.Camera,
    feature: str,
    value: int,
    strict: bool,
    messages: list[str],
) -> None:
    node = _quickspin_node(cam, feature)

    if not _is_writable(node):
        _skip_or_raise(strict, messages, f"{feature} is not available/writable.")
        return

    try:
        if hasattr(node, "GetMin") and hasattr(node, "GetMax"):
            lo = int(node.GetMin())
            hi = int(node.GetMax())
            if not (lo <= int(value) <= hi):
                raise ValueError(
                    f"{feature}={value} is outside camera range [{lo}, {hi}]."
                )

        node.SetValue(int(value))

    except (PySpin.SpinnakerException, ValueError) as ex:
        _skip_or_raise(strict, messages, f"Failed to set {feature}={value}: {ex}")

def _get_enum(
    cam: PySpin.Camera,
    feature: str,
    *,
    default: Optional[str],
) -> Optional[str]:
    try:
        node_map = cam.GetNodeMap()
        enum_node = PySpin.CEnumerationPtr(node_map.GetNode(feature))

        if not PySpin.IsAvailable(enum_node) or not PySpin.IsReadable(enum_node):
            return default

        entry = enum_node.GetCurrentEntry()
        if not PySpin.IsAvailable(entry) or not PySpin.IsReadable(entry):
            return default

        return str(entry.GetSymbolic())

    except PySpin.SpinnakerException:
        return default


def _get_float(
    cam: PySpin.Camera,
    feature: str,
    *,
    default: Optional[float],
) -> Optional[float]:
    node = _quickspin_node(cam, feature)

    if not _is_readable(node):
        return default

    try:
        return float(node.GetValue())
    except PySpin.SpinnakerException:
        return default


def _get_bool(
    cam: PySpin.Camera,
    feature: str,
    *,
    default: Optional[bool],
) -> Optional[bool]:
    node = _quickspin_node(cam, feature)

    if not _is_readable(node):
        return default

    try:
        return bool(node.GetValue())
    except PySpin.SpinnakerException:
        return default

def _get_integer(
    cam: PySpin.Camera,
    feature: str,
    *,
    default: Optional[int],
) -> Optional[int]:
    node = _quickspin_node(cam, feature)

    if not _is_readable(node):
        return default

    try:
        return int(node.GetValue())
    except PySpin.SpinnakerException:
        return default

def _read_tl_string(
    cam: PySpin.Camera,
    node_name: str,
    *,
    default: str,
) -> str:
    try:
        node_map = cam.GetTLDeviceNodeMap()
        node = PySpin.CStringPtr(node_map.GetNode(node_name))

        if not PySpin.IsAvailable(node) or not PySpin.IsReadable(node):
            return default

        return str(node.GetValue())

    except PySpin.SpinnakerException:
        return default


def _get_selected_float(
    cam: PySpin.Camera,
    *,
    selector_feature: str,
    selector_entry: str,
    value_feature: str,
    default: Optional[float],
) -> Optional[float]:
    messages: list[str] = []

    _set_enum(
        cam,
        selector_feature,
        selector_entry,
        strict=False,
        messages=messages,
    )

    return _get_float(cam, value_feature, default=default)

def _read_writeable(node):
    return PySpin.IsReadable(node) and PySpin.IsWritable(node)
