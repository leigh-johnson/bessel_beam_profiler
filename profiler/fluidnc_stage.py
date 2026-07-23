"""
FluidNC (GRBL-protocol) stage control for the Bessel-beam imaging gantry.

The Jackpot3/FluidNC controller speaks the plain-text GRBL protocol over a
raw TCP socket (Telnet server, port 23 — the same board the WebUI talks to
at fluidnc-sr2.local). The conversation looks like:

    -> $H                      (home; blocks until done)
    <- ok
    -> G53 G1 X60 Y80 Z-50 F400
    <- ok                      (motion is QUEUED, not finished!)
    -> ?                       (realtime status query, no newline needed)
    <- <Run|MPos:42.1,80.0,-50.0|FS:400,0>
    -> ?
    <- <Idle|MPos:60.000,80.000,-50.000|FS:0,0>

Because "ok" only acknowledges that a command was queued, motion completion
is detected by polling "?" until the state is Idle and MPos matches the
commanded target.

Machine-coordinate conventions for this machine (see project notes):

    * All moves are sent as G53 G1 (absolute MACHINE coordinates), the same
      convention as rangetest.gcode, so work-offset state on the controller
      can never surprise us.
    * ScanPoint.GantryPosition_mm IS the machine coordinate (MPos).
    * Z homes UP to +3 mm; usable machine Z is about -127..3 with the beam
      propagating in +Z, so larger Z = farther from the axicon.
    * soft_limits is enabled on all axes; we ALSO validate every target
      against MachineLimits_mm before sending, because a soft-limit
      violation raises ALARM:2 and requires a reset + re-home.

This module deliberately imports no camera code (and no PySpin) so it can
be unit tested with a fake transport and reused standalone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import logging
import re
import socket
import time

from coordinates import Bounds3D, ScanPoint, Vec3D

logger = logging.getLogger(__name__)


class FluidNCError(RuntimeError):
    pass


class FluidNCCommandError(FluidNCError):
    """The controller replied error:N to a command."""


class FluidNCAlarmError(FluidNCError):
    """The controller is in (or entered) an Alarm state."""


class FluidNCTimeoutError(FluidNCError):
    pass


# GRBL error code cheat sheet for the ones we are most likely to hit.
GRBL_ERROR_HINTS = {
    2: "Bad number format in the G-code line.",
    9: "G-code locked out during alarm or jog state ($X to unlock).",
    15: "Jog target exceeds machine travel.",
    20: "Unsupported or invalid G-code command.",
}

GRBL_ALARM_HINTS = {
    1: "Hard limit triggered. Machine position is lost — re-home with $H.",
    2: "Soft limit: motion target exceeds machine travel. Reset, then $H.",
    8: "Homing fail: cycle did not complete.",
    9: "Homing fail: could not find limit switch.",
}


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class TelnetTransport:
    """
    Line-oriented wrapper around FluidNC's raw-TCP Telnet server (port 23).

    FluidNC does not do Telnet option negotiation; it is a plain socket that
    happens to live on port 23, so no telnetlib is needed.
    """

    def __init__(self, host: str, port: int = 23, connect_timeout_s: float = 5.0):
        self.host = host
        self.port = port
        self.connect_timeout_s = connect_timeout_s

        self._sock: Optional[socket.socket] = None
        self._rx_buffer = b""

    def connect(self) -> None:
        sock = socket.create_connection(
            (self.host, self.port), timeout=self.connect_timeout_s
        )
        # Motion commands are tiny; latency matters more than throughput.
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock
        self._rx_buffer = b""

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    @property
    def is_connected(self) -> bool:
        return self._sock is not None

    def write(self, data: bytes) -> None:
        if self._sock is None:
            raise FluidNCError("Transport is not connected.")
        self._sock.sendall(data)

    def read_line(self, timeout_s: float) -> Optional[str]:
        """
        Return the next complete line (without EOL), or None on timeout.
        """

        if self._sock is None:
            raise FluidNCError("Transport is not connected.")

        deadline = time.monotonic() + timeout_s

        while True:
            line = self._pop_buffered_line()
            if line is not None:
                return line

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None

            self._sock.settimeout(min(remaining, 0.5))
            try:
                chunk = self._sock.recv(4096)
            except socket.timeout:
                continue

            if not chunk:
                raise FluidNCError(
                    f"Connection to {self.host}:{self.port} closed by controller."
                )

            self._rx_buffer += chunk

    def drain(self) -> list[str]:
        """
        Read and return whatever lines are immediately available
        (startup banners, [MSG:...] pushes) without blocking.
        """

        lines = []
        while True:
            line = self.read_line(timeout_s=0.05)
            if line is None:
                return lines
            lines.append(line)

    def _pop_buffered_line(self) -> Optional[str]:
        if b"\n" not in self._rx_buffer:
            return None

        raw, self._rx_buffer = self._rx_buffer.split(b"\n", 1)
        return raw.decode("utf-8", errors="replace").strip("\r")


# ---------------------------------------------------------------------------
# Status parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FluidNCStatus:
    """Parsed form of a <State|MPos:...|...> realtime status report."""

    State: str  # Idle, Run, Home, Alarm, Hold:0, Jog, ...
    MPos: Optional[Vec3D] = None

    # Triggered input pins from the |Pn:...| field, e.g. "X" or "XYZ".
    # With NC limit switches, a pin reads triggered when the switch is
    # PRESSED **or** when its signal wire is loose/disconnected — a pin
    # stuck triggered away from the switch means a wiring problem, and
    # homing that axis will retract away and fail.
    Pins: str = ""

    Raw: str = ""

    @property
    def is_idle(self) -> bool:
        return self.State == "Idle"

    @property
    def is_alarm(self) -> bool:
        return self.State.startswith("Alarm")


STATUS_RE = re.compile(r"^<(?P<state>[^|>]+)(?P<fields>(\|[^>]*)?)>$")
MPOS_RE = re.compile(
    r"\|MPos:(?P<x>-?[\d.]+),(?P<y>-?[\d.]+),(?P<z>-?[\d.]+)"
)
PINS_RE = re.compile(r"\|Pn:(?P<pins>[A-Za-z]+)")


def parse_status_report(line: str) -> Optional[FluidNCStatus]:
    match = STATUS_RE.match(line.strip())

    if match is None:
        return None

    fields = match.group("fields") or ""

    mpos = None
    mpos_match = MPOS_RE.search(fields)

    if mpos_match is not None:
        mpos = Vec3D(
            x_mm=float(mpos_match.group("x")),
            y_mm=float(mpos_match.group("y")),
            z_mm=float(mpos_match.group("z")),
        )

    pins_match = PINS_RE.search(fields)
    pins = pins_match.group("pins") if pins_match else ""

    return FluidNCStatus(
        State=match.group("state"), MPos=mpos, Pins=pins, Raw=line.strip()
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@dataclass
class FluidNCClientConfig:
    Host: str = "fluidnc-sr2.local"
    Port: int = 23

    # Default feed for G1 moves, mm/min. Machine max_rate is 500.
    Feed_mm_min: float = 400.0

    # "ok" wait for ordinary (non-homing) commands.
    CommandTimeout_s: float = 10.0

    # $H can take a while: Z retract + X/Y auto-square seek at 400 mm/min.
    HomingTimeout_s: float = 180.0

    # Motion-complete polling.
    StatusPollInterval_s: float = 0.1
    PositionTolerance_mm: float = 0.01


class FluidNCClient:
    """
    Small GRBL-protocol client: send G-code lines, wait for ok, poll status.
    """

    def __init__(
        self,
        config: FluidNCClientConfig = FluidNCClientConfig(),
        transport=None,
    ):
        self.config = config
        self.transport = transport or TelnetTransport(config.Host, config.Port)

        # Anything the controller pushes that we didn't ask for
        # ([MSG:...], ALARM:n, banners) lands here for diagnostics.
        self.unsolicited: list[str] = []

    # -- lifecycle -----------------------------------------------------

    def connect(self) -> None:
        if not self.transport.is_connected:
            self.transport.connect()

        banner = self.transport.drain()
        if banner:
            logger.info("FluidNC banner: %s", banner)
            self.unsolicited.extend(banner)

        # Known-good modal state for everything we do: mm units, absolute
        # coords, units-per-minute feed. On a freshly powered board
        # (must_home -> Alarm state) G-code is locked out with error:9;
        # that is fine — home() re-sends the modal line after homing.
        try:
            self.send_command("G21 G90 G94")
        except FluidNCCommandError as ex:
            if "error:9" in str(ex):
                logger.info(
                    "Machine is alarm-locked (not homed yet); modal state "
                    "will be set after homing."
                )
            else:
                raise

    def close(self) -> None:
        self.transport.close()

    def __enter__(self) -> "FluidNCClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- protocol primitives -------------------------------------------

    def send_command(self, line: str, timeout_s: Optional[float] = None) -> list[str]:
        """
        Send one G-code / $-command line and block until ok or error:N.

        Returns any informational lines received before the ok (e.g. the
        output of $Report or $$ style queries).
        """

        timeout_s = timeout_s if timeout_s is not None else self.config.CommandTimeout_s

        self.transport.write((line.strip() + "\n").encode("utf-8"))

        received: list[str] = []
        deadline = time.monotonic() + timeout_s

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FluidNCTimeoutError(
                    f"Timed out after {timeout_s:g}s waiting for ok to {line!r}. "
                    f"Received so far: {received!r}"
                )

            reply = self.transport.read_line(timeout_s=remaining)
            if reply is None:
                continue

            if reply == "ok":
                return received

            if reply.startswith("error:"):
                code = _parse_trailing_int(reply)
                hint = GRBL_ERROR_HINTS.get(code, "")
                raise FluidNCCommandError(
                    f"FluidNC rejected {line!r}: {reply}. {hint}".strip()
                )

            if reply.startswith("ALARM:"):
                code = _parse_trailing_int(reply)
                hint = GRBL_ALARM_HINTS.get(code, "")
                raise FluidNCAlarmError(f"{reply} while running {line!r}. {hint}".strip())

            received.append(reply)

        # unreachable

    def query_status(self, timeout_s: float = 2.0) -> FluidNCStatus:
        """
        Send the realtime '?' query and return the parsed status report.
        """

        self.transport.write(b"?")

        deadline = time.monotonic() + timeout_s

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FluidNCTimeoutError("Timed out waiting for a status report.")

            reply = self.transport.read_line(timeout_s=remaining)
            if reply is None:
                continue

            status = parse_status_report(reply)
            if status is not None:
                return status

            if reply.startswith("ALARM:"):
                code = _parse_trailing_int(reply)
                hint = GRBL_ALARM_HINTS.get(code, "")
                raise FluidNCAlarmError(f"{reply}. {hint}".strip())

            self.unsolicited.append(reply)

    # -- realtime (single-byte, no ok expected) ------------------------

    def feed_hold(self) -> None:
        self.transport.write(b"!")

    def resume(self) -> None:
        self.transport.write(b"~")

    def soft_reset(self) -> None:
        self.transport.write(b"\x18")

    # -- firmware configuration ----------------------------------------

    def read_config_value(self, path: str) -> Optional[str]:
        """
        Read one FluidNC config item, e.g. read_config_value(
        "axes/x/max_travel_mm") -> "120.000". Returns None (with a logged
        warning) if the item cannot be read or parsed.
        """

        command = f"$/{path.lstrip('/')}"

        try:
            lines = self.send_command(command)
        except FluidNCError as ex:
            logger.warning(f"Config read {command} failed: {ex}")
            return None

        wanted = path.strip("/").lower()

        for line in lines:
            key, sep, value = line.partition("=")
            if sep and key.strip().lstrip("$").strip("/").lower() == wanted:
                return value.strip()

        logger.warning(f"Config read {command}: no value in reply {lines!r}")
        return None

    def read_soft_limits(self, margin_mm: float = 0.0) -> Optional[Bounds3D]:
        """
        The firmware's ACTUAL soft-limit ranges, derived per axis from
        max_travel_mm, homing mpos_mm, and homing direction (FluidNC
        allows [mpos - travel, mpos] for positive-homing axes and
        [mpos, mpos + travel] for negative-homing ones). margin_mm shrinks
        the returned bounds on every side.

        Returns None (with logged warnings) if any item cannot be read —
        callers should fall back to conservative hardcoded limits.
        """

        bounds: dict[str, float] = {}

        for axis in ("x", "y", "z"):
            travel_s = self.read_config_value(f"axes/{axis}/max_travel_mm")
            mpos_s = self.read_config_value(f"axes/{axis}/homing/mpos_mm")
            positive_s = self.read_config_value(
                f"axes/{axis}/homing/positive_direction"
            )

            if travel_s is None or mpos_s is None or positive_s is None:
                logger.warning(
                    f"Could not read firmware soft limits for axis "
                    f"{axis.upper()}; falling back to hardcoded limits."
                )
                return None

            try:
                travel = float(travel_s)
                mpos = float(mpos_s)
            except ValueError as ex:
                logger.warning(
                    f"Unparseable firmware limit values for axis "
                    f"{axis.upper()} ({ex}); falling back to hardcoded limits."
                )
                return None

            positive = positive_s.strip().lower() in ("true", "yes", "1")

            if positive:
                lo, hi = mpos - travel, mpos
            else:
                lo, hi = mpos, mpos + travel

            bounds[f"{axis}_min_mm"] = lo + margin_mm
            bounds[f"{axis}_max_mm"] = hi - margin_mm

        limits = Bounds3D(**bounds)

        logger.info(
            f"Firmware soft limits (margin {margin_mm:g} mm): "
            f"X {limits.x_min_mm:g}..{limits.x_max_mm:g}, "
            f"Y {limits.y_min_mm:g}..{limits.y_max_mm:g}, "
            f"Z {limits.z_min_mm:g}..{limits.z_max_mm:g}"
        )

        return limits

    def set_config_value(self, path: str, value) -> None:
        """
        Set one FluidNC config item at runtime, e.g. set_config_value(
        "axes/z/max_travel_mm", 90). VOLATILE: lost on reboot unless the
        controller's config.yaml is updated too.
        """

        self.send_command(f"$/{path.lstrip('/')}={value}")

    def jog_incremental(
        self,
        axis: str,
        delta_mm: float,
        feed_mm_min: float = 150.0,
        timeout_s: float = 120.0,
    ) -> FluidNCStatus:
        """
        Relative jog on one axis ($J=G91), blocking until motion stops.
        Jogs against an ENABLED soft limit are rejected with error:15
        (no alarm, no position loss).
        """

        self.send_command(
            f"$J=G91 {axis.upper()}{delta_mm:.3f} F{feed_mm_min:.1f}"
        )
        return self.wait_until_idle(timeout_s=timeout_s)

    # -- high-level operations -----------------------------------------

    def unlock(self) -> None:
        """$X — clear an alarm WITHOUT re-establishing position. Prefer home()."""
        self.send_command("$X")

    def home(self, axes: str = "") -> None:
        """
        $H (all axes: Z first, then X+Y auto-square) or $H<axis>.
        Blocks until homing finishes, then (re-)establishes the modal
        state, which connect() may have been unable to set on an
        alarm-locked (not-yet-homed) machine.
        """

        command = f"$H{axes.upper()}" if axes else "$H"
        self.send_command(command, timeout_s=self.config.HomingTimeout_s)
        self.send_command("G21 G90 G94")

    def move_machine(
        self,
        x_mm: Optional[float] = None,
        y_mm: Optional[float] = None,
        z_mm: Optional[float] = None,
        feed_mm_min: Optional[float] = None,
        wait: bool = True,
        timeout_s: Optional[float] = None,
    ) -> None:
        """
        Absolute machine-coordinate move (G53 G1). Omitted axes stay put.

        With wait=True (default), blocks until the controller reports Idle
        at the commanded position.
        """

        parts = ["G53", "G1"]

        for label, value in (("X", x_mm), ("Y", y_mm), ("Z", z_mm)):
            if value is not None:
                parts.append(f"{label}{value:.3f}")

        if len(parts) == 2:
            raise ValueError("move_machine called with no axis targets.")

        feed = feed_mm_min if feed_mm_min is not None else self.config.Feed_mm_min
        parts.append(f"F{feed:.1f}")

        self.send_command(" ".join(parts))

        if wait:
            self.wait_until_idle(
                timeout_s=timeout_s,
                target=Vec3D(
                    x_mm=x_mm if x_mm is not None else float("nan"),
                    y_mm=y_mm if y_mm is not None else float("nan"),
                    z_mm=z_mm if z_mm is not None else float("nan"),
                ),
            )

    def wait_until_idle(
        self,
        timeout_s: Optional[float] = None,
        target: Optional[Vec3D] = None,
    ) -> FluidNCStatus:
        """
        Poll '?' until the machine reports Idle (and, if target given, MPos
        within PositionTolerance_mm on every non-NaN axis).
        """

        timeout_s = timeout_s if timeout_s is not None else 60.0
        deadline = time.monotonic() + timeout_s

        while True:
            status = self.query_status()

            if status.is_alarm:
                raise FluidNCAlarmError(
                    f"Machine entered Alarm state while waiting for Idle: {status.Raw}"
                )

            if status.is_idle and self._at_target(status, target):
                return status

            if time.monotonic() > deadline:
                raise FluidNCTimeoutError(
                    f"Machine not Idle at target after {timeout_s:g}s "
                    f"(last status: {status.Raw})."
                )

            time.sleep(self.config.StatusPollInterval_s)

    def _at_target(self, status: FluidNCStatus, target: Optional[Vec3D]) -> bool:
        if target is None:
            return True

        if status.MPos is None:
            return False

        tolerance = self.config.PositionTolerance_mm

        for got, want in (
            (status.MPos.x_mm, target.x_mm),
            (status.MPos.y_mm, target.y_mm),
            (status.MPos.z_mm, target.z_mm),
        ):
            if want == want and abs(got - want) > tolerance:  # want==want: skip NaN
                return False

        return True


def _parse_trailing_int(line: str) -> Optional[int]:
    match = re.search(r"(\d+)\s*$", line)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# StageController implementation for FLIRDatasetWriter
# ---------------------------------------------------------------------------


# Conservative FALLBACK envelope, used only when the firmware's soft
# limits cannot be read (read_soft_limits is the source of truth): 5 mm
# inside the config.yaml travels as of 2026-07-22 — max_travel X 110 /
# Y 160 / Z 90 with homing mpos 3, giving firmware ranges X [3, 113],
# Y [3, 163], Z [-87, 3].
DEFAULT_MACHINE_LIMITS = Bounds3D(
    x_min_mm=8.0,
    x_max_mm=108.0,
    y_min_mm=8.0,
    y_max_mm=158.0,
    z_min_mm=-82.0,
    z_max_mm=-2.0,
)


@dataclass
class FluidNCStageConfig:
    MachineLimits_mm: Bounds3D = field(default_factory=lambda: DEFAULT_MACHINE_LIMITS)
    Feed_mm_min: float = 400.0

    # Extra settle after Idle, for vibration ring-down of the camera mast.
    SettleAfterIdle_s: float = 0.2


class FluidNCStageController:
    """
    StageController implementation (see dataset_writer.StageController) that
    drives the FluidNC gantry. ScanPoint.GantryPosition_mm is interpreted as
    an absolute MACHINE coordinate (G53).
    """

    def __init__(
        self,
        client: FluidNCClient,
        config: FluidNCStageConfig = None,
    ):
        self.client = client
        self.config = config or FluidNCStageConfig()

    def validate_point(self, position_mm: Vec3D) -> None:
        """
        Fail fast (before sending) on targets that would trip ALARM:2,
        which loses position and forces a reset + $H.
        """

        if not self.config.MachineLimits_mm.contains(position_mm):
            raise FluidNCError(
                f"Refusing move to {position_mm}: outside machine limits "
                f"{self.config.MachineLimits_mm}."
            )

    # -- StageController interface -------------------------------------

    def move_to_scan_point(self, point: ScanPoint, signals) -> None:
        xyz = point.GantryPosition_mm
        self.validate_point(xyz)

        signals.MovementStarted.set()

        self.client.move_machine(
            x_mm=xyz.x_mm,
            y_mm=xyz.y_mm,
            z_mm=xyz.z_mm,
            feed_mm_min=self.config.Feed_mm_min,
            # Completion is detected in wait_until_motion_complete so the
            # writer's timeout accounting stays in charge.
            wait=False,
        )

    def wait_until_motion_complete(self, point: ScanPoint, timeout_s: float, signals) -> None:
        xyz = point.GantryPosition_mm

        self.client.wait_until_idle(timeout_s=timeout_s, target=xyz)

        if self.config.SettleAfterIdle_s > 0:
            time.sleep(self.config.SettleAfterIdle_s)

        signals.MovementComplete.set()
