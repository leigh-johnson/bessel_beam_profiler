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


def parse_status_report(line: str) -> Optional[FluidNCStatus]:
    match = STATUS_RE.match(line.strip())

    if match is None:
        return None

    mpos = None
    mpos_match = MPOS_RE.search(match.group("fields") or "")

    if mpos_match is not None:
        mpos = Vec3D(
            x_mm=float(mpos_match.group("x")),
            y_mm=float(mpos_match.group("y")),
            z_mm=float(mpos_match.group("z")),
        )

    return FluidNCStatus(State=match.group("state"), MPos=mpos, Raw=line.strip())


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

        # Known-good modal state for everything we do:
        # mm units, absolute coords, units-per-minute feed.
        self.send_command("G21 G90 G94")

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

    # -- high-level operations -----------------------------------------

    def unlock(self) -> None:
        """$X — clear an alarm WITHOUT re-establishing position. Prefer home()."""
        self.send_command("$X")

    def home(self, axes: str = "") -> None:
        """
        $H (all axes: Z first, then X+Y auto-square) or $H<axis>.
        Blocks until homing finishes.
        """

        command = f"$H{axes.upper()}" if axes else "$H"
        self.send_command(command, timeout_s=self.config.HomingTimeout_s)

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


# 5 mm inside the homed machine travel (X 0..120, Y 0..160, Z -127..3),
# matching the envelope proven by rangetest.gcode.
DEFAULT_MACHINE_LIMITS = Bounds3D(
    x_min_mm=5.0,
    x_max_mm=115.0,
    y_min_mm=5.0,
    y_max_mm=155.0,
    z_min_mm=-120.0,
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
