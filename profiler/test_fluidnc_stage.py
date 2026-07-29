"""
Unit tests for the FluidNC client and stage controller, using a scripted
fake transport (no network, no PySpin).
"""

from collections import deque
import threading
import types

import pytest

from coordinates import Bounds3D, ScanPoint, Vec3D
from fluidnc_stage import (
    FluidNCAlarmError,
    FluidNCClient,
    FluidNCClientConfig,
    FluidNCCommandError,
    FluidNCError,
    FluidNCStageConfig,
    FluidNCStageController,
    FluidNCTimeoutError,
    parse_status_report,
)


class FakeTransport:
    """
    Replies 'ok' to every line unless a scripted response is registered.
    '?' pops the next scripted status report (Idle at origin by default).
    """

    def __init__(self):
        self.sent: list[bytes] = []
        self.rx: deque[str] = deque()
        self.status_reports: deque[str] = deque()
        self.responses: dict[str, list[str]] = {}
        self.closed = False

    @property
    def is_connected(self) -> bool:
        return not self.closed

    def connect(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def drain(self) -> list[str]:
        return []

    def write(self, data: bytes) -> None:
        self.sent.append(data)

        if data == b"?":
            if self.status_reports:
                self.rx.append(self.status_reports.popleft())
            else:
                self.rx.append("<Idle|MPos:0.000,0.000,0.000|FS:0,0>")
            return

        if data in (b"!", b"~", b"\x18"):
            return

        line = data.decode().strip()
        self.rx.extend(self.responses.get(line, ["ok"]))

    def read_line(self, timeout_s: float):
        if self.rx:
            return self.rx.popleft()
        return None


def make_client(transport=None) -> tuple[FluidNCClient, FakeTransport]:
    transport = transport or FakeTransport()
    client = FluidNCClient(
        FluidNCClientConfig(
            CommandTimeout_s=0.5,
            StatusPollInterval_s=0.0,
        ),
        transport=transport,
    )
    return client, transport


def make_signals():
    return types.SimpleNamespace(
        MovementStarted=threading.Event(),
        MovementComplete=threading.Event(),
    )


# ---------------------------------------------------------------------------
# Status parsing
# ---------------------------------------------------------------------------


def test_parse_status_report_idle_with_mpos():
    status = parse_status_report(
        "<Idle|MPos:60.000,80.000,-50.000|FS:0,0|WCO:0.000,0.000,0.000>"
    )

    assert status is not None
    assert status.is_idle
    assert not status.is_alarm
    assert status.MPos == Vec3D(x_mm=60.0, y_mm=80.0, z_mm=-50.0)


def test_parse_status_report_alarm_and_run_states():
    assert parse_status_report("<Alarm|MPos:0.000,0.000,0.000>").is_alarm
    run = parse_status_report("<Run|MPos:1.000,-2.500,3.000|FS:400,0>")
    assert run.State == "Run"
    assert run.MPos.y_mm == -2.5


def test_parse_status_report_extracts_triggered_pins():
    status = parse_status_report("<Alarm|MPos:2.974,-16.279,3.000|FS:0,0|Pn:Y>")
    assert status.Pins == "Y"

    stuck = parse_status_report("<Idle|MPos:50.000,80.000,-50.000|FS:0,0|Pn:X>")
    assert stuck.Pins == "X"  # away from the switch: loose signal wire

    clean = parse_status_report("<Idle|MPos:3.000,3.000,3.000|FS:0,0>")
    assert clean.Pins == ""


def test_parse_status_report_rejects_non_status_lines():
    assert parse_status_report("ok") is None
    assert parse_status_report("[MSG:INFO: Homing done]") is None
    assert parse_status_report("error:20") is None


# ---------------------------------------------------------------------------
# Client protocol
# ---------------------------------------------------------------------------


def test_connect_sets_modal_state():
    client, transport = make_client()
    client.connect()

    assert transport.sent == [b"G21 G90 G94\n"]


def test_send_command_returns_informational_lines_before_ok():
    client, transport = make_client()
    transport.responses["$SS"] = ["[MSG:INFO: line1]", "[MSG:INFO: line2]", "ok"]

    received = client.send_command("$SS")

    assert received == ["[MSG:INFO: line1]", "[MSG:INFO: line2]"]


def test_send_command_raises_on_error_with_hint():
    client, transport = make_client()
    transport.responses["G1 X999"] = ["error:20"]

    with pytest.raises(FluidNCCommandError, match="error:20"):
        client.send_command("G1 X999")


def test_send_command_raises_on_alarm():
    client, transport = make_client()
    transport.responses["G53 G1 X190.000 F400.0"] = ["ALARM:2"]

    with pytest.raises(FluidNCAlarmError, match="Soft limit"):
        client.send_command("G53 G1 X190.000 F400.0")


def test_send_command_times_out():
    client, transport = make_client()
    transport.responses["$H"] = []  # never replies

    with pytest.raises(FluidNCTimeoutError):
        client.send_command("$H", timeout_s=0.2)


def test_move_machine_formats_g53_line_and_waits_for_target():
    client, transport = make_client()
    transport.status_reports.append("<Run|MPos:10.000,80.000,-50.000|FS:400,0>")
    transport.status_reports.append("<Idle|MPos:60.000,80.000,-50.000|FS:0,0>")

    client.move_machine(x_mm=60.0, y_mm=80.0, z_mm=-50.0)

    assert b"G53 G1 X60.000 Y80.000 Z-50.000 F400.0\n" in transport.sent
    # Two '?' polls: one Run, one Idle-at-target.
    assert transport.sent.count(b"?") == 2


def test_move_machine_partial_axes_omits_unset_axes():
    client, transport = make_client()

    client.move_machine(z_mm=-30.0, wait=False)

    assert transport.sent[-1] == b"G53 G1 Z-30.000 F400.0\n"


def test_wait_until_idle_keeps_polling_until_position_matches():
    client, transport = make_client()
    # Idle but at the WRONG position first (e.g. stale report), then correct.
    transport.status_reports.append("<Idle|MPos:0.000,0.000,0.000|FS:0,0>")
    transport.status_reports.append("<Idle|MPos:5.000,6.000,-7.000|FS:0,0>")

    status = client.wait_until_idle(
        timeout_s=1.0, target=Vec3D(x_mm=5.0, y_mm=6.0, z_mm=-7.0)
    )

    assert status.MPos == Vec3D(x_mm=5.0, y_mm=6.0, z_mm=-7.0)


def test_wait_until_idle_raises_on_alarm_state():
    client, transport = make_client()
    transport.status_reports.append("<Alarm|MPos:0.000,0.000,0.000>")

    with pytest.raises(FluidNCAlarmError, match="Alarm"):
        client.wait_until_idle(timeout_s=1.0)


def test_query_status_stores_unsolicited_lines():
    client, transport = make_client()
    transport.rx.append("[MSG:INFO: probe]")
    transport.status_reports.append("<Idle|MPos:0.000,0.000,0.000|FS:0,0>")

    status = client.query_status()

    assert status.is_idle
    assert "[MSG:INFO: probe]" in client.unsolicited


def test_home_homes_then_restores_modal_state():
    client, transport = make_client()

    client.home()
    assert transport.sent[-2] == b"$H\n"
    assert transport.sent[-1] == b"G21 G90 G94\n"

    client.home("z")
    assert transport.sent[-2] == b"$HZ\n"


def test_connect_tolerates_alarm_locked_modal_state():
    client, transport = make_client()
    # must_home boot: G-code is locked out until $H.
    transport.responses["G21 G90 G94"] = ["error:9"]

    client.connect()  # must not raise

    # Homing clears the alarm; the modal line goes through afterwards.
    transport.responses["G21 G90 G94"] = ["ok"]
    client.home()
    assert transport.sent[-1] == b"G21 G90 G94\n"


def test_connect_still_raises_on_other_modal_errors():
    client, transport = make_client()
    transport.responses["G21 G90 G94"] = ["error:20"]

    with pytest.raises(FluidNCCommandError, match="error:20"):
        client.connect()


# ---------------------------------------------------------------------------
# Firmware config / soft-limit reads
# ---------------------------------------------------------------------------


def prime_soft_limit_responses(transport, travels, mposes, positives):
    for axis, travel, mpos, positive in zip("xyz", travels, mposes, positives):
        transport.responses[f"$/axes/{axis}/max_travel_mm"] = [
            f"$/axes/{axis}/max_travel_mm={travel}",
            "ok",
        ]
        transport.responses[f"$/axes/{axis}/homing/mpos_mm"] = [
            f"$/axes/{axis}/homing/mpos_mm={mpos}",
            "ok",
        ]
        transport.responses[f"$/axes/{axis}/homing/positive_direction"] = [
            f"$/axes/{axis}/homing/positive_direction={positive}",
            "ok",
        ]


def test_read_config_value_parses_key_value_reply():
    client, transport = make_client()
    transport.responses["$/axes/x/max_travel_mm"] = [
        "$/axes/x/max_travel_mm=120.000",
        "ok",
    ]

    assert client.read_config_value("axes/x/max_travel_mm") == "120.000"


def test_read_config_value_returns_none_on_unexpected_reply():
    client, transport = make_client()
    transport.responses["$/axes/x/max_travel_mm"] = ["[MSG:INFO: nope]", "ok"]

    assert client.read_config_value("axes/x/max_travel_mm") is None


def test_read_soft_limits_derives_ranges_from_homing_direction():
    client, transport = make_client()
    # X/Y home negative (switch at min, mpos 0 -> range [0, travel]);
    # Z homes positive at mpos 3 -> range [3 - travel, 3].
    prime_soft_limit_responses(
        transport,
        travels=("120.000", "160.000", "130.000"),
        mposes=("0.000", "0.000", "3.000"),
        positives=("false", "false", "true"),
    )

    limits = client.read_soft_limits()

    assert limits == Bounds3D(
        x_min_mm=0.0, x_max_mm=120.0,
        y_min_mm=0.0, y_max_mm=160.0,
        z_min_mm=-127.0, z_max_mm=3.0,
    )


def test_read_soft_limits_applies_margin():
    client, transport = make_client()
    prime_soft_limit_responses(
        transport,
        travels=("120.000", "160.000", "130.000"),
        mposes=("0.000", "0.000", "3.000"),
        positives=("false", "false", "true"),
    )

    limits = client.read_soft_limits(margin_mm=1.0)

    assert limits.x_min_mm == 1.0
    assert limits.x_max_mm == 119.0
    assert limits.z_min_mm == -126.0
    assert limits.z_max_mm == 2.0


def test_read_soft_limits_returns_none_when_any_item_unreadable():
    client, transport = make_client()
    prime_soft_limit_responses(
        transport,
        travels=("120.000", "160.000", "130.000"),
        mposes=("0.000", "0.000", "3.000"),
        positives=("false", "false", "true"),
    )
    transport.responses["$/axes/y/homing/mpos_mm"] = ["error:3"]

    assert client.read_soft_limits() is None


# ---------------------------------------------------------------------------
# Stage controller
# ---------------------------------------------------------------------------


LIMITS = Bounds3D(
    x_min_mm=5.0, x_max_mm=115.0,
    y_min_mm=5.0, y_max_mm=155.0,
    z_min_mm=-120.0, z_max_mm=-2.0,
)


def make_point(x=60.0, y=80.0, z=-50.0) -> ScanPoint:
    return ScanPoint(
        PlacementID="test",
        GantryPosition_mm=Vec3D(x_mm=x, y_mm=y, z_mm=z),
        TablePosition_mm=Vec3D(x_mm=0.0, y_mm=0.0, z_mm=1000.0),
    )


def make_controller():
    client, transport = make_client()
    controller = FluidNCStageController(
        client,
        FluidNCStageConfig(MachineLimits_mm=LIMITS, SettleAfterIdle_s=0.0),
    )
    return controller, transport


def test_controller_rejects_out_of_limits_point_before_sending():
    controller, transport = make_controller()
    signals = make_signals()

    with pytest.raises(FluidNCError, match="outside machine limits"):
        controller.move_to_scan_point(make_point(x=190.0), signals)

    assert transport.sent == []
    assert not signals.MovementStarted.is_set()


def test_controller_move_and_wait_sets_signals_in_order():
    controller, transport = make_controller()
    signals = make_signals()
    point = make_point()

    controller.move_to_scan_point(point, signals)

    assert signals.MovementStarted.is_set()
    assert not signals.MovementComplete.is_set()
    assert transport.sent[-1] == b"G53 G1 X60.000 Y80.000 Z-50.000 F400.0\n"

    transport.status_reports.append("<Run|MPos:59.000,80.000,-50.000|FS:400,0>")
    transport.status_reports.append("<Idle|MPos:60.000,80.000,-50.000|FS:0,0>")

    controller.wait_until_motion_complete(point, timeout_s=1.0, signals=signals)

    assert signals.MovementComplete.is_set()


def test_controller_wait_raises_on_alarm_without_completing():
    controller, transport = make_controller()
    signals = make_signals()
    point = make_point()

    controller.move_to_scan_point(point, signals)
    transport.status_reports.append("<Alarm|MPos:0.000,0.000,0.000>")

    with pytest.raises(FluidNCAlarmError):
        controller.wait_until_motion_complete(point, timeout_s=1.0, signals=signals)

    assert not signals.MovementComplete.is_set()


# ---------------------------------------------------------------------------
# Link auto-reconnect (WiFi drop-offs on the moving ESP32)
# ---------------------------------------------------------------------------


class DroppyTransport(FakeTransport):
    """FakeTransport whose next N writes fail like a dropped WiFi link."""

    def __init__(self, drop_next_writes=0, connects_fail=0):
        super().__init__()
        self.drop_next_writes = drop_next_writes
        self.connects_fail = connects_fail
        self.connect_count = 0

    def connect(self) -> None:
        self.connect_count += 1
        if self.connects_fail > 0:
            self.connects_fail -= 1
            raise OSError("no route to host (AP rebooting)")
        self.closed = False

    def write(self, data: bytes) -> None:
        if self.drop_next_writes > 0:
            self.drop_next_writes -= 1
            raise OSError("broken pipe (wifi drop)")
        super().write(data)


def make_reconnecting_client(transport):
    client = FluidNCClient(
        FluidNCClientConfig(
            CommandTimeout_s=0.5,
            StatusPollInterval_s=0.0,
            ReconnectAttempts=3,
            ReconnectDelay_s=0.0,
        ),
        transport=transport,
    )
    client.connect()
    return client


def test_query_status_reconnects_after_link_drop():
    transport = DroppyTransport()
    client = make_reconnecting_client(transport)
    connects_before = transport.connect_count

    transport.drop_next_writes = 1  # the '?' write dies mid-air

    status = client.query_status()

    assert status.is_idle
    assert transport.connect_count == connects_before + 1


def test_send_command_resends_after_link_drop():
    transport = DroppyTransport()
    client = make_reconnecting_client(transport)

    transport.drop_next_writes = 1

    replies = client.send_command("G53 G1 X10.000 F400.0")

    assert replies == []
    # The command reached the controller exactly once after the drop.
    sent_moves = [s for s in transport.sent if s.startswith(b"G53")]
    assert len(sent_moves) == 1


def test_reconnect_gives_up_after_configured_attempts():
    transport = DroppyTransport()
    client = make_reconnecting_client(transport)

    # Link dies AND the AP stays down: every reconnect fails.
    transport.drop_next_writes = 1
    transport.connects_fail = 99

    with pytest.raises(FluidNCError, match="after 3 attempts"):
        client.query_status()


def test_command_error_does_not_trigger_reconnect():
    transport = DroppyTransport()
    client = make_reconnecting_client(transport)
    connects_before = transport.connect_count

    transport.responses["G1 X9999"] = ["error:2"]

    with pytest.raises(FluidNCCommandError):
        client.send_command("G1 X9999")

    assert transport.connect_count == connects_before  # link never touched
