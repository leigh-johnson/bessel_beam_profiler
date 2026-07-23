"""
Tests for the set-limits helpers: travel/range math, the comment-preserving
config.yaml patcher, and the runtime config-set / jog line formats.
"""

import pytest

from fluidnc_stage import FluidNCClient, FluidNCClientConfig
from set_limits_cli import (
    LimitSettingError,
    far_limit_to_travel,
    patch_max_travel,
    travel_to_range,
)
from test_fluidnc_stage import FakeTransport


# Structure mirrors the real cnc_gantry_mounts/config.yaml.
CONFIG_SNIPPET = """axes:
  shared_stepper_disable_pin: NO_PIN
  homing_runs: 2
  x:
    steps_per_mm: 1600.000000
    max_travel_mm: 110.000000
    soft_limits: true
    homing:
      positive_direction: false
      mpos_mm: 3.000000

    motor0:
      limit_neg_pin: gpio.25

  y:
    max_travel_mm: 160.000000
    soft_limits: true

  z:
    max_travel_mm: 90.000000
    soft_limits: true
    homing:
      positive_direction: true
      mpos_mm: 3.000000

control:
  reset_pin: NO_PIN
"""


# ---------------------------------------------------------------------------
# Travel math
# ---------------------------------------------------------------------------


def test_far_limit_to_travel_positive_homing():
    # Z homes to +3 at the top; far limit below it.
    assert far_limit_to_travel(3.0, True, -87.0) == 90.0


def test_far_limit_to_travel_negative_homing():
    # X homes to +3 at the bottom; far limit above it.
    assert far_limit_to_travel(3.0, False, 113.0) == 110.0


def test_far_limit_on_wrong_side_raises():
    with pytest.raises(LimitSettingError, match="below"):
        far_limit_to_travel(3.0, True, 10.0)

    with pytest.raises(LimitSettingError, match="above"):
        far_limit_to_travel(3.0, False, -5.0)


def test_travel_to_range_round_trip():
    assert travel_to_range(3.0, True, 90.0) == (-87.0, 3.0)
    assert travel_to_range(3.0, False, 110.0) == (3.0, 113.0)


# ---------------------------------------------------------------------------
# config.yaml patcher
# ---------------------------------------------------------------------------


def test_patch_updates_only_the_requested_axis():
    patched = patch_max_travel(CONFIG_SNIPPET, "z", 85.5)

    assert "max_travel_mm: 85.500000" in patched
    assert "max_travel_mm: 110.000000" in patched  # x untouched
    assert "max_travel_mm: 160.000000" in patched  # y untouched
    # Everything else byte-identical.
    assert patched.replace("85.500000", "90.000000") == CONFIG_SNIPPET


def test_patch_each_axis_independently():
    text = CONFIG_SNIPPET
    text = patch_max_travel(text, "x", 108.0)
    text = patch_max_travel(text, "y", 155.0)

    assert "max_travel_mm: 108.000000" in text
    assert "max_travel_mm: 155.000000" in text
    assert "max_travel_mm: 90.000000" in text  # z untouched


def test_patch_raises_on_unrecognized_structure():
    with pytest.raises(LimitSettingError, match="not recognized"):
        patch_max_travel("axes:\n  q:\n    max_travel_mm: 1\n", "x", 100.0)


# ---------------------------------------------------------------------------
# Runtime set + jog line formats
# ---------------------------------------------------------------------------


def make_client():
    transport = FakeTransport()
    client = FluidNCClient(
        FluidNCClientConfig(CommandTimeout_s=0.5, StatusPollInterval_s=0.0),
        transport=transport,
    )
    return client, transport


def test_set_config_value_line_format():
    client, transport = make_client()

    client.set_config_value("axes/z/max_travel_mm", "85.500")

    assert transport.sent[-1] == b"$/axes/z/max_travel_mm=85.500\n"


def test_jog_incremental_line_format_and_wait():
    client, transport = make_client()
    transport.status_reports.append("<Idle|MPos:60.000,80.000,-50.000|FS:0,0>")

    client.jog_incremental("z", -2.5, feed_mm_min=150.0)

    assert transport.sent[0] == b"$J=G91 Z-2.500 F150.0\n"
    assert transport.sent[-1] == b"?"  # waited for idle
