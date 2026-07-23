"""
Tests for the CLI-layer soft-limit bounds checking and the corrected
re-runnable command generation.
"""

import pytest

from coordinates import Bounds3D
from fluidnc_stage_cli import check_bounds_against_limits, rewrite_argv


LIMITS = Bounds3D(
    x_min_mm=1.0, x_max_mm=119.0,
    y_min_mm=1.0, y_max_mm=159.0,
    z_min_mm=-126.0, z_max_mm=2.0,
)


def test_in_range_bounds_produce_no_violations():
    violations, replacements = check_bounds_against_limits(
        {"--x-min": (5.0, "x"), "--z-max": (-3.0, "z"), "--y-start": (10.0, "y")},
        LIMITS,
    )

    assert violations == {}
    assert replacements == {}


def test_out_of_range_bounds_are_clamped_per_axis():
    violations, replacements = check_bounds_against_limits(
        {
            "--x-min": (0.0, "x"),      # below x floor -> 1.0
            "--x-max": (115.0, "x"),    # fine
            "--z-min": (-130.0, "z"),   # below z floor -> -126.0
            "--z-max": (5.0, "z"),      # above z ceiling -> 2.0
            "--calibration-x": (None, "x"),  # unset flags are skipped
        },
        LIMITS,
    )

    assert violations == {
        "--x-min": (0.0, 1.0),
        "--z-min": (-130.0, -126.0),
        "--z-max": (5.0, 2.0),
    }
    assert replacements == {"--x-min": 1.0, "--z-min": -126.0, "--z-max": 2.0}


def test_rewrite_argv_replaces_space_separated_values():
    argv = [
        "cli.py", "dataset", "auto",
        "--x-min", "0", "--x-max", "115",
        "--z-min", "-130",
    ]

    command = rewrite_argv(argv, {"--x-min": 1.0, "--z-min": -126.0})

    assert command == (
        "python cli.py dataset auto --x-min 1 --x-max 115 --z-min -126"
    )


def test_rewrite_argv_replaces_equals_form_and_appends_missing_flags():
    argv = ["cli.py", "gantry", "move", "--z=-130"]

    command = rewrite_argv(argv, {"--z": -126.0, "--x": 60.0})

    assert command == "python cli.py gantry move --z=-126 --x 60"


def test_resolve_scan_caps_defaults_to_envelope_and_keeps_explicit():
    from auto_scan_cli import resolve_scan_caps

    x_min, x_max, z_min, z_max = resolve_scan_caps(LIMITS, None, None, None, None)
    assert (x_min, x_max, z_min, z_max) == (1.0, 119.0, -126.0, 2.0)

    x_min, x_max, z_min, z_max = resolve_scan_caps(LIMITS, 40.0, None, -95.0, -2.0)
    assert (x_min, x_max, z_min, z_max) == (40.0, 119.0, -95.0, -2.0)
