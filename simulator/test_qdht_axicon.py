"""Smoke tests for simulator.qdht_axicon (extracted 2026-07-23 from the
notebook copies). These check the transform and the physics invariants, not
fine numerical accuracy. NB: the end-to-end tests need N=4096 — smaller
grids cannot resolve the 16 µm axicon fringes over the ~13 mm domain and
trip the module's own resolution guard."""

import numpy as np
import pytest

try:  # runnable both from the repo root (pytest simulator/) and standalone
    from simulator.qdht_axicon import (
        QDHT, K0, N_AXICON, axicon_kr, mm, propagate_spectrum,
        simulate_axicon_pair, simulate_recorded_optic, simulate_third_axicon,
    )
except ImportError:
    from qdht_axicon import (
        QDHT, K0, N_AXICON, axicon_kr, mm, propagate_spectrum,
        simulate_axicon_pair, simulate_recorded_optic, simulate_third_axicon,
    )

OPTIC = {  # as recorded in the jul13/jul22-23 sweep_setup / auto_scan_setup
    "GaussianBeamWaist_mm": 4.59,
    "Axicon1_deg": 5.0,
    "Axicon2_deg": 5.0,
    "Axicon3_deg": 0.5,
    "L12_mm": 190.0,
    "L23_mm": 50.0,
}


def test_axicon_kr_small_angle():
    kr = axicon_kr(0.5)
    expected = K0 * (N_AXICON - 1) * np.tan(np.radians(0.5))
    assert kr == pytest.approx(expected, rel=1e-12)
    # 0.5 deg fused-silica axicon at 650 nm: kr ~ 3.87e4 rad/m (core ~58 um FWHM)
    assert 3.5e4 < kr < 4.2e4


def test_qdht_round_trip_and_parseval():
    q = QDHT(512, 5 * mm)
    f = np.exp(-((q.r / (1 * mm)) ** 2))
    y = q.forward(f)
    f_back = q.backward(y)
    assert np.allclose(f_back, f, atol=1e-10 * f.max())
    assert q.power_spec(y) == pytest.approx(q.power(f), rel=1e-10)


def test_propagate_spectrum_preserves_power_and_shape():
    q = QDHT(512, 5 * mm)
    y0 = q.forward(np.exp(-((q.r / (1 * mm)) ** 2)) + 0j)
    Y = propagate_spectrum(y0, q.kr, K0, [0.0, 0.1, 0.2])
    assert Y.shape == (3, 512)
    assert np.allclose(Y[0], y0)                       # z=0 is identity
    for row in Y:                                      # unitary in each plane
        assert q.power_spec(row) == pytest.approx(q.power_spec(y0), rel=1e-12)


def test_pair_ring_radius_matches_geometry():
    pair = simulate_axicon_pair(
        D=OPTIC["GaussianBeamWaist_mm"] * mm, alpha_deg=5.0, L_sep=190 * mm,
        z_after=100 * mm, N=4096, Nz1=4, Nz2=8,
        pad=2.5 * OPTIC["GaussianBeamWaist_mm"] * mm / 2, verbose=False,
    )
    # annulus radius ~ theta * L_sep
    assert pair["R_ring"] == pytest.approx(pair["theta"] * 190 * mm, rel=1e-12)
    # energy conserved through both axicons (phase-only elements)
    assert pair["P_out"] == pytest.approx(pair["P_in"], rel=1e-6)


def test_third_axicon_window_position():
    pair, model = simulate_recorded_optic(OPTIC, N=4096, Nz2=8, Nz3=200,
                                          verbose=False)
    z_pk = model["z3"][np.argmax(model["onax"])]
    # window center ~ R_a / tan(theta3); nominal geometry puts it near 1.9 m
    z_geom = model["R_a"] / model["th3"]
    assert z_pk == pytest.approx(z_geom, rel=0.25)
    assert 1.2 < z_pk < 2.6
    # window must be bright compared to the annulus edges of the sweep
    assert model["onax"].max() > 20 * model["onax"][0]
