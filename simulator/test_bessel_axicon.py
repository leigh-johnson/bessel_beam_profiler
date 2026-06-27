import pytest

from .bessel_axicon import (
    AxiconParams,
    zmax_axicon,
    bessel_gauss_intensity,
)


def test_rao_table_1_axicon_zrange_and_zpeak_for_0th_order():
    """
    Rao sanity check:
    1064 nm beam, w = 1.5 mm, n = 1.5, alpha = 1 deg

    Expected:
        zrange ≈ 17.2 cm
        zpeak  ≈ 8.6 cm for ell = 0
    """
    params = AxiconParams(
        wavelength_nm=1064,
        n=1.5,
        alpha_deg=1.0,
    )

    ell = 0
    w_um = 1500

    zrange_m = zmax_axicon(w_um, params)
    zrange_cm = zrange_m * 100

    zpeak_m = zrange_m / 2
    zpeak_cm = zpeak_m * 100

    assert zrange_cm == pytest.approx(17.2, abs=0.05)
    assert zpeak_cm == pytest.approx(8.6, abs=0.05)

    # Since normalize_peak=True, the on-axis intensity at zpeak should be 1.
    I_peak = bessel_gauss_intensity(
        r_m=0,
        z_m=zpeak_m,
        w_um=w_um,
        params=params,
        ell=ell,
        normalize_peak=True,
    )

    assert I_peak == pytest.approx(1.0)