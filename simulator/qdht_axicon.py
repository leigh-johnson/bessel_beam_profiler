"""Quasi-discrete Hankel transform (QDHT) model of the three-axicon Bessel-beam
optic (Yu et al. 1998 QDHT; thin-element axicons; angular-spectrum propagation).

Extracted 2026-07-23 from the copies living in
`notebooks/axicon_bessel_beam_fourier_optics_2026_07_13.ipynb` (§3, §10–11) and
`notebooks/exposure_vs_z_analysis_2026_07_13.ipynb` (model cells) — this module
is now the single source of truth ("extract to simulator/ if this happens a
third time" — it did).

Geometry / conventions
----------------------
* Axicons 1 and 2 (identical, angle ``alpha_deg``) separated by ``L_sep``
  produce an annulus; axicon 3 (angle ``alpha3_deg``) sits ``L23`` after
  axicon 2 and folds the annulus into an on-axis Bessel window.
* All lengths in meters; ``simulate_*`` docstrings note the exceptions.
* Default wavelength / index match the lab's red alignment laser
  (650 nm, fused silica n = 1.4585) — pass ``k``/``n`` to override.
  TODO(leigh): confirm the actual laser wavelength at the bench.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j0 as bessel_j0, j1 as bessel_j1, jn_zeros

mm = 1e-3

WAVELENGTH_NM_DEFAULT = 650.0
N_AXICON_DEFAULT = 1.4585  # fused silica @ ~650 nm
LAMBDA = WAVELENGTH_NM_DEFAULT * 1e-9
K0 = 2 * np.pi / LAMBDA
N_AXICON = N_AXICON_DEFAULT


class QDHT:
    """Quasi-discrete Hankel transform of order 0 on r in [0, R]."""

    _cache = {}  # N -> (zeros, jN1, J1abs, T); T is independent of R

    def __init__(self, N, R):
        self.N, self.R = N, R
        if N not in QDHT._cache:
            jz = jn_zeros(0, N + 1)
            jN1 = jz[N]
            jj = jz[:N]
            J1a = np.abs(bessel_j1(jj))
            T = np.empty((N, N))  # built in row blocks to limit peak memory
            step = 1024
            for i0 in range(0, N, step):
                i1 = min(i0 + step, N)
                T[i0:i1] = (
                    (2.0 / jN1)
                    * bessel_j0(np.outer(jj[i0:i1], jj) / jN1)
                    / np.outer(J1a[i0:i1], J1a)
                )
            QDHT._cache[N] = (jj, jN1, J1a, T)
        self.j, self.jN1, self.J1, self.T = QDHT._cache[N]
        self.r = self.j * (R / self.jN1)  # radial grid [m]
        self.kr = self.j / R              # radial wavevector grid [rad/m]
        self.kr_max = self.jN1 / R

    def forward(self, f):
        return self.T @ (f / self.J1)

    def backward(self, y):
        return (self.T @ y) * self.J1

    def eval_matrix(self, r_eval):
        return (2.0 / self.jN1) * bessel_j0(np.outer(self.kr, r_eval)) / self.J1[:, None]

    def onax(self, Y):
        """|E(0)|^2 for spectrum rows Y. NB: an extrapolation — r=0 is not a grid point."""
        return np.abs((2.0 / self.jN1) * (Y @ (1.0 / self.J1))) ** 2

    def power(self, f):
        return (2.0 * self.R**2 / self.jN1**2) * np.sum(np.abs(f) ** 2 / self.J1**2)

    def power_spec(self, y):
        return (2.0 * self.R**2 / self.jN1**2) * np.sum(np.abs(y) ** 2)


def propagate_spectrum(y0, kr, k, z):
    """Angular-spectrum propagation: multiply by exp(i z sqrt(k^2 - kr^2))."""
    kz = np.sqrt(np.maximum(k * k - kr * kr, 0.0))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    Y = y0[None, :] * np.exp(1j * kz[None, :] * z[:, None])
    return Y[0] if Y.shape[0] == 1 else Y


def axicon_kr(alpha_deg, n=N_AXICON, k=K0):
    """Thin-element radial wavevector kr = k (n-1) tan(alpha)."""
    return k * (n - 1) * np.tan(np.radians(alpha_deg))


def simulate_axicon_pair(D, alpha_deg, L_sep, z_after=300 * mm, N=4096,
                         Nz1=140, Nz2=220, pad=None, k=K0, n=N_AXICON,
                         verbose=True):
    """Gaussian -> axicon 1 -> L_sep -> axicon 2 -> free space (identical thin axicons).

    D is the 1/e^2 *diameter* of the input Gaussian, in meters.
    """
    w0 = D / 2.0
    kr_ax = axicon_kr(alpha_deg, n=n, k=k)
    theta = kr_ax / k
    zmax = w0 / theta
    R_ring = theta * L_sep
    R_dom = R_ring + (5 * w0 if pad is None else pad)
    q = QDHT(N, R_dom)
    fringe = 2 * np.pi / kr_ax
    dr = q.r[1] - q.r[0]
    if verbose:
        print(f"alpha={alpha_deg} x2  D={D/mm:g} mm  L={L_sep/mm:g} mm | "
              f"zmax={zmax/mm:6.1f} mm  ring={R_ring/mm:5.2f} mm | "
              f"{fringe/dr:4.1f} pts/fringe, kr_max/kr={q.kr_max/kr_ax:4.1f}, "
              f"wall margin={(R_dom-R_ring)/w0:4.1f} w0")
    assert L_sep > 1.2 * zmax, "separation should exceed zmax for a clean annulus"
    assert q.kr_max > 1.5 * kr_ax and fringe / dr > 2.5

    t_ax = np.exp(-1j * kr_ax * q.r)
    E0 = np.exp(-(q.r / w0) ** 2)
    y1 = q.forward(E0 * t_ax)
    z1 = np.linspace(0.0, L_sep, Nz1)
    Y1 = propagate_spectrum(y1, q.kr, k, z1)
    E_at2 = q.backward(Y1[-1])
    y2 = q.forward(E_at2 * t_ax)
    z2 = np.linspace(0.0, z_after, Nz2)
    Y2 = propagate_spectrum(y2, q.kr, k, z2)
    return dict(D=D, w0=w0, alpha_deg=alpha_deg, kr=kr_ax, theta=theta, zmax=zmax,
                L_sep=L_sep, R_ring=R_ring, q=q, z1=z1, z2=z2, Y1=Y1, Y2=Y2,
                k=k, n=n,
                onax2=q.onax(Y2), P_in=q.power(E0), P_out=q.power_spec(y2))


def ring_at(pair_sim, L23):
    """Field, ring peak radius, and ring FWHM at plane L23 after axicon 2."""
    q = pair_sim["q"]
    iz = np.argmin(np.abs(pair_sim["z2"] - L23))
    E_in = q.backward(pair_sim["Y2"][iz])
    I = np.abs(E_in) ** 2
    i0 = np.searchsorted(q.r, 0.4 * pair_sim["R_ring"])
    j = i0 + np.argmax(I[i0:])
    half = 0.5 * I[j]
    lo = np.where(I[i0:j] < half)[0]
    hi = np.where(I[j:] < half)[0]
    wid = q.r[j + hi[0]] - q.r[i0 + lo[-1]] if len(lo) and len(hi) else np.nan
    return E_in, q.r[j], wid


def simulate_third_axicon(pair_sim, alpha3_deg, L23=100 * mm, Nz3=300,
                          verbose=True):
    """Apply axicon 3 in the annulus (L23 after axicon 2) and propagate."""
    q = pair_sim["q"]
    k = pair_sim.get("k", K0)
    n = pair_sim.get("n", N_AXICON)
    kr3 = axicon_kr(alpha3_deg, n=n, k=k)
    th3 = kr3 / k
    E_in, R_a, Delta = ring_at(pair_sim, L23)
    I_ring = np.abs(E_in) ** 2
    y3 = q.forward(E_in * np.exp(-1j * kr3 * q.r))
    z3 = np.linspace(1e-3, 1.3 * (R_a + Delta) / th3, Nz3)
    Y3 = propagate_spectrum(y3, q.kr, k, z3)
    onax = q.onax(Y3)
    onax_th = 2 * np.pi * kr3**2 / k * z3 * np.interp(th3 * z3, q.r, I_ring)
    z_wall = (q.R + R_a) / th3
    if verbose:
        print(f"alpha3={alpha3_deg}: ring R_a={R_a/mm:.2f} mm, "
              f"Delta={Delta/mm:.2f} mm -> zone "
              f"{(R_a-Delta/2)/th3:.2f}-{(R_a+Delta/2)/th3:.2f} m | "
              f"wall headroom={z_wall/z3[-1]:.1f}x")
    return dict(alpha3=alpha3_deg, kr3=kr3, th3=th3, R_a=R_a, Delta=Delta,
                z3=z3, y3=y3, Y3=Y3, onax=onax, onax_th=onax_th, q=q,
                P3=q.power_spec(y3))


def simulate_recorded_optic(optic, N=4096, Nz2=40, Nz3=300, k=K0, n=N_AXICON,
                            waist_is_radius=False, verbose=True):
    """Run the full pair + third-axicon model from an OpticConfiguration dict
    (as recorded in sweep_setup.json / auto_scan_setup.json).

    ``waist_is_radius=False`` interprets GaussianBeamWaist_mm as the 1/e^2
    *diameter* — the reading that matched the 2026-07-13 data to within 5%
    (see exposure_vs_z_analysis_2026_07_13.ipynb).
    Returns (pair, model) dicts.
    """
    assert optic["Axicon1_deg"] == optic["Axicon2_deg"], (
        "the pair model assumes identical axicons 1 and 2"
    )
    D_in = (2 if waist_is_radius else 1) * optic["GaussianBeamWaist_mm"] * mm
    pair = simulate_axicon_pair(
        D=D_in,
        alpha_deg=optic["Axicon1_deg"],
        L_sep=optic["L12_mm"] * mm,
        z_after=2 * optic["L23_mm"] * mm,
        N=N,
        Nz1=8,                 # only the final plane of leg 1 is used
        Nz2=Nz2,
        pad=2.5 * D_in / 2,    # tighter domain so the radial grid still
                               # resolves the axicon fringes at this D
        k=k, n=n,
        verbose=verbose,
    )
    model = simulate_third_axicon(pair, optic["Axicon3_deg"],
                                  L23=optic["L23_mm"] * mm, Nz3=Nz3,
                                  verbose=verbose)
    return pair, model
