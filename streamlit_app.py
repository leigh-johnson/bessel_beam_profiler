"""Interactive three-axicon Bessel-beam designer.

Run from the repo root (so `simulator/` is importable):

    pip install streamlit          # once, in the .venv
    streamlit run streamlit_app.py

Two models side by side:
* GEOMETRIC (instant): exact-Snell meridional rays -> range endpoints, the
  1/2-inch fit point going in and the 1/2-inch void-clear point going out.
* WAVE OPTICS (QDHT, `simulator/qdht_axicon.py`): on-axis / near-axis
  intensity, radial FWHM, and power-based envelope metrics. The pair stage is
  generalized here to allow alpha1 != alpha2 (the library function assumes an
  identical pair).

Conventions: all axicons deflect toward the axis (positive, as built);
"outer diameter" = diameter enclosing ENC_OUT of the power at that plane;
"void diameter" = diameter enclosing only ENC_IN of the power (the dark hole).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import streamlit as st

from scipy.special import jn_zeros

from simulator.qdht_axicon import (QDHT, propagate_spectrum, axicon_kr,
                                   ring_at, mm)

UM, MRAD = 1e-6, 1e-3
J0_ZERO = 2.40483
X_HALF = 1.12556       # J0(x)^2 = 1/2 -> half-max radius of an ideal core
HALF_INCH = 12.7 * mm  # clearance diameter for both fit and void metrics


# ----------------------------------------------------------------------------
# geometric model (exact-Snell meridional rays; all axicons toward-axis)
# ----------------------------------------------------------------------------
def deflect(alpha_rad, n):
    return np.arcsin(np.clip(n * np.sin(alpha_rad), -1, 1)) - alpha_rad


def geometric(p):
    """Ray metrics. z coordinates are measured from axicon 3."""
    w = p["D"] / 2
    h = np.linspace(1e-4 * w, w, 1500)          # h <= w: 86.5% of the power
    u = np.zeros_like(h)
    betas = deflect(np.deg2rad(np.array([p["a1"], p["a2"], p["a3"]])), p["n"])
    for L, b in zip((0.0, p["L12"], p["L23"]), betas):
        h = h + u * L
        u = u - b * np.where(h == 0, 1.0, np.sign(h))
    u = np.where(h < 0, -u, u)
    h = np.abs(h)
    g = dict(rho_in=h.min(), rho_out=h.max(), betas=betas)
    if not np.all(u < 0):
        return g | dict(ok=False)
    zc, th = -h / u, -u
    g |= dict(ok=True, z_start=zc.min(), z_end=zc.max(),
              th_end=th[np.argmax(zc)], th_mean=th.mean(),
              core_d_end=J0_ZERO * p["lam"] / (np.pi * th[np.argmax(zc)]))
    zg = np.linspace(0.0, g["z_end"], 1200)
    r_all = np.abs(h[:, None] + u[:, None] * zg)
    outer = r_all.max(axis=0)
    below = outer <= HALF_INCH / 2
    g["z_fit"] = zg[below][0] if below.any() else np.nan
    g["z_clear"] = g["z_end"] + (HALF_INCH / 2) / g["th_end"]
    return g


def ray_diagram(p, geo):
    """Schematic meridional ray diagram, x measured from axicon 1.
    Broken x-axis: left panel zooms the optics (axicons 1-3), right panel
    spans the propagation region; diagonal slashes mark the scale break."""
    z1, z2, z3p = 0.0, p["L12"], p["L12"] + p["L23"]
    w = p["D"] / 2
    x_end = z3p + (1.15 * geo["z_clear"] if geo["ok"] else 20 * p["L12"])
    optics_w = max(z3p, 1e-3)
    xl0, xl1 = z1 - 0.30 * optics_w, z3p + 0.85 * optics_w
    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(11.5, 3.6), sharey=True, constrained_layout=True,
        gridspec_kw=dict(width_ratios=[1.0, 2.4], wspace=0.03))
    span_l, span_r = xl1 - xl0, x_end - xl1
    axes = (axl, axr)

    def pick(x):
        return axl if x <= xl1 else axr

    for ax in axes:
        ax.axhline(0, color="0.6", lw=0.8, ls="-.")

    for h0 in w * np.array([0.05, 0.3, 0.6, 0.85, 1.0]):
        for sgn in (1, -1):
            zs, hs, h, u = [xl0], [sgn * h0], sgn * h0, 0.0
            for zpl, b in zip((z1, z2, z3p), geo["betas"]):
                h = h + u * (zpl - zs[-1])
                zs.append(zpl)
                hs.append(h)
                u = u - b * (1.0 if h >= 0 else -1.0)
            zs.append(x_end)
            hs.append(h + u * (x_end - z3p))
            for ax in axes:
                ax.plot(zs, np.array(hs) / mm, color="0.3", lw=0.7)

    H = max(geo["rho_out"], w) / mm
    # triangle half-width: never wide enough to overlap the 2-3 gap
    dx = min(0.018 * span_l, 0.5 * max(z3p - z2, 1e-4))  # 2 and 3 may
    #                                                     touch, never cross
    # orientations: 1 flat-first (apex downstream), 2 apex-first (mirror-
    # symmetric pair — cancels the exact-Snell residual), 3 flat-first
    crowded = (z3p - z2) < 0.15 * span_l   # stagger labels 2/3 if close
    for z, lab_t, apex, ytxt in ((z1, "axicon 1", +1, H * 1.06),
                                 (z2, "axicon 2", -1, H * 1.06),
                                 (z3p, "axicon 3", +1,
                                  H * 1.26 if crowded else H * 1.06)):
        axl.add_patch(plt.Polygon([(z - apex * dx, -H), (z - apex * dx, H),
                                   (z + apex * dx, 0)], closed=True,
                                  fc="#aadcec", ec="#2b7a9b", lw=1, zorder=3))
        axl.annotate(lab_t, (z, ytxt), ha="center", fontsize=8, color="0.3")
    axl.annotate(r"$L_{12}$", (0.5 * (z1 + z2), -H * 1.12), ha="center",
                 fontsize=8, color="0.4")
    axl.annotate(r"$L_{23}$", (0.5 * (z2 + z3p), -H * 1.32), ha="center",
                 fontsize=8, color="0.4")

    # symbol labels (slide notation): alpha = base angles, beta = deflections,
    # theta = final cone angle, rho_in/out = annulus edges at axicon 3
    lab = dict(fontsize=9, color="0.35")
    axl.annotate(r"$\alpha_1$", (z1 - 1.6 * dx, 0.55 * H), ha="right", **lab)
    axl.annotate(r"$\alpha_2$", (z2 - 1.6 * dx, 0.40 * H), ha="right", **lab)
    axl.annotate(r"$\alpha_3$", (z3p + 1.6 * dx, -0.45 * H), **lab)
    b1 = geo["betas"][0]
    xb1 = z1 + min(0.28 * span_l, 0.30 * H * mm / b1)   # keep the drop on-plot
    axl.plot([z1, xb1], [w / mm, w / mm], ls=":", color="0.55", lw=0.8)
    axl.annotate(r"$\beta_1$", (xb1 + 0.015 * span_l,
                                 (w - 0.55 * b1 * (xb1 - z1)) / mm), **lab)

    if geo["ok"]:
        xb3 = z3p + 0.45 * span_l
        axl.plot([z3p, xb3], [geo["rho_out"] / mm] * 2, ls=":", color="0.55",
                 lw=0.8)
        axl.annotate(r"$\beta_3$", (xb3 + 0.015 * span_l,
                                     (geo["rho_out"]
                                      - 0.55 * geo["th_end"] * (xb3 - z3p)) / mm),
                     **lab)
        for xr, rr, s in ((z3p + 0.12 * span_l, geo["rho_in"],
                           r"$\rho_{in}$"),
                          (z3p + 0.24 * span_l, geo["rho_out"],
                           r"$\rho_{out}$")):
            axl.annotate("", (xr, rr / mm), (xr, 0),
                         arrowprops=dict(arrowstyle="<->", color="0.35",
                                         lw=0.8))
            axl.annotate(s, (xr + 0.015 * span_l, 0.45 * rr / mm), **lab)

    if geo["ok"]:
        zs_r, ze_r = z3p + geo["z_start"], z3p + geo["z_end"]
        for ax in axes:
            for zz in (zs_r, ze_r):
                ax.axvline(zz, color="crimson", lw=0.9, ls="--")
            ax.plot([zs_r, ze_r], [0, 0], color="crimson", lw=2.5, zorder=4)
        pick(0.5 * (zs_r + ze_r)).annotate(
            f"Bessel range {geo['z_end'] - geo['z_start']:.2f} m",
            (0.5 * (zs_r + ze_r), H * 0.55), ha="center", fontsize=8.5,
            color="crimson")
        pick(zs_r).annotate(r"$\theta$",
                            (zs_r - 0.05 * span_r,
                             0.30 * geo["th_end"] * 0.05 * span_r / mm), **lab)
        R_mm = HALF_INCH / 2 / mm
        if np.isfinite(geo["z_fit"]):
            zf = z3p + geo["z_fit"]
            for ax in axes:
                ax.plot([zf, zf], [-R_mm, R_mm], color="#4a6fa5", lw=3.5,
                        solid_capstyle="butt", zorder=4)
            pick(zf).annotate("1/2'' mirror 1", (zf, -H * 0.7), ha="center",
                              fontsize=8, color="#4a6fa5")
        zc = z3p + geo["z_clear"]
        for ax in axes:
            ax.plot([zc, zc], [-R_mm, R_mm], color="#4a6fa5", lw=3.5,
                    solid_capstyle="butt", zorder=4)
        pick(zc).annotate("1/2'' mirror 2", (zc, H * 1.12), ha="center",
                          fontsize=8, color="#4a6fa5")
    else:
        axr.annotate("no complete Bessel region with these angles",
                     (xl1 + 0.5 * span_r, H * 0.7), ha="center", fontsize=9,
                     color="crimson")

    axl.set_xlim(xl0, xl1)
    axr.set_xlim(xl1, x_end)
    axl.set_ylim(-H * 1.45, H * 1.45)
    axl.set_ylabel("r (mm)")
    fig.supxlabel("z from axicon 1 (m)  —  scale break after the optics",
                  fontsize=9)
    # break marks + spine cleanup
    axl.spines["right"].set_visible(False)
    axr.spines["left"].set_visible(False)
    axr.tick_params(left=False)
    slash = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=10, ls="none",
                 color="k", mec="k", mew=1, clip_on=False)
    axl.plot([1, 1], [0, 1], transform=axl.transAxes, **slash)
    axr.plot([0, 0], [0, 1], transform=axr.transAxes, **slash)
    for ax in axes:
        for s in ("top",):
            ax.spines[s].set_visible(False)
    axr.spines["right"].set_visible(False)
    return fig


# ----------------------------------------------------------------------------
# wave optics (QDHT) — generalized pair (alpha1 != alpha2 allowed)
# ----------------------------------------------------------------------------
def run_wave(p, geo):
    k = 2 * np.pi / p["lam"]
    w0 = p["D"] / 2
    kr1 = axicon_kr(p["a1"], n=p["n"], k=k)
    kr2 = axicon_kr(p["a2"], n=p["n"], k=k)
    kr3 = axicon_kr(p["a3"], n=p["n"], k=k)
    th1 = kr1 / k
    zmax1 = w0 / th1
    R_ring = th1 * p["L12"]
    q = QDHT(p["N"], R_ring + p["pad_w0"] * w0)
    notes = []
    fringe_pts = (2 * np.pi / max(kr1, kr2, kr3)) / (q.r[1] - q.r[0])
    if p["L12"] < 2.4 * zmax1:
        notes.append(f"L12 = {p['L12']/zmax1:.1f} x zmax1 — annulus not cleanly "
                     f"separated below ~2.4 x (zmax1 = {zmax1/mm:.0f} mm); "
                     f"expect a rippled window")
    if fringe_pts < 2.5:
        notes.append(f"only {fringe_pts:.1f} radial pts/fringe — raise N")

    # axicon 1 -> L12 -> axicon 2
    E0 = np.exp(-(q.r / w0) ** 2)
    P_in = q.power(E0)
    y1 = q.forward(E0 * np.exp(-1j * kr1 * q.r))
    E_at2 = q.backward(propagate_spectrum(y1, q.kr, k, p["L12"]))
    y2 = q.forward(E_at2 * np.exp(-1j * kr2 * q.r))
    z2 = np.linspace(0.0, max(2 * p["L23"], 50 * mm), 32)
    pair = dict(q=q, z2=z2, Y2=propagate_spectrum(y2, q.kr, k, z2),
                R_ring=R_ring, w0=w0)

    # annulus at axicon 3, then fold
    E_in, R_a, Delta = ring_at(pair, p["L23"])
    if not np.isfinite(Delta):
        Delta = w0
        notes.append("ring FWHM undefined at axicon 3 (unseparated annulus) — "
                     "using Delta := w0 for the propagation span")
    I_ring = np.abs(E_in) ** 2
    y3 = q.forward(E_in * np.exp(-1j * kr3 * q.r))

    th_net = geo["th_end"] if geo["ok"] else kr3 / k
    z_wall = (q.R + R_a) / th_net           # domain-wall aliasing horizon
    z_hi = min(1.1 * geo.get("z_clear", 2 * (R_a + Delta) / th_net), 0.95 * z_wall)
    z3 = np.linspace(1e-3, z_hi, p["Nz3"])
    Y3 = propagate_spectrum(y3, q.kr, k, z3)
    onax = q.onax(Y3)

    # full-domain field map (native grid) -> encircled-power envelopes
    E_rz = (q.T @ Y3.T) * q.J1[:, None]     # backward transform, all z at once
    wgt = (2.0 * q.R**2 / q.jN1**2) / q.J1**2
    P_cum = np.cumsum(np.abs(E_rz) ** 2 * wgt[:, None], axis=0)
    frac = P_cum / P_cum[-1]
    r_out = np.array([np.interp(p["enc_out"], frac[:, i], q.r)
                      for i in range(len(z3))])
    r_in = np.array([np.interp(p["enc_in"], frac[:, i], q.r)
                     for i in range(len(z3))])

    # fine map near the axis for core metrics
    r_fine = np.linspace(0.0, max(3 * p["r_near"], 450 * UM), 261)
    M = np.abs(Y3 @ q.eval_matrix(r_fine)) ** 2          # (Nz3, Nr)
    I0 = M[:, 0]
    band = r_fine <= p["r_near"]
    I_near = (M[:, band] * r_fine[band]).sum(1) / r_fine[band].sum()
    fwhm_r = np.full(len(z3), np.nan)
    for i in range(len(z3)):
        below = np.where(M[i] < 0.5 * I0[i])[0]
        if len(below):
            j = below[0]
            fwhm_r[i] = np.interp(0.5 * I0[i], [M[i, j], M[i, j - 1]],
                                  [r_fine[j], r_fine[j - 1]])

    # on-axis window (FWHM of I(0,z), ignoring the pre-zone foot)
    zone_lo = (R_a - Delta / 2) / th_net
    on_m = np.where(z3 > 0.6 * zone_lo, onax, 0.0)
    i_pk = int(np.argmax(on_m))
    half = 0.5 * on_m[i_pk]
    lo = np.where(on_m[:i_pk] < half)[0]
    hi = np.where(onax[i_pk:] < half)[0]
    z_lo = z3[lo[-1]] if len(lo) else z3[0]
    z_hi_w = z3[i_pk + hi[0]] if len(hi) else z3[-1]
    win = (z3 >= z_lo) & (z3 <= z_hi_w)

    # envelope crossings of the half-inch clearance
    fit_ok = 2 * r_out <= HALF_INCH
    z_fit = z3[fit_ok][0] if fit_ok.any() else np.nan
    clear_ok = (z3 > z_hi_w) & (2 * r_in >= HALF_INCH)
    z_clear, clear_extrap = (z3[clear_ok][0], False) if clear_ok.any() else \
        (z_hi_w + (HALF_INCH / 2 - np.interp(z_hi_w, z3, r_in)) / th_net, True)

    # C-metric within r_near at the window peak
    row = M[i_pk, band]
    C_near = (row.max() - row.min()) / row.mean() if row.mean() > 0 else np.nan

    return dict(q=q, z3=z3, onax=onax, I_near=I_near, I0=I0, M=M, r_fine=r_fine,
                r_out=r_out, r_in=r_in, frac=frac, R_a=R_a, Delta=Delta,
                z_lo=z_lo, z_hi=z_hi_w, i_pk=i_pk, win=win, fwhm_r=fwhm_r,
                z_fit=z_fit, z_clear=z_clear, clear_extrap=clear_extrap,
                C_near=C_near, z_wall=z_wall, th_net=th_net, zmax1=zmax1,
                notes=notes, P_frac3=q.power_spec(y3), P_in=P_in,
                onax_th=2 * np.pi * kr3 ** 2 / k * z3
                        * np.interp(kr3 / k * z3, q.r, I_ring))


def sampling_table(p, N_list=(2048, 4096, 8192, 16384)):
    """Grid adequacy for the CURRENT optical configuration, per candidate N.

    Cheap: needs only the Bessel zeros, never the NxN transform matrix.
    Criteria are the ones asserted inside simulator.qdht_axicon —
    k_r,max > 1.5 k_r,axicon (the grid can represent the axicon phase ramp)
    and > 2.5 radial samples per axicon fringe.
    """
    k = 2 * np.pi / p["lam"]
    w0 = p["D"] / 2
    kr = [axicon_kr(a, n=p["n"], k=k) for a in (p["a1"], p["a2"], p["a3"])]
    kr_max_ax = max(kr)                       # finest fringe in the system
    R = (kr[0] / k) * p["L12"] + p["pad_w0"] * w0
    fringe = 2 * np.pi / kr_max_ax
    rows = []
    for N in N_list:
        jz = jn_zeros(0, N + 1)
        dr = (jz[1] - jz[0]) * R / jz[N]
        band = (jz[N] / R) / kr_max_ax
        ppf = fringe / dr
        rows.append({
            "N": N,
            "dr (µm)": round(float(dr / UM), 2),
            "k_r,max / k_r,axicon": round(float(band), 2),
            "pts / fringe": round(float(ppf), 2),
            "Memory": f"{8 * N * N / 1e6:.0f} MB" if N < 16384
                        else f"{8 * N * N / 1e9:.2f} GB",
            "verdict": ("OK" if (band > 1.5 and ppf > 2.5) else
                        "MARGINAL" if (band > 1.0 and ppf > 2.5) else
                        "UNDER-RESOLVED"),
        })
    return rows, R, kr_max_ax


def convergence_check(p, geo, N_lo, N_hi):
    """Rerun the current design at two grid sizes and compare the on-axis curve
    on a common z grid. Grid-independence is the real artifact test."""
    a = run_wave(p | dict(N=N_lo), geo)
    b = run_wave(p | dict(N=N_hi), geo)
    z = np.linspace(max(a["z3"][0], b["z3"][0]), min(a["z3"][-1], b["z3"][-1]),
                    800)
    ca = np.interp(z, a["z3"], a["onax"])
    cb = np.interp(z, b["z3"], b["onax"])
    return dict(N_lo=N_lo, N_hi=N_hi,
                rel_l2=float(np.linalg.norm(ca - cb) / np.linalg.norm(cb)),
                corr=float(np.corrcoef(ca, cb)[0, 1]),
                pk_lo=float(z[np.argmax(ca)]), pk_hi=float(z[np.argmax(cb)]),
                amp_ratio=float(ca.max() / cb.max()))


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Three-axicon Gauss-Bessel Beam Simulator", layout="wide")
    st.title("Three-axicon Gauss-Bessel Beam")
    st.info("Ray diagrams automatically update with parameter changes. Wave-optics results are only updated when you hit **Run wave optics**.")
    with st.sidebar:
        st.header("Design Parameters")
        st.caption("Design parameters are used for both the geometric and wave-optics models.")
        lam_nm = st.selectbox("Wavelength (nm)", (650.0, 698.0), index=0)
        D_mm = st.number_input("Input 1/e² diameter D (mm)", 1.0, 25.0, 9.50, 0.25)
        a1 = st.number_input("Axicon 1 base angle (deg)", 0.01, 20.0, 5.0, 0.1)
        a2 = st.number_input("Axicon 2 base angle (deg)", 0.01, 20.0, 5.0, 0.1)
        a3 = st.number_input("Axicon 3 base angle (deg)", 0.01, 20.0, 0.5, 0.05)
        L12_mm = st.number_input("L₁₂ (mm)", 10.0, 3000.0, 300.0, 10.0)
        L23_mm = st.number_input("L₂₃ (mm)", 1.0, 1000.0, 50.0, 5.0)
        with st.expander("QDHT & wave optic parameters"):
            N = st.selectbox("QDHT N", (2048, 4096, 8192, 16384), index=1,
                             help="Radial grid size. 16384 needs ~2.1 GB and "
                                  "runs minutes per evaluation — see the "
                                  "Numerics & validation panel.")
            Nz3 = st.slider("z-planes after axicon 3 (grid along propagation axis)", 100, 600, 260, 20)
            pad_w0 = st.slider("Domain pad (x w_0)", 1.5, 6.0, 2.5, 0.5)
        with st.expander("Definitions"):
            n_glass = st.number_input("Axicon index n", 1.3, 1.6, 1.4585, 0.0005,
                                      format="%.4f")
            enc_out = st.slider("Outer envelope: encircled power", 0.90, 0.999,
                                0.99, 0.001)
            enc_in = st.slider("Mirror #2 max power (sets void zone)", 0.001, 0.10, 0.01, 0.001)
            r_near_um = st.slider("Near-axis radius (µm)", 25, 400, 150, 25)
        run = st.button("Run wave optics", type="primary", use_container_width=True)

    p = dict(lam=lam_nm * 1e-9, D=D_mm * mm, a1=a1, a2=a2, a3=a3, n=n_glass,
             L12=L12_mm * mm, L23=L23_mm * mm, N=int(N), Nz3=int(Nz3),
             pad_w0=pad_w0, enc_out=enc_out, enc_in=enc_in, r_near=r_near_um * UM)

    geo = geometric(p)

    st.subheader("Geometric Model (Thin Lens Approximation)")
    st.caption("The geometric/ray-optics model is useful for exploring the parameter space and approximate boundaries of the Bessel region, but it does not capture the wave-optics effects that determine the actual on-axis intensity and radial FWHM (see Wave Optics section below).")
    if not geo["ok"]:
        st.error("Some rays never cross the axis with these angles — no complete "
                 "Bessel region. Check that β₁ − β₂ + β₃ nets out converging.")
    else:
        c1 = st.columns(3)
        c1[0].metric("Bessel range", f"{geo['z_end'] - geo['z_start']:.2f} m",
                    f"z = {geo['z_start']:.2f} – {geo['z_end']:.2f} m",
                    delta_color="off")
        c1[1].metric("Central Bessel lobe diameter", f"{geo['core_d_end']/UM:.0f} um")
        c1[2].metric("Annulus at axicon 3",
                    f"{geo['rho_in']/mm:.1f}–{geo['rho_out']/mm:.1f} mm")

        c2 = st.columns(3)
        c2[0].metric("1/2'' mirror 1 placement",
                    "—" if np.isnan(geo["z_fit"]) else f"{geo['z_fit']:.2f} m")
        c2[1].metric("1/2'' mirror 2 placement", f"{geo['z_clear']:.2f} m",
                    f"{geo['z_clear'] - geo['z_end']:.2f} m past range end",
                    delta_color="off")
        c2[2].metric("Final cone angle $\\theta$", f"{geo['th_end']/MRAD:.2f} mrad")
    st.warning("Diagram objects are NOT to scale. axes are in meters (z) and millimeters (r).")
    st.pyplot(ray_diagram(p, geo), use_container_width=True)
    with st.expander("How the geometric Bessel range is determined"):
        st.markdown(
            "**Meridional rays** (rays that pass through the optical axis) are launched at input heights $h \\in (0, D/2]$, where D is the input diameter of the Gaussian beam. D is determined by the input beam's 1/e² radius, enclosing 86.5 % of the power. See `beam_gaussian_fit_analysis_2026_07_24.ipynb` for determination of D. "
            "Each axicon bends a ray toward the axis by Snell's law:"
            "$\\beta = \\arcsin(n\\sin\\alpha) - \\alpha$, on a single plane (thin lens approximation)."
            " After axicon 3 a ray at height $\\rho$ "
            "with inward slope $\\theta$ crosses the axis at $z = \\rho/\\theta$; "
            "the axis is 'lit' (central Bessel lobe) only where rays are "
            "crossing. The **Bessel range** is the span between the first and "
            "last crossings, $[\\rho_{in}/\\theta,\\ \\rho_{out}/\\theta]$ — "
            "for a collimated annulus this is just (annulus width)/θ = "
            "$(D/2)/\\theta$.")

    key = tuple(sorted(p.items()))
    if run:
        with st.spinner("QDHT: pair → annulus → axicon 3 → propagation…"):
            st.session_state["wave"] = (key, run_wave(p, geo))
    have = st.session_state.get("wave")
    if have is None:
        st.stop()
    elif have[0] != key:
        st.warning("Parameters changed since the last run — wave-optics results below "
                   "are for the previous settings. Hit **Run wave optics**.")
    w = have[1]

    st.subheader("Wave optics")
    st.caption("The wave-optics model uses Quasi-Discrete Hankel Transform (QDHT) [ref Yu, 1998] to propagate the Gaussian input through the axicon pair and then through the third axicon, producing on-axis and near-axis intensity, radial FWHM, and power-based envelope metrics. We assume alpha1 == alpha2 in the simulator.")
    for msg in w["notes"]:
        st.warning(msg)

    c1 = st.columns(3)
    c1[0].metric("Bessel range (on-axis FWHM)", f"{w['z_hi'] - w['z_lo']:.2f} m",
                f"z = {w['z_lo']:.2f} – {w['z_hi']:.2f} m", delta_color="off")
    c1[1].metric("1/2'' mirror 1", "—" if np.isnan(w["z_fit"]) else
                f"{w['z_fit']:.2f} m")
    c1[2].metric("1/2'' mirror 2",
                f"{w['z_clear']:.2f} m" + (" *" if w["clear_extrap"] else ""),
                f"{w['z_clear'] - w['z_hi']:.2f} m past range end", delta_color="off")

    c2 = st.columns(3)

    c2[0].metric("Peak on-axis I/I_in", f"{w['onax'][w['i_pk']]:.2f}",
                f"at z = {w['z3'][w['i_pk']]:.2f} m", delta_color="off")
    c2[1].metric(f"Near-axis flatness C({int(p['r_near']/UM)})",
                f"{w['C_near']:.2f}", "at window peak", delta_color="off")
    fw = w["fwhm_r"][w["win"]]
    c2[2].metric("Radial FWHM in window",
                f"{np.nanmedian(fw)/UM:.0f} µm",
                f"{np.nanmin(fw)/UM:.0f}–{np.nanmax(fw)/UM:.0f} µm", delta_color="off")
    if w["clear_extrap"]:
        st.caption("\\* void-clear point is past the alias-safe domain "
                   f"(z_wall = {w['z_wall']:.1f} m) — extrapolated geometrically.")

    z3 = w["z3"]
    colA, colB = st.columns(2)
    with colA:
        fig, ax = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
        ax.plot(z3, w["onax"], lw=1.2, label="on-axis")
        ax.plot(z3, w["I_near"], lw=1.0, ls="--",
                label=f"near-axis mean (r ≤ {int(p['r_near']/UM)} µm)")
        ax.axvspan(w["z_lo"], w["z_hi"], color="tab:orange", alpha=0.15)
        ax.set_xlabel("z after axicon 3 (m)")
        ax.set_ylabel("I / I_in")
        ax.legend(frameon=False, fontsize=8)
        ax.minorticks_on(); ax.grid(alpha=0.3)
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(6.4, 3.0), constrained_layout=True)
        ax.plot(z3[w["win"]], w["fwhm_r"][w["win"]] / UM, lw=1.2)
        ax.axhline(X_HALF * p["lam"] / (2 * np.pi * w["th_net"]) / UM,
                   color="0.6", ls=":", lw=1, label="ideal J₀ half-max radius")
        ax.set_xlabel("z after axicon 3 (m)")
        ax.set_ylabel("radial FWHM (µm)")
        ax.legend(frameon=False, fontsize=8)
        ax.minorticks_on(); ax.grid(alpha=0.3)
        st.pyplot(fig)

    with colB:
        fig, ax = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
        ax.plot(z3, 2 * w["r_out"] / mm, lw=1.2,
                label=f"outer Ø ({p['enc_out']:.0%} power)")
        ax.plot(z3, 2 * w["r_in"] / mm, lw=1.2,
                label=f"void Ø ({p['enc_in']:.1%} inside)")
        ax.axhline(HALF_INCH / mm, color="crimson", lw=1, ls="--", label='½″')
        for zz, lab in ((w["z_fit"], "fit"), (w["z_clear"], "clear")):
            if np.isfinite(zz) and zz <= z3[-1]:
                ax.axvline(zz, color="0.5", lw=0.8, ls=":")
        ax.axvspan(w["z_lo"], w["z_hi"], color="tab:orange", alpha=0.15)
        ax.set_xlabel("z after axicon 3 (m)")
        ax.set_ylabel("diameter (mm)")
        ax.legend(frameon=False, fontsize=8)
        ax.minorticks_on(); ax.grid(alpha=0.3)
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(6.4, 3.0), constrained_layout=True)
        im = ax.imshow(w["M"].T, origin="lower", aspect="auto", cmap="inferno",
                       extent=[z3[0], z3[-1], 0, w["r_fine"][-1] / UM],
                       norm=PowerNorm(0.5, vmax=max(w["M"].max(), 1e-30)))
        ax.axhline(p["r_near"] / UM, color="w", ls="--", lw=0.7)
        ax.set_xlabel("z after axicon 3 (m)")
        ax.set_ylabel("r (µm)")
        plt.colorbar(im, ax=ax, shrink=0.85, label="I/I_in (γ=0.5)")
        st.pyplot(fig)

        

    st.caption(f"Domain: R = {w['q'].R/mm:.1f} mm, N = {p['N']}; alias-safe to "
               f"z_wall ≈ {w['z_wall']:.1f} m. Ring at axicon 3: R_a = "
               f"{w['R_a']/mm:.2f} mm, FWHM Δ = {w['Delta']/mm:.2f} mm. "
               f"zmax₁ = {w['zmax1']/mm:.0f} mm.")

    with st.expander("How the wave-optics Bessel range is determined"):
        st.markdown(
            "The wave-optics **Bessel range** is the full width at half maximum "
            "of the on-axis intensity curve $I(0,z)$ plotted below, shaeded in orange.")

    with st.expander(f"What near-axis flatness "
                     f"C({int(p['r_near']/UM)}) means"):
        st.markdown(
            f"At the z-plane where the on-axis intensity peaks, the radial "
            f"profile is taken from the axis out to "
            f"r = {int(p['r_near']/UM)} µm (the **Near-axis radius** knob — set "
            f"it to the radius of the atom cloud) and reduced to "
            f"$C = (I_{{max}} - I_{{min}})/I_{{mean}}$.\n\n"
            f"The scale reads directly as a uniformity spec: $C = 0$ is a "
            f"perfect flat-top, $C = 0.2$ is $\\pm$10 % uniform, and "
            f"$C = 1.0$ is $\\pm$50 %. Values of several units mean the disc is "
            f"not a plateau at all — at the as-built settings "
            f"r $\\leq$ 150 µm spans the central lobe, the first null "
            f"($\\approx$ 62 µm), the whole first ring and the second null "
            f"($\\approx$ 143 µm), i.e. a full oscillation of $J_0^2$. "
            f"Shrinking the radius shows where the usable region really ends: "
            f"C = 0.44 at 25 µm, 1.54 at 50 µm, 3.01 at 100 µm, 4.11 at 150 µm.\n\n"
            f"Note $I_{{mean}}$ is a simple mean over evenly spaced radial "
            f"samples, not an area-weighted average (area-weighting would raise "
            f"C by roughly 2.3x) — this matches the C150 convention used in "
            f"`theory_vs_manual_vs_gantry_2026_07_23.ipynb`, so the two are "
            f"directly comparable.")

    with st.expander("What Peak on-axis I/I_in is a ratio of"):
        st.markdown(
            "$I_{in}$ is the **peak intensity of the input Gaussian** at axicon "
            "1: the model launches $E = \\exp(-(r/w_0)^2)$, so $|E|^2 = 1$ on "
            "the axis there. The metric is therefore $I(0,z)/I_{peak,in}$ — a "
            "ratio of intensity at a point, not a percentage and not a ratio of "
            "power. A value of 370 means 370x, i.e. the axis in the Bessel zone "
            "is 370 times brighter than the brightest point of the input beam.\n\n")

    with st.expander("Is the sampling grid adequate for this configuration?"):
        rows, R_dom, kr_ax = sampling_table(p)
        st.markdown(
            f"Domain radius for this configuration is "
            f"R = {R_dom/mm:.1f} mm and the finest fringe belongs to the "
            f"steepest axicon ($k_r$ = {kr_ax:.3g} rad/m, period "
            f"{2*np.pi/kr_ax/UM:.1f} µm). The QDHT samples r at N zeros of "
            f"$J_0$ scaled to R, so N sets both the radial resolution and the "
            f"highest radial spatial frequency of the sampling grid. "
            f"If N is too small, the grid cannot represent the phase ramp of the axicon, and the result is non-physical due to aliasing. "
            f"If N is too large, the grid may be overkill and waste computational resources."
        )
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.caption("OK = both criteria met (k_r,max > 1.5 k_r,axicon and "
                   "> 2.5 pts/fringe, as asserted in simulator/qdht_axicon.py). "
                   "Memory is the NxN transform matrix")

        st.markdown("**Physics self-checks for the run above**")
        ok = "✅"; bad = "⚠️"
        pratio = w["P_frac3"] / w["P_in"]
        i_edge = np.searchsorted(w["q"].r, 0.9 * w["q"].R)
        edge = float(np.max(1.0 - w["frac"][min(i_edge, len(w["q"].r) - 1), :]))
        m = w["onax"] > 0.05 * w["onax"][w["i_pk"]]     # the whole lit region
        th_pk = w["onax_th"][w["i_pk"]]
        ratio = th_pk / w["onax"][w["i_pk"]] if w["onax"][w["i_pk"]] > 0 else np.nan
        corr = float(np.corrcoef(w["onax_th"][m], w["onax"][m])[0, 1]) \
            if m.sum() > 3 else np.nan
        st.markdown(
            f"- {ok if abs(pratio-1) < 1e-4 else bad} **Power conservation** "
            f"$P_{{out}}/P_{{in}}$ = {pratio:.6f}.\n"
            f"- {ok if edge < 1e-3 else bad} **Domain-wall leakage** "
            f"{100*edge:.3f} % of the power lies beyond 0.9 R at the furthest plane."
            f"Light reaching the domain wall is an artifact of the finite domain (R) and does not represent the physical system. z_wall = {w['z_wall']:.1f} m; the model stops at {w['z3'][-1]:.1f} m (0.95 z_wall).\n"
            f"- {ok if (np.isfinite(corr) and corr > 0.95) else bad} "
            f"**Analytic cross-check**: the closed-form Bessel-Gauss "
            f"prediction $2\\pi (k_{{r3}}^2/k)\\, z\\, I_{{ring}}(\\theta_3 z)$ "
            f"gives a peak {ratio:.3f}x the QDHT value with correlation "
            f"{corr:.4f} across the window. This uses different physics from QDHT propagation, so agreement is evidence the QDHT result is not under-sampled.\n"
            f"- {ok} **Alias-safe horizon** z_wall = {w['z_wall']:.1f} m; the "
            f"model stops at {w['z3'][-1]:.1f} m (0.95 z_wall).")

        st.markdown("**Grid-independence test** Rerun the "
                    "same design on a coarser and a finer grid. If the on-axis "
                    "curve does not move, N is not shaping the answer.")
        opts = [n for n in (1024, 2048, 4096, 8192, 16384) if n != p["N"]]
        finer = [n for n in opts if n > p["N"]]          # a FINER grid is the
        default = opts.index(finer[0]) if finer else len(opts) - 1   # real test
        cc1, cc2 = st.columns([2, 1])
        N_other = cc1.selectbox("Compare current N = "
                                f"{p['N']} against", opts,
                                index=default, key="conv_N")
        if cc2.button("Run test", use_container_width=True, key="conv_btn"):
            if N_other >= 16384 or p["N"] >= 16384:
                st.warning("N = 16384 needs ~2.1 GB and runs for minutes — "
                           "this will be slow.")
            with st.spinner(f"QDHT at N = {p['N']} and {N_other}…"):
                st.session_state["conv"] = convergence_check(
                    p, geo, min(p["N"], N_other), max(p["N"], N_other))
        cv = st.session_state.get("conv")
        if cv:
            good = cv["rel_l2"] < 0.02
            st.markdown(
                f"{'✅' if good else '⚠️'} N = {cv['N_lo']} vs {cv['N_hi']}: "
                f"relative L2 difference of the on-axis curve "
                f"**{100*cv['rel_l2']:.1f} %**, correlation {cv['corr']:.4f}, "
                f"peak amplitude ratio {cv['amp_ratio']:.3f}, peak position "
                f"{cv['pk_lo']:.2f} m vs {cv['pk_hi']:.2f} m."
                + ("  Converged — the answer is grid-independent."
                   if good else
                   "  NOT converged — treat these numbers as qualitative and "
                   "raise N, or check whether the design sits in the "
                   "unseparated-annulus regime, which is intrinsically hard to "
                   "resolve."))

if __name__ == "__main__":
    main()
