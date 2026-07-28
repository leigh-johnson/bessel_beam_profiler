"""
Axicon alignment on the FluidNC gantry: patrol the annulus after
axicon 3 and turn what the camera sees into live alignment numbers.

The annulus (~9.5 mm and up) is wider than the BFS sensor (~7.1 x 5.3 mm),
so no single frame sees the whole ring. Instead the gantry PATROLS a set
of stations around the current ring estimate, one frame per station, and
every lap ("cycle") refits the ring from all stations at once:

    1. Bootstrap: the auto-scan find-beam sweep locates structured light,
       exposure is calibrated there (both reused from auto_scan), then two
       vertical survey columns measure ring chords -> center + radius.
    2. Patrol cycle: N stations spaced around the ring estimate; each
       frame's lit pixels are mapped to MACHINE coordinates (same
       orientation/scale conventions as composite.py, verified on
       hardware 2026-07-22) and collapsed to ring-locus points.
    3. Fits and metrics per cycle: circle fit (Kasa) and ellipse fit
       (Halir-Flusser) -> center offset vs a reference, roundness
       (minor/major axis ratio + major-axis angle), azimuthal intensity
       uniformity (the classic signature of input-beam decenter on an
       axicon), and optionally the beam-axis tilt from ring centers
       measured at two machine-Y planes.

Coordinate convention (same one frame as auto_scan): X horizontal
transverse, Y beam propagation, Z vertical; the ring lives in the X-Z
plane. All angles are measured in the machine X-Z frame: 0 deg = +X,
90 deg = +Z (up), counterclockwise as seen looking down the beam.

The session is UI-agnostic: run() reports through callbacks, and
align_preview.py renders them in a live matplotlib window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional
import json
import logging
import math
import time

import numpy as np

from auto_scan import AutoScanConfig, AutoScanSession, _grab_frame
from calibration import ExposureCalibrationConfig
from composite import CompositeConfig, mean_pool, orient
from coordinates import Bounds3D

logger = logging.getLogger(__name__)


class AlignError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignConfig:
    # Machine-Y plane(s) to patrol. MachineY2_mm enables the two-plane
    # pointing-tilt measurement: cycles alternate between the planes and
    # the tilt is the ring-center difference over the beam-path distance.
    MachineY_mm: float = 20.0
    MachineY2_mm: Optional[float] = None

    # X of the find-beam sweep / survey columns. None = center of the
    # machine X envelope. Point it somewhere that crosses the ring.
    ProbeX_mm: Optional[float] = None

    # +1 if machine +Y points downstream, -1 toward the optic (rig
    # default, verified 2026-07-22). Only the SIGN of the two-plane tilt
    # depends on this.
    BeamDirectionSign: int = -1

    # Patrol stations per cycle. 8 stations x ~5.3 mm of arc per frame
    # covers a ~14 mm ring with margin; raise for bigger rings.
    Stations: int = 8

    # "ring": stations on the fitted ring only (fastest; a thin annulus
    # has nothing inside). "disk": ALSO image the interior — half the
    # station count again on a half-radius ring, plus one frame at the
    # center — so the composite has no blind spot in the middle. Fill
    # frames are display-only: they are EXCLUDED from the ring fit so
    # interior light (diffraction fringes) cannot pollute the geometry.
    CoverMode: str = "ring"

    # Optional prior on the ring size (sanity bound only; the chord
    # bootstrap measures the actual radius).
    RingDiameter_mm: Optional[float] = None

    # Chord-bootstrap survey columns: X offset between the two columns.
    SurveyDX_mm: float = 5.0

    # Ring-estimate update clamp per cycle (center AND radius), so one
    # bad fit cannot fling the patrol off the beam.
    MaxRingShift_mm: float = 3.0

    # Signal test per frame (no background frames in alignment mode:
    # the threshold is frame-median-referenced instead).
    SignalMargin_counts: float = 8.0
    MinSignalPixels: int = 30

    # Ring-locus extraction: lit pixels are collapsed to one
    # intensity-weighted radius per angular bin.
    AngleBin_deg: float = 2.0

    # Image geometry — MUST match composite.py's verified conventions.
    PixelSize_um: float = 3.45
    Downsample: int = 8
    FlipX: bool = True
    FlipZ: bool = True
    Transpose: bool = False

    # Cycle-level exposure servo (cheap, between full recalibrations):
    # halve on saturation, double when the brightest station is dim.
    Saturation: ExposureCalibrationConfig = ExposureCalibrationConfig()
    DimFraction: float = 0.25
    # Optional hard exposure ceiling (us). With background-referenced
    # thresholds a dim-but-detected ring aligns just as well, so capping
    # exposure trades unneeded dynamic range for lap speed on dim beams.
    MaxExposure_us: Optional[float] = None

    # Consecutive signal-less cycles before re-running the find-beam
    # bootstrap from scratch.
    LostCyclesBeforeRefind: int = 2

    # Minimum seconds per cycle (0 = free-running).
    CycleInterval_s: float = 0.0

    def mm_per_px(self) -> float:
        return self.PixelSize_um / 1000.0 * self.Downsample

    def composite_config(self) -> CompositeConfig:
        return CompositeConfig(
            PixelSize_um=self.PixelSize_um,
            Downsample=self.Downsample,
            FlipX=self.FlipX,
            FlipZ=self.FlipZ,
            Transpose=self.Transpose,
        )


# ---------------------------------------------------------------------------
# Pure geometry: pixels -> machine coordinates -> ring locus -> fits
# ---------------------------------------------------------------------------


def frame_axes_mm(
    shape: tuple[int, int], x_center_mm: float, z_center_mm: float, mm_per_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Machine coordinates of an ORIENTED frame's pixel centers (cols -> +X,
    row 0 -> +Z), for a frame whose center is at the commanded machine
    position. Same placement math as composite.composite_slice.
    Returns (x_of_col, z_of_row).
    """

    rows, cols = shape
    width_mm = cols * mm_per_px
    height_mm = rows * mm_per_px

    x = x_center_mm - width_mm / 2.0 + (np.arange(cols) + 0.5) * mm_per_px
    z = z_center_mm + height_mm / 2.0 - (np.arange(rows) + 0.5) * mm_per_px
    return x, z


def frame_signal_threshold(
    frame: np.ndarray,
    config: AlignConfig,
    background_level: Optional[float] = None,
) -> float:
    """
    Signal threshold for one frame.

    With a measured off-axis background level (captured at bootstrap),
    the threshold is simply background + margin — correct even when a
    BROAD beam fills the entire frame.

    Without one, fall back to frame-median-referencing. CAVEAT
    (hardware-observed 2026-07-27 on the wide axicon-2 band): when the
    beam fills the frame the median IS signal and dim arcs get
    thresholded away — the fallback is only sound for beams that cover
    a small fraction of each frame.
    """

    if background_level is not None:
        return background_level + config.SignalMargin_counts

    median = float(np.median(frame))
    peak = float(frame.max())
    return median + max(config.SignalMargin_counts, 0.2 * (peak - median))


@dataclass(frozen=True)
class ArcExtraction:
    # Ring-locus points in machine mm, one per lit angular bin: (x, z).
    Points_mm: np.ndarray
    # Bin angles (deg, machine X-Z frame) and their peak intensities.
    Angles_deg: np.ndarray
    Intensities: np.ndarray
    LitPixels: int
    Peak: float
    # Mean radial FWHM of the ring across this frame's angular bins
    # (2.355 x the intensity-weighted radial std per bin).
    WidthFWHM_mm: float = 0.0


def extract_arc(
    frame: np.ndarray,
    x_center_mm: float,
    z_center_mm: float,
    ring_center_guess: tuple[float, float],
    config: AlignConfig,
    background_level: Optional[float] = None,
) -> Optional[ArcExtraction]:
    """
    Collapse one oriented, downsampled frame to ring-locus points: lit
    pixels are binned by angle around the current ring-center guess and
    each bin contributes one intensity-weighted mean-radius point.
    Returns None when the frame has no usable signal.
    """

    threshold = frame_signal_threshold(frame, config, background_level)
    lit = frame > threshold
    lit_count = int(lit.sum())

    if lit_count < config.MinSignalPixels:
        return None

    x_axis, z_axis = frame_axes_mm(
        frame.shape, x_center_mm, z_center_mm, config.mm_per_px()
    )
    rows, cols = np.nonzero(lit)
    x = x_axis[cols]
    z = z_axis[rows]
    baseline = (
        background_level
        if background_level is not None
        else float(np.median(frame))
    )
    weights = frame[rows, cols].astype(np.float64) - baseline
    weights = np.clip(weights, 1e-6, None)

    cx, cz = ring_center_guess
    angles = np.degrees(np.arctan2(z - cz, x - cx)) % 360.0
    radii = np.hypot(x - cx, z - cz)

    bin_ids = (angles // config.AngleBin_deg).astype(int)
    n_bins = int(math.ceil(360.0 / config.AngleBin_deg))

    weight_sum = np.bincount(bin_ids, weights=weights, minlength=n_bins)
    radius_sum = np.bincount(bin_ids, weights=weights * radii, minlength=n_bins)
    radius_sq_sum = np.bincount(
        bin_ids, weights=weights * radii * radii, minlength=n_bins
    )

    # Ring brightness per bin = the bin's PEAK background-subtracted
    # value: summed weights would scale with how much of the bin the
    # frame happens to cover, biasing the azimuthal uniformity readout.
    peak_per_bin = np.zeros(n_bins)
    np.maximum.at(peak_per_bin, bin_ids, weights)

    occupied = weight_sum > 0
    bin_angles = (np.arange(n_bins) + 0.5) * config.AngleBin_deg
    mean_radii = np.zeros(n_bins)
    mean_radii[occupied] = radius_sum[occupied] / weight_sum[occupied]

    theta = np.radians(bin_angles[occupied])
    points = np.column_stack(
        (
            cx + mean_radii[occupied] * np.cos(theta),
            cz + mean_radii[occupied] * np.sin(theta),
        )
    )

    # Radial width: intensity-weighted radial variance per bin -> FWHM
    # (for a Gaussian ring profile FWHM = 2.355 * weighted radial std).
    with np.errstate(invalid="ignore"):
        variance = (
            radius_sq_sum[occupied] / weight_sum[occupied]
            - mean_radii[occupied] ** 2
        )
    variance = np.clip(variance, 0.0, None)
    width_fwhm = float(2.355 * np.mean(np.sqrt(variance)))

    return ArcExtraction(
        Points_mm=points,
        Angles_deg=bin_angles[occupied],
        Intensities=peak_per_bin[occupied],
        LitPixels=lit_count,
        Peak=float(frame.max()),
        WidthFWHM_mm=width_fwhm,
    )


@dataclass(frozen=True)
class CircleFit:
    CenterX_mm: float
    CenterZ_mm: float
    Radius_mm: float
    RMS_mm: float
    NPoints: int


def fit_circle(points: np.ndarray) -> Optional[CircleFit]:
    """Kasa algebraic circle fit: linear least squares, no iteration."""

    if points.shape[0] < 3:
        return None

    x = points[:, 0]
    z = points[:, 1]
    A = np.column_stack((2.0 * x, 2.0 * z, np.ones_like(x)))
    b = x * x + z * z

    try:
        solution, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError as ex:
        logger.warning(f"Circle fit failed (singular system): {ex}")
        return None

    cx, cz, c = solution
    r_squared = c + cx * cx + cz * cz

    if not np.isfinite(r_squared) or r_squared <= 0:
        logger.warning("Circle fit produced a non-positive radius; rejecting.")
        return None

    radius = float(np.sqrt(r_squared))
    residuals = np.hypot(x - cx, z - cz) - radius
    return CircleFit(
        CenterX_mm=float(cx),
        CenterZ_mm=float(cz),
        Radius_mm=radius,
        RMS_mm=float(np.sqrt(np.mean(residuals**2))),
        NPoints=int(points.shape[0]),
    )


def fit_center_fixed_radius(
    points: np.ndarray,
    radius: float,
    initial_center: tuple[float, float],
    iterations: int = 15,
) -> Optional[tuple[float, float, float]]:
    """
    Center-only circle fit with the radius held fixed — the well-posed
    version of the problem for a SINGLE frame's partial arc, where a
    free-radius fit is hopelessly ill-conditioned. Radius comes from
    the last full orbit. Gauss-Seidel style: each point votes for a
    center at distance `radius` inward along its own radial direction.
    Returns (cx, cz, rms) or None.
    """

    if points.shape[0] < 2 or radius <= 0:
        return None

    center = np.array(initial_center, dtype=np.float64)

    # Gauss-Newton on sum((|p - c| - r)^2): residual f = d - r has
    # gradient -u (unit radial vector), so solve (U^T U) delta = U^T f
    # each step. (Simple "project each point inward and average" is
    # BIASED on short arcs — verified by test.)
    for _ in range(iterations):
        delta_points = points - center
        distance = np.hypot(delta_points[:, 0], delta_points[:, 1])

        keep = distance > 1e-9
        if keep.sum() < 2:
            return None

        u = delta_points[keep] / distance[keep, None]
        f = distance[keep] - radius

        try:
            step, *_ = np.linalg.lstsq(u, f, rcond=None)
        except np.linalg.LinAlgError as ex:
            logger.warning(f"Fixed-radius center fit failed: {ex}")
            return None

        center += step

        if np.hypot(step[0], step[1]) < 1e-9:
            break

    residuals = np.hypot(*(points - center).T) - radius
    return (
        float(center[0]),
        float(center[1]),
        float(np.sqrt(np.mean(residuals**2))),
    )


@dataclass(frozen=True)
class EllipseFit:
    CenterX_mm: float
    CenterZ_mm: float
    SemiMajor_mm: float
    SemiMinor_mm: float
    # Major-axis angle in the machine X-Z frame, degrees in [0, 180).
    MajorAxisAngle_deg: float

    @property
    def axis_ratio(self) -> float:
        return self.SemiMinor_mm / self.SemiMajor_mm


def fit_ellipse(points: np.ndarray) -> Optional[EllipseFit]:
    """
    Direct least-squares ellipse fit (Halir & Flusser's numerically
    stable variant of Fitzgibbon), then conic -> center/axes/angle.
    Returns None when the fit is degenerate or not an ellipse.
    """

    if points.shape[0] < 6:
        return None

    x = points[:, 0]
    z = points[:, 1]

    # Center the data for conditioning; un-shift the center afterwards.
    x0, z0 = float(x.mean()), float(z.mean())
    xc = x - x0
    zc = z - z0

    D1 = np.column_stack((xc * xc, xc * zc, zc * zc))
    D2 = np.column_stack((xc, zc, np.ones_like(xc)))
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2

    try:
        T = -np.linalg.solve(S3, S2.T)
        M = S1 + S2 @ T
        C_inv_M = np.array(
            [M[2] / 2.0, -M[1], M[0] / 2.0]
        )  # inv(constraint) @ M for constraint 4ac - b^2 = 1
        eigenvalues, eigenvectors = np.linalg.eig(C_inv_M)
    except np.linalg.LinAlgError as ex:
        logger.warning(f"Ellipse fit failed (singular system): {ex}")
        return None

    # The valid solution satisfies 4ac - b^2 > 0. eig can return complex
    # pairs for degenerate inputs; only real eigenvectors qualify.
    real_vectors = np.real(eigenvectors)
    condition = (
        4.0 * real_vectors[0] * real_vectors[2] - real_vectors[1] ** 2
    )
    valid = np.where(np.isreal(eigenvalues) & (condition > 0))[0]

    if valid.size == 0:
        return None

    a1 = np.real(eigenvectors[:, valid[0]])
    a2 = T @ a1
    A, B, C = a1
    D, E, F = a2

    denom = B * B - 4.0 * A * C
    if denom >= 0:  # not an ellipse
        return None

    cx = (2.0 * C * D - B * E) / denom
    cz = (2.0 * A * E - B * D) / denom

    # Conic value at the center; semi-axes from the quadratic-form
    # eigenvalues: axis_i = sqrt(-Fc / lambda_i).
    Fc = A * cx * cx + B * cx * cz + C * cz * cz + D * cx + E * cz + F
    quad = np.array([[A, B / 2.0], [B / 2.0, C]])
    lambdas, vectors = np.linalg.eigh(quad)

    with np.errstate(invalid="ignore", divide="ignore"):
        axes_squared = -Fc / lambdas

    if not np.all(np.isfinite(axes_squared)) or np.any(axes_squared <= 0):
        return None

    axes = np.sqrt(axes_squared)
    major_index = int(np.argmax(axes))
    major_vector = vectors[:, major_index]
    angle = math.degrees(math.atan2(major_vector[1], major_vector[0])) % 180.0

    return EllipseFit(
        CenterX_mm=float(cx + x0),
        CenterZ_mm=float(cz + z0),
        SemiMajor_mm=float(axes[major_index]),
        SemiMinor_mm=float(axes[1 - major_index]),
        MajorAxisAngle_deg=float(angle),
    )


@dataclass(frozen=True)
class UniformityMetrics:
    MinMaxRatio: float
    CoefficientOfVariation: float
    BrightestAngle_deg: float
    DimmestAngle_deg: float
    CoverageFraction: float


def azimuthal_uniformity(
    angles_deg: np.ndarray,
    intensities: np.ndarray,
    sector_deg: float = 15.0,
) -> Optional[UniformityMetrics]:
    """
    Ring brightness vs angle, from all stations' per-bin intensities.
    Overlapping stations are reconciled by taking the MAX per sector
    (sums would double-count overlap regions). Uniformity is computed
    over COVERED sectors only; CoverageFraction says how much of the
    ring that is.
    """

    if angles_deg.size == 0:
        return None

    n_sectors = int(round(360.0 / sector_deg))
    sector_ids = (angles_deg // sector_deg).astype(int) % n_sectors

    levels = np.zeros(n_sectors)
    covered = np.zeros(n_sectors, dtype=bool)
    np.maximum.at(levels, sector_ids, intensities)
    covered[sector_ids] = True

    values = levels[covered]
    if values.size == 0:
        return None

    centers = (np.arange(n_sectors) + 0.5) * sector_deg
    covered_centers = centers[covered]
    mean = float(values.mean())

    return UniformityMetrics(
        MinMaxRatio=float(values.min() / values.max()) if values.max() > 0 else 0.0,
        CoefficientOfVariation=float(values.std() / mean) if mean > 0 else 0.0,
        BrightestAngle_deg=float(covered_centers[int(np.argmax(values))]),
        DimmestAngle_deg=float(covered_centers[int(np.argmin(values))]),
        CoverageFraction=float(covered.mean()),
    )


def compose_canvas(
    frames: list[tuple[np.ndarray, float, float]], mm_per_px: float
) -> Optional[tuple[np.ndarray, tuple[float, float, float, float]]]:
    """
    Place oriented, downsampled frames at their machine positions on one
    canvas (overlaps averaged) — a lightweight in-memory sibling of
    composite.composite_slice for the live preview.
    Returns (canvas, extent) with extent = (x_min, x_max, z_min, z_max).
    """

    if not frames:
        return None

    rows, cols = frames[0][0].shape
    frame_w = cols * mm_per_px
    frame_h = rows * mm_per_px

    xs = [x for _, x, _ in frames]
    zs = [z for _, _, z in frames]
    x_left = min(xs) - frame_w / 2.0
    z_top = max(zs) + frame_h / 2.0

    canvas_cols = int(round((max(xs) - min(xs) + frame_w) / mm_per_px))
    canvas_rows = int(round((max(zs) - min(zs) + frame_h) / mm_per_px))

    total = np.zeros((canvas_rows, canvas_cols), dtype=np.float64)
    count = np.zeros((canvas_rows, canvas_cols), dtype=np.int32)

    for frame, x_mm, z_mm in frames:
        col0 = int(round((x_mm - frame_w / 2.0 - x_left) / mm_per_px))
        row0 = int(round((z_top - (z_mm + frame_h / 2.0)) / mm_per_px))
        total[row0:row0 + frame.shape[0], col0:col0 + frame.shape[1]] += frame
        count[row0:row0 + frame.shape[0], col0:col0 + frame.shape[1]] += 1

    # Un-imaged pixels are NaN, NOT zero: the preview renders them as a
    # neutral gray so coverage gaps between station frames don't read
    # as "the beam is dark here".
    covered = count > 0
    canvas = np.full_like(total, np.nan, dtype=np.float32)
    canvas[covered] = (total[covered] / count[covered]).astype(np.float32)

    extent = (
        x_left,
        x_left + canvas_cols * mm_per_px,
        z_top - canvas_rows * mm_per_px,
        z_top,
    )
    return canvas, extent


# ---------------------------------------------------------------------------
# Chord bootstrap: two survey columns -> ring center + radius
# ---------------------------------------------------------------------------


def column_crossing_intervals(
    column: list[tuple[np.ndarray, float]],
    config: AlignConfig,
    x_center_mm: float,
    background_level: Optional[float] = None,
) -> list[tuple[float, float]]:
    """
    Merge one survey column's frames into lit machine-Z intervals near
    the column's X. A vertical line through a ring crosses it twice, so
    a well-placed column yields two intervals (one if near-tangent).

    Only the central third of each frame's columns is examined so the
    intervals measure the crossing AT the column X, not wherever the
    ring happens to clip a frame corner.
    """

    lit_z: list[np.ndarray] = []

    for frame, z_center_mm in column:
        threshold = frame_signal_threshold(frame, config, background_level)
        rows, cols = frame.shape
        central = frame[:, cols // 3: cols - cols // 3]
        lit_rows = np.nonzero(
            (central > threshold).sum(axis=1) >= max(1, central.shape[1] // 20)
        )[0]

        if lit_rows.size == 0:
            continue

        _, z_axis = frame_axes_mm(
            frame.shape, x_center_mm, z_center_mm, config.mm_per_px()
        )
        lit_z.append(z_axis[lit_rows])

    if not lit_z:
        return []

    z_values = np.sort(np.concatenate(lit_z))

    # Merge into intervals: a gap larger than a few pixels ends one.
    gap = 5.0 * config.mm_per_px()
    intervals: list[tuple[float, float]] = []
    start = previous = z_values[0]

    for z in z_values[1:]:
        if z - previous > gap:
            intervals.append((start, previous))
            start = z
        previous = z
    intervals.append((start, previous))

    # Ignore slivers (single-pixel noise clusters).
    minimum_span = 2.0 * config.mm_per_px()
    return [(lo, hi) for lo, hi in intervals if hi - lo >= minimum_span]


def solve_ring_from_chords(
    x1: float, intervals1: list[tuple[float, float]],
    x2: float, intervals2: list[tuple[float, float]],
) -> Optional[tuple[float, float, float]]:
    """
    Ring (cx, cz, r) from lit intervals of two vertical survey columns.

    Each column's chord is the ENVELOPE of its lit intervals — from the
    lowest lit Z to the highest — which is a chord of the beam's outer
    boundary. This is the definition that survives BROAD annuli
    (hardware lesson 2026-07-27: a wide band with a small dark hole
    gives ONE lit interval on columns that miss the hole; the earlier
    "one interval = tangent, h=0" reading skewed the solve by ~7 mm).
    For a thin ring the envelope spans both crossings, so the same
    definition holds there too (r comes out at the outer edge; the
    first orbit fit refines it to the band centroid).

    For a column at x the half-chord h obeys h^2 = r^2 - (x - cx)^2 and
    the envelope midpoint is cz — two columns make cx a linear solve.
    """

    def chord(intervals) -> Optional[tuple[float, float]]:
        if not intervals:
            return None
        lo = min(iv[0] for iv in intervals)
        hi = max(iv[1] for iv in intervals)
        return (lo + hi) / 2.0, (hi - lo) / 2.0

    chord1 = chord(intervals1)
    chord2 = chord(intervals2)

    if chord1 is None or chord2 is None or abs(x2 - x1) < 1e-9:
        return None

    cz1, h1 = chord1
    cz2, h2 = chord2

    cx = (h2 * h2 - h1 * h1 + x2 * x2 - x1 * x1) / (2.0 * (x2 - x1))
    r_squared = h1 * h1 + (x1 - cx) ** 2

    if r_squared <= 0:
        return None

    # Longer envelope = closer to center = better-determined midpoint.
    weight1 = max(h1, 0.5)
    weight2 = max(h2, 0.5)
    cz = (cz1 * weight1 + cz2 * weight2) / (weight1 + weight2)
    return float(cx), float(cz), float(np.sqrt(r_squared))


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RingEstimate:
    CenterX_mm: float
    CenterZ_mm: float
    Radius_mm: float


@dataclass(frozen=True)
class StationSample:
    X_mm: float
    Z_mm: float
    HasSignal: bool
    Peak: float
    Skipped: bool = False  # outside the machine envelope
    # "ring" stations feed the fits; "fill" stations (disk cover mode)
    # only fill in the composite.
    Role: str = "ring"


@dataclass(frozen=True)
class TiltMetrics:
    TiltX_mrad: float
    TiltZ_mrad: float
    DeltaBeamPath_mm: float


@dataclass(frozen=True)
class CycleResult:
    Index: int
    MachineY_mm: float
    Estimate: RingEstimate
    Circle: Optional[CircleFit]
    Ellipse: Optional[EllipseFit]
    Uniformity: Optional[UniformityMetrics]
    # Offset of the fitted center vs the reference center (dx, dz), mm.
    Reference: Optional[tuple[float, float]]
    Offset_mm: Optional[tuple[float, float]]
    Tilt: Optional[TiltMetrics]
    Stations: list[StationSample]
    Points_mm: np.ndarray
    Exposure_us: Optional[float]
    Lost: bool
    Elapsed_s: float
    # Mean radial FWHM of the ring across stations that saw it.
    WidthFWHM_mm: Optional[float] = None

    def to_jsonable(self) -> dict[str, Any]:
        def plain(value):
            if isinstance(value, np.ndarray):
                return None  # bulky; the JSONL log keeps metrics only
            if isinstance(value, (np.floating, np.integer)):
                return value.item()
            if hasattr(value, "__dataclass_fields__"):
                return {
                    k: plain(getattr(value, k))
                    for k in value.__dataclass_fields__
                }
            if isinstance(value, (list, tuple)):
                return [plain(v) for v in value]
            return value

        payload = {
            k: plain(getattr(self, k)) for k in self.__dataclass_fields__
        }
        payload.pop("Points_mm")
        payload["NPoints"] = int(self.Points_mm.shape[0])
        return payload


class AxiconAlignSession:
    """
    Owns the patrol loop. Wraps an AutoScanSession for everything the
    auto scan already solved: motion + soft limits, find-beam sweep,
    headless exposure calibration, exposure restore after a camera
    reconnect, and software-trigger frame grabs.
    """

    def __init__(
        self,
        writer,  # FLIRDatasetWriter
        config: AlignConfig,
        machine_limits: Bounds3D,
        scan_config: Optional[AutoScanConfig] = None,
    ):
        self.config = config
        self.machine_limits = machine_limits
        self.writer = writer

        if scan_config is None:
            scan_config = AutoScanConfig(
                PlacementID="align",
                MeasuredSensorY_mm=0.0,
                YStart_machine_mm=config.MachineY_mm,
                YStop_machine_mm=config.MachineY_mm,
                BeamDirectionSign=config.BeamDirectionSign,
                # Ranges double as the find-beam sweep extent.
                X=_axis_range(machine_limits.x_min_mm, machine_limits.x_max_mm, 5.0),
                Z=_axis_range(machine_limits.z_min_mm, machine_limits.z_max_mm, 5.0),
                CalibrationX_mm=config.ProbeX_mm,
                BackgroundMode="none",
                RasterMode="fixed",
                FindBeam=True,
            )

        self.inner = AutoScanSession(writer, scan_config, pause_fn=lambda m: None)

        # Per-Y-plane state (two planes when MachineY2_mm is set).
        self.estimates: dict[float, RingEstimate] = {}
        self.references: dict[float, tuple[float, float]] = {}
        self.centers: dict[float, tuple[float, float]] = {}

        self._lost_cycles = 0
        self._acquiring = False

        # Off-axis ambient background measured at bootstrap: the signal
        # threshold everywhere. Scaled linearly when the exposure servo
        # moves; recaptured when the drift exceeds 30%.
        self._background: Optional[dict[str, float]] = None

        # Planes whose estimate came straight from a chord survey: the
        # FIRST full-lap fit replaces it unclamped (the lap fit is
        # strictly better than the two-column seed, and clamping a
        # multi-mm bootstrap error makes convergence crawl).
        self._fresh_bootstrap: set[float] = set()

    # -- acquisition plumbing ------------------------------------------

    def _begin(self) -> None:
        if not self._acquiring:
            self.writer._begin_acquisition()
            if self.writer.config.TriggerArmDelay_s > 0:
                time.sleep(self.writer.config.TriggerArmDelay_s)
            self._acquiring = True

    def _end(self) -> None:
        if self._acquiring:
            self.writer._end_acquisition()
            self._acquiring = False

    def close(self) -> None:
        self._end()

    def grab_oriented(
        self, x_mm: float, machine_y_mm: float, z_mm: float, retries: int = 2
    ) -> Optional[np.ndarray]:
        """Move, grab, orient + downsample. None = no frame (logged)."""

        self.inner._move_to(x_mm, machine_y_mm, z_mm)
        timeout_ms = self.writer.config.AcquisitionTimeout_ms

        for _ in range(1 + retries):
            arr = _grab_frame(self.writer, timeout_ms)
            if arr is not None:
                composite_config = self.config.composite_config()
                return mean_pool(
                    orient(arr, composite_config), self.config.Downsample
                )

        logger.warning(
            f"No frame after {1 + retries} attempts at "
            f"X{x_mm:g} Y{machine_y_mm:g} Z{z_mm:g}; skipping this station."
        )
        return None

    # -- bootstrap ------------------------------------------------------

    def bootstrap(self, machine_y_mm: float) -> RingEstimate:
        """
        find-beam sweep (locates ONE point on the ring) -> exposure
        calibration there -> two-column chord survey -> ring estimate.
        """

        self._end()  # find_beam/calibrate manage their own acquisition

        if not self.inner.find_beam(machine_y_mm):
            raise AlignError(
                "find-beam saw no structured light: check the beam is on "
                "and the probe X crosses the ring (--probe-x)."
            )

        calibration = self.inner.calibrate_at(
            machine_y_mm,
            start_exposure_us=self.inner._current_exposure_us
            or self.inner.config.FindBeamStartExposure_us,
        )
        logger.info(
            f"Alignment exposure at Y{machine_y_mm:g}: "
            f"{calibration.FinalExposure_us:.0f} us "
            f"(converged={calibration.Converged})"
        )

        cap = self.config.MaxExposure_us
        current = self.inner._current_exposure_us
        if cap is not None and current is not None and current > cap:
            self.inner._set_exposure(cap)
            logger.info(
                f"Calibrated exposure {current:.0f} us exceeds the "
                f"--max-exposure cap; clamping to {cap:.0f} us."
            )

        self._capture_background(machine_y_mm)

        hit_x, hit_z = self.inner._calibration_xz
        estimate = self._chord_survey(hit_x, machine_y_mm, hit_z)
        self.estimates[machine_y_mm] = estimate
        self._fresh_bootstrap.add(machine_y_mm)

        logger.info(
            f"Bootstrap ring estimate at Y{machine_y_mm:g}: center "
            f"(X {estimate.CenterX_mm:.2f}, Z {estimate.CenterZ_mm:.2f}), "
            f"radius {estimate.Radius_mm:.2f} mm."
        )
        return estimate

    def _capture_background(self, machine_y_mm: float) -> None:
        """
        One ambient frame with the camera parked off-axis (the machine
        corner farthest from the beam, reusing the auto-scan heuristic)
        at the calibrated exposure. Its p99 anchors every signal
        threshold — the frame-median fallback breaks when a broad beam
        fills the frame (hardware lesson 2026-07-27).
        """

        bg_x, bg_z = self.inner.background_xz(self.machine_limits)
        self._begin()
        frame = self.grab_oriented(bg_x, machine_y_mm, bg_z)

        if frame is None:
            logger.warning(
                "Could not capture an off-axis background frame; falling "
                "back to frame-median thresholds (unreliable for beams "
                "wider than the sensor)."
            )
            return

        self._background = {
            "Exposure_us": float(self.inner._current_exposure_us or 0.0),
            "P99": float(np.percentile(frame, 99)),
        }
        logger.info(
            f"Off-axis background at (X {bg_x:g}, Z {bg_z:g}): p99 "
            f"{self._background['P99']:.1f} counts at "
            f"{self._background['Exposure_us']:.0f} us."
        )

    def background_level(self) -> Optional[float]:
        """Current-exposure-scaled ambient level (None = not measured)."""

        if self._background is None:
            return None

        level = self._background["P99"]
        bg_exposure = self._background["Exposure_us"]
        exposure = self.inner._current_exposure_us

        if exposure and bg_exposure:
            scale = exposure / bg_exposure
            level *= min(max(scale, 0.05), 20.0)
        return level

    def _refresh_background_if_stale(self, machine_y_mm: float) -> None:
        if self._background is None:
            return

        bg_exposure = self._background["Exposure_us"]
        exposure = self.inner._current_exposure_us

        if not exposure or not bg_exposure:
            return

        if abs(exposure - bg_exposure) / bg_exposure > 0.30:
            logger.info(
                f"Exposure drifted {exposure:.0f} us vs background's "
                f"{bg_exposure:.0f} us — recapturing the off-axis "
                "background."
            )
            self._capture_background(machine_y_mm)

    def _survey_column(
        self, x_mm: float, machine_y_mm: float
    ) -> list[tuple[float, float]]:
        z_low = self.machine_limits.z_min_mm
        z_high = self.machine_limits.z_max_mm

        rows = self._frame_shape_hint()[0]
        step = max(2.0, rows * self.config.mm_per_px() * 0.9)  # ~10% overlap
        z_values = list(np.arange(z_low, z_high + 1e-9, step))

        column: list[tuple[np.ndarray, float]] = []
        for z_mm in z_values:
            frame = self.grab_oriented(x_mm, machine_y_mm, z_mm)
            if frame is not None:
                column.append((frame, z_mm))

        return column_crossing_intervals(
            column, self.config, x_mm, self.background_level()
        )

    def _chord_survey(
        self, hit_x: float, machine_y_mm: float, hit_z: float
    ) -> RingEstimate:
        self._begin()

        x1 = hit_x
        intervals1 = self._survey_column(x1, machine_y_mm)

        if not intervals1:
            raise AlignError(
                f"Survey column at X{x1:g} saw no ring crossings even "
                "though find-beam hit there — exposure may have drifted."
            )

        # Second column: prefer +dx, fall back to -dx (ring may end
        # before the machine limit on one side).
        for dx in (self.config.SurveyDX_mm, -self.config.SurveyDX_mm):
            x2 = x1 + dx
            if not (
                self.machine_limits.x_min_mm <= x2 <= self.machine_limits.x_max_mm
            ):
                continue

            intervals2 = self._survey_column(x2, machine_y_mm)
            solution = solve_ring_from_chords(x1, intervals1, x2, intervals2)

            if solution is not None:
                cx, cz, radius = solution
                radius = self._sane_radius(radius)
                return RingEstimate(cx, cz, radius)

        raise AlignError(
            "Chord survey could not solve the ring: only one usable "
            "survey column. Try --probe-x closer to the ring center or "
            "provide --ring-diameter."
        )

    def _sane_radius(self, radius: float) -> float:
        if self.config.RingDiameter_mm is not None:
            prior = self.config.RingDiameter_mm / 2.0
            if not (0.4 * prior <= radius <= 2.5 * prior):
                logger.warning(
                    f"Chord-survey radius {radius:.2f} mm is far from the "
                    f"--ring-diameter prior ({prior:.2f} mm); using the "
                    "prior for the first patrol."
                )
                return prior
        return radius

    def _frame_shape_hint(self) -> tuple[int, int]:
        """Downsampled frame shape (rows, cols); BFS-PGE-31S4M default."""

        try:
            rows = int(self.writer.cam.Height.GetValue())
            cols = int(self.writer.cam.Width.GetValue())
        except Exception:  # noqa: BLE001 - fakes/older settings: use default
            rows, cols = 1536, 2048
        return rows // self.config.Downsample, cols // self.config.Downsample

    # -- patrol cycle ---------------------------------------------------

    def _reachable(self, x: float, z: float) -> bool:
        return (
            self.machine_limits.x_min_mm <= x <= self.machine_limits.x_max_mm
            and self.machine_limits.z_min_mm <= z <= self.machine_limits.z_max_mm
        )

    def station_positions(
        self, estimate: RingEstimate
    ) -> list[tuple[float, float, bool]]:
        """(x, z, reachable) for each patrol station on the estimate."""

        stations = []
        for i in range(self.config.Stations):
            angle = 2.0 * math.pi * i / self.config.Stations
            x = estimate.CenterX_mm + estimate.Radius_mm * math.cos(angle)
            z = estimate.CenterZ_mm + estimate.Radius_mm * math.sin(angle)
            stations.append((x, z, self._reachable(x, z)))
        return stations

    def fill_positions(
        self, estimate: RingEstimate
    ) -> list[tuple[float, float, bool]]:
        """
        Disk cover mode's interior stations: a half-radius ring with half
        the station count (offset half a step so they sit between the
        outer stations) plus the center itself.
        """

        if self.config.CoverMode != "disk":
            return []

        positions = []
        n_inner = max(1, self.config.Stations // 2)
        for i in range(n_inner):
            angle = 2.0 * math.pi * (i + 0.5) / n_inner
            x = estimate.CenterX_mm + 0.5 * estimate.Radius_mm * math.cos(angle)
            z = estimate.CenterZ_mm + 0.5 * estimate.Radius_mm * math.sin(angle)
            positions.append((x, z, self._reachable(x, z)))

        positions.append(
            (
                estimate.CenterX_mm,
                estimate.CenterZ_mm,
                self._reachable(estimate.CenterX_mm, estimate.CenterZ_mm),
            )
        )
        return positions

    def run_cycle(
        self,
        index: int,
        machine_y_mm: float,
        on_station: Optional[Callable[[StationSample, np.ndarray], None]] = None,
    ) -> tuple[CycleResult, list[tuple[np.ndarray, float, float]]]:
        started = time.monotonic()
        self._begin()
        self._refresh_background_if_stale(machine_y_mm)

        estimate = self.estimates[machine_y_mm]
        samples: list[StationSample] = []
        frames: list[tuple[np.ndarray, float, float]] = []
        all_points: list[np.ndarray] = []
        all_angles: list[np.ndarray] = []
        all_intensities: list[np.ndarray] = []
        all_widths: list[float] = []
        peak_seen = 0.0

        stations = [
            (x, z, reachable, "ring")
            for x, z, reachable in self.station_positions(estimate)
        ] + [
            (x, z, reachable, "fill")
            for x, z, reachable in self.fill_positions(estimate)
        ]

        for x_mm, z_mm, reachable, role in stations:
            if not reachable:
                samples.append(
                    StationSample(x_mm, z_mm, False, 0.0, Skipped=True, Role=role)
                )
                continue

            frame = self.grab_oriented(x_mm, machine_y_mm, z_mm)
            if frame is None:
                samples.append(StationSample(x_mm, z_mm, False, 0.0, Role=role))
                continue

            arc = extract_arc(
                frame,
                x_mm,
                z_mm,
                (estimate.CenterX_mm, estimate.CenterZ_mm),
                self.config,
                background_level=self.background_level(),
            )

            peak = float(frame.max())
            peak_seen = max(peak_seen, peak)
            sample = StationSample(x_mm, z_mm, arc is not None, peak, Role=role)
            samples.append(sample)
            frames.append((frame, x_mm, z_mm))

            # Fill stations only fill in the composite: their "arc"
            # points are interior light, not the ring locus.
            if arc is not None and role == "ring":
                all_points.append(arc.Points_mm)
                all_angles.append(arc.Angles_deg)
                all_intensities.append(arc.Intensities)
                all_widths.append(arc.WidthFWHM_mm)

            if on_station is not None:
                on_station(sample, frame)

        skipped = sum(1 for s in samples if s.Skipped)
        if skipped:
            logger.warning(
                f"{skipped}/{len(samples)} stations are outside the "
                "machine envelope — the ring is only partially reachable."
            )

        points = (
            np.concatenate(all_points) if all_points else np.empty((0, 2))
        )
        lost = points.shape[0] < 3

        circle = fit_circle(points) if not lost else None
        ellipse = fit_ellipse(points) if not lost else None
        uniformity = (
            azimuthal_uniformity(
                np.concatenate(all_angles), np.concatenate(all_intensities)
            )
            if all_angles
            else None
        )

        estimate = self._update_estimate(machine_y_mm, circle)

        reference = self.references.get(machine_y_mm)
        offset = None
        if circle is not None:
            self.centers[machine_y_mm] = (circle.CenterX_mm, circle.CenterZ_mm)
            if reference is not None:
                offset = (
                    circle.CenterX_mm - reference[0],
                    circle.CenterZ_mm - reference[1],
                )

        self._exposure_servo(peak_seen, lost)

        result = CycleResult(
            Index=index,
            MachineY_mm=machine_y_mm,
            Estimate=estimate,
            Circle=circle,
            Ellipse=ellipse,
            Uniformity=uniformity,
            Reference=reference,
            Offset_mm=offset,
            Tilt=self._tilt_metrics(),
            Stations=samples,
            Points_mm=points,
            Exposure_us=self.inner._current_exposure_us,
            Lost=lost,
            Elapsed_s=time.monotonic() - started,
            WidthFWHM_mm=float(np.mean(all_widths)) if all_widths else None,
        )
        return result, frames

    def _update_estimate(
        self, machine_y_mm: float, circle: Optional[CircleFit]
    ) -> RingEstimate:
        estimate = self.estimates[machine_y_mm]

        if circle is None:
            return estimate

        if machine_y_mm in self._fresh_bootstrap:
            # First full lap after a chord survey: the lap fit is
            # strictly better than the two-column seed — adopt it
            # unclamped (a multi-mm seed error would otherwise take
            # many laps at MaxRingShift_mm per cycle to walk off).
            self._fresh_bootstrap.discard(machine_y_mm)
            updated = RingEstimate(
                CenterX_mm=circle.CenterX_mm,
                CenterZ_mm=circle.CenterZ_mm,
                Radius_mm=circle.Radius_mm,
            )
            self.estimates[machine_y_mm] = updated
            return updated

        clamp = self.config.MaxRingShift_mm

        def clamped(new: float, old: float) -> float:
            return old + max(-clamp, min(clamp, new - old))

        updated = RingEstimate(
            CenterX_mm=clamped(circle.CenterX_mm, estimate.CenterX_mm),
            CenterZ_mm=clamped(circle.CenterZ_mm, estimate.CenterZ_mm),
            Radius_mm=clamped(circle.Radius_mm, estimate.Radius_mm),
        )
        self.estimates[machine_y_mm] = updated
        return updated

    def _exposure_servo(self, peak_seen: float, lost: bool) -> None:
        """
        Between-cycles exposure trim: halve on saturation, double when
        everything is dim (a full headless recalibration is only used at
        bootstrap — the servo keeps up with slow drift for free).
        """

        exposure = self.inner._current_exposure_us
        if exposure is None or lost:
            return

        cap = self.config.MaxExposure_us
        if cap is not None and exposure > cap:
            self.inner._set_exposure(cap)
            logger.info(
                f"Exposure servo: clamping {exposure:.0f} us to the "
                f"--max-exposure cap ({cap:.0f} us)."
            )
            return

        saturation = self.config.Saturation.SaturationThreshold
        if peak_seen >= saturation:
            self.inner._set_exposure(exposure * 0.5)
            logger.info(
                f"Exposure servo: saturated (peak {peak_seen:g}); halving "
                f"to {exposure * 0.5:.0f} us."
            )
        elif peak_seen < self.config.DimFraction * saturation:
            doubled = exposure * 2.0
            if cap is not None:
                doubled = min(doubled, cap)
            if doubled > exposure:
                self.inner._set_exposure(doubled)
                logger.info(
                    f"Exposure servo: dim (peak {peak_seen:g}); raising to "
                    f"{doubled:.0f} us."
                )

    def _tilt_metrics(self) -> Optional[TiltMetrics]:
        if self.config.MachineY2_mm is None:
            return None

        y1 = self.config.MachineY_mm
        y2 = self.config.MachineY2_mm
        if y1 not in self.centers or y2 not in self.centers:
            return None

        # Beam-path distance from plane 1 to plane 2: positive downstream.
        delta_beam = self.config.BeamDirectionSign * (y2 - y1)
        if abs(delta_beam) < 1e-9:
            return None

        c1 = self.centers[y1]
        c2 = self.centers[y2]
        return TiltMetrics(
            TiltX_mrad=(c2[0] - c1[0]) / delta_beam * 1000.0,
            TiltZ_mrad=(c2[1] - c1[1]) / delta_beam * 1000.0,
            DeltaBeamPath_mm=delta_beam,
        )

    # -- user actions ---------------------------------------------------

    def set_reference_here(self) -> None:
        """Zero the offset readout at the CURRENT fitted centers."""

        for y, center in self.centers.items():
            self.references[y] = center
        logger.info(f"Alignment reference set: {self.references}")

    def set_reference(self, x_mm: float, z_mm: float) -> None:
        for y in self._planes():
            self.references[y] = (x_mm, z_mm)

    def _planes(self) -> list[float]:
        planes = [self.config.MachineY_mm]
        if self.config.MachineY2_mm is not None:
            planes.append(self.config.MachineY2_mm)
        return planes

    # -- park-and-stream mode ------------------------------------------

    def park_position(
        self,
        cycle: CycleResult,
        azimuth_deg: Optional[float] = None,
    ) -> tuple[float, float]:
        """
        Where to park for streaming: on the ring at the requested
        azimuth, else at the station that saw the brightest signal
        during the given orbit cycle.
        """

        estimate = self.estimates[cycle.MachineY_mm]

        if azimuth_deg is not None:
            theta = math.radians(azimuth_deg)
            return (
                estimate.CenterX_mm + estimate.Radius_mm * math.cos(theta),
                estimate.CenterZ_mm + estimate.Radius_mm * math.sin(theta),
            )

        lit = [
            s
            for s in cycle.Stations
            if s.HasSignal and not s.Skipped and s.Role == "ring"
        ]
        if lit:
            best = max(lit, key=lambda s: s.Peak)
            return best.X_mm, best.Z_mm

        return (
            estimate.CenterX_mm + estimate.Radius_mm,
            estimate.CenterZ_mm,
        )

    def stream_frame(
        self, index: int, machine_y_mm: float, park: tuple[float, float]
    ) -> tuple[StreamSample, Optional[np.ndarray]]:
        """
        One parked measurement: grab (no motion beyond the first move to
        the park point), extract the arc in view, and fit the ring
        center with the radius FIXED to the current estimate — the only
        well-posed single-arc fit. Everything is relative to the last
        orbit's geometry.
        """

        started = time.monotonic()
        estimate = self.estimates[machine_y_mm]
        park_x, park_z = park

        frame = self.grab_oriented(park_x, machine_y_mm, park_z)
        arc = (
            extract_arc(
                frame,
                park_x,
                park_z,
                (estimate.CenterX_mm, estimate.CenterZ_mm),
                self.config,
                background_level=self.background_level(),
            )
            if frame is not None
            else None
        )

        center_x = center_z = rms = None
        offset = None
        width = None
        peak = 0.0

        if arc is not None:
            peak = arc.Peak
            width = arc.WidthFWHM_mm
            fit = fit_center_fixed_radius(
                arc.Points_mm,
                estimate.Radius_mm,
                (estimate.CenterX_mm, estimate.CenterZ_mm),
            )
            if fit is not None:
                center_x, center_z, rms = fit
                reference = self.references.get(machine_y_mm)
                self.centers[machine_y_mm] = (center_x, center_z)
                if reference is not None:
                    offset = (
                        center_x - reference[0],
                        center_z - reference[1],
                    )

        self._exposure_servo(peak, lost=arc is None)

        sample = StreamSample(
            Index=index,
            MachineY_mm=machine_y_mm,
            ParkX_mm=park_x,
            ParkZ_mm=park_z,
            CenterX_mm=center_x,
            CenterZ_mm=center_z,
            FitRMS_mm=rms,
            Radius_mm=estimate.Radius_mm,
            Offset_mm=offset,
            WidthFWHM_mm=width,
            ArcPeak=peak,
            Lost=arc is None,
            Elapsed_s=time.monotonic() - started,
        )
        return sample, frame

    def run_stream(
        self,
        on_frame: Callable[[StreamSample, Optional[np.ndarray]], str],
        on_cycle: Optional[
            Callable[[CycleResult, list[tuple[np.ndarray, float, float]]], None]
        ] = None,
        park_azimuth_deg: Optional[float] = None,
        orbit_every_s: float = 0.0,
        max_frames: Optional[int] = None,
    ) -> None:
        """
        Find -> orbit once -> park -> stream. on_frame gets every
        StreamSample (plus the oriented frame) and returns one of the
        STREAM_* actions: continue, stop, orbit (full patrol lap now —
        refreshes radius/roundness/uniformity and the ring estimate),
        or refind (find-beam bootstrap from scratch). orbit_every_s > 0
        also runs a lap on a timer. The ring being missing from
        LostCyclesBeforeRefind consecutive frames forces an orbit, and
        a lost orbit escalates to a re-find, so a blocked beam heals
        without user input.
        """

        machine_y = self.config.MachineY_mm
        cycle_index = 0
        frame_index = 0
        consecutive_lost = 0

        def orbit() -> CycleResult:
            nonlocal cycle_index
            if machine_y not in self.estimates:
                self.bootstrap(machine_y)
            result, frames = self.run_cycle(cycle_index, machine_y)
            cycle_index += 1
            if on_cycle is not None:
                on_cycle(result, frames)
            return result

        cycle = orbit()
        park = self.park_position(cycle, park_azimuth_deg)
        last_orbit = time.monotonic()

        try:
            while max_frames is None or frame_index < max_frames:
                self._begin()
                sample, frame = self.stream_frame(frame_index, machine_y, park)

                action = on_frame(sample, frame)
                frame_index += 1

                consecutive_lost = consecutive_lost + 1 if sample.Lost else 0
                timed_orbit = (
                    orbit_every_s > 0
                    and time.monotonic() - last_orbit >= orbit_every_s
                )

                if action == STREAM_STOP:
                    break

                if action == STREAM_REFIND:
                    self.estimates.pop(machine_y, None)

                if (
                    action in (STREAM_ORBIT, STREAM_REFIND)
                    or timed_orbit
                    or consecutive_lost >= self.config.LostCyclesBeforeRefind
                ):
                    if consecutive_lost >= self.config.LostCyclesBeforeRefind:
                        logger.warning(
                            f"Ring missing from {consecutive_lost} parked "
                            "frames — running an orbit to relocate it."
                        )
                    try:
                        cycle = orbit()
                        if cycle.Lost:
                            logger.warning(
                                "Orbit lost the ring too — re-running the "
                                "find-beam bootstrap."
                            )
                            self.estimates.pop(machine_y, None)
                            cycle = orbit()
                        park = self.park_position(cycle, park_azimuth_deg)
                        consecutive_lost = 0
                    except AlignError as ex:
                        # Beam blocked / gone: keep streaming rather than
                        # dying — it may come back. Back off so we do not
                        # burn a find-beam sweep every couple of frames.
                        logger.warning(
                            f"Relocating the ring failed ({ex}); still "
                            "streaming — press f to retry, or unblock the "
                            "beam and it will retry on its own."
                        )
                        if machine_y not in self.estimates:
                            # Keep the last known geometry so parked
                            # frames still have a center guess.
                            self.estimates[machine_y] = RingEstimate(
                                *self.centers.get(
                                    machine_y,
                                    (park[0], park[1]),
                                ),
                                Radius_mm=cycle.Estimate.Radius_mm,
                            )
                        consecutive_lost = (
                            -3 * self.config.LostCyclesBeforeRefind
                        )
                    last_orbit = time.monotonic()
        finally:
            self._end()

    # -- main loop ------------------------------------------------------

    def run(
        self,
        on_cycle: Callable[
            [CycleResult, list[tuple[np.ndarray, float, float]]], bool
        ],
        max_cycles: Optional[int] = None,
        on_station: Optional[Callable[[StationSample, np.ndarray], None]] = None,
    ) -> list[CycleResult]:
        """
        Patrol until on_cycle returns False (or max_cycles). on_cycle
        gets each CycleResult plus that cycle's frames for the preview.
        """

        results: list[CycleResult] = []
        planes = self._planes()
        index = 0

        try:
            while max_cycles is None or index < max_cycles:
                machine_y = planes[index % len(planes)]

                if machine_y not in self.estimates:
                    self.bootstrap(machine_y)

                result, frames = self.run_cycle(
                    index, machine_y, on_station=on_station
                )
                results.append(result)

                if result.Lost:
                    self._lost_cycles += 1
                    logger.warning(
                        f"Cycle {index}: ring not seen "
                        f"({self._lost_cycles} consecutive)."
                    )
                    if self._lost_cycles >= self.config.LostCyclesBeforeRefind:
                        logger.warning(
                            "Ring lost — re-running the find-beam bootstrap."
                        )
                        self.estimates.pop(machine_y, None)
                        self._lost_cycles = 0
                else:
                    self._lost_cycles = 0

                if not on_cycle(result, frames):
                    break

                if self.config.CycleInterval_s > 0:
                    remaining = self.config.CycleInterval_s - result.Elapsed_s
                    if remaining > 0:
                        time.sleep(remaining)

                index += 1
        finally:
            self._end()

        return results


@dataclass(frozen=True)
class StreamSample:
    """One parked single-frame measurement (a few Hz, no motion)."""

    Index: int
    MachineY_mm: float
    ParkX_mm: float
    ParkZ_mm: float
    # Fixed-radius arc fit of THIS frame (radius from the last orbit).
    CenterX_mm: Optional[float]
    CenterZ_mm: Optional[float]
    FitRMS_mm: Optional[float]
    Radius_mm: float
    Offset_mm: Optional[tuple[float, float]]
    WidthFWHM_mm: Optional[float]
    ArcPeak: float
    Lost: bool
    Elapsed_s: float


# Actions an on_frame callback can return to steer the stream loop.
STREAM_CONTINUE = "continue"
STREAM_STOP = "stop"
STREAM_ORBIT = "orbit"  # run one full patrol lap now
STREAM_REFIND = "refind"  # rerun the find-beam bootstrap


def _axis_range(start: float, stop: float, step: float):
    from coordinates import AxisRange

    return AxisRange(start_mm=start, stop_mm=stop, step_mm=step)


def append_cycle_log(path, result: CycleResult) -> None:
    """One JSONL line of metrics per cycle (before/after lab records)."""

    with open(path, "a") as handle:
        handle.write(json.dumps(result.to_jsonable()) + "\n")
