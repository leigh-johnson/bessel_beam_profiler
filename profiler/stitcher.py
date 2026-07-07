"""
Stitch overlapping beam-profile frames taken with a manually translated camera
into one composite image.

Registration strategy
---------------------

The default method is FFT phase correlation, which directly estimates the
(dy, dx) translation between two overlapping frames. This is the right tool
for a camera on a translation stage: the transform between frames is (very
nearly) a pure translation, and smooth beam profiles rarely have enough
corner-like texture for feature-based matchers to work reliably.

An optional OpenCV ORB feature-matching path is provided as a fallback for
frames where phase correlation is not confident (e.g. very small overlap).
It is only used if `opencv-python` is importable.

No PySpin dependency: this module operates on saved .npy arrays / in-memory
numpy arrays, so it can be run offline on an existing run directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Sequence
import json

import numpy as np


class StitchError(RuntimeError):
    pass


@dataclass(frozen=True)
class ShiftEstimate:
    """
    Estimated translation of frame B's field of view relative to frame A's,
    in pixels — i.e. the camera/stage displacement between the two frames.

    A positive dy means the camera's field of view moved `dy` pixels down
    (row-wise) in the scene, so the beam's content appears `dy` pixels
    *higher* in frame B. These offsets are directly usable as canvas
    placement positions when compositing.
    """

    dy_px: float
    dx_px: float

    # Peak height of the normalized phase-correlation surface (0..1] or a
    # match-agreement fraction for the OpenCV path. Higher is better.
    Confidence: float

    # Pearson correlation of the two frames over their implied overlap
    # region (-1..1). This is the primary quality gate: ~1.0 means the two
    # frames agree almost perfectly once shifted onto each other.
    OverlapNCC: float = 0.0

    Method: str = "PhaseCorrelation"


@dataclass(frozen=True)
class StitchConfig:
    # "phase"  = phase correlation only
    # "opencv" = ORB feature matching only
    # "auto"   = phase correlation, falling back to OpenCV (if installed)
    #            whenever the phase result's overlap correlation is low
    Method: str = "auto"

    # Below this overlap Pearson correlation, the "auto" method tries the
    # OpenCV fallback and keeps whichever estimate validates better.
    MinOverlapNCC: float = 0.6

    # Candidate shifts must imply at least this many overlapping pixels.
    MinOverlap_px: int = 1024

    # Percentile normalization applied before registration (robust to hot
    # pixels) and when rendering the composite PNG.
    NormalizationPercentiles: tuple[float, float] = (0.5, 99.8)

    # Colormap for the quick-look composite PNG.
    Colormap: str = "inferno"


@dataclass(frozen=True)
class StitchResult:
    # One (cumulative) offset per input frame, relative to the first frame.
    OffsetsPx: list[tuple[float, float]]

    # Pairwise estimates between consecutive frames (len = nframes - 1).
    PairwiseShifts: list[ShiftEstimate]

    # Feather-blended composite (float32) and per-pixel weight coverage.
    Composite: np.ndarray
    Coverage: np.ndarray

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "OffsetsPx": [
                {"dy_px": dy, "dx_px": dx} for (dy, dx) in self.OffsetsPx
            ],
            "PairwiseShifts": [asdict(s) for s in self.PairwiseShifts],
            "CompositeShape": list(self.Composite.shape),
        }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _normalize_for_registration(
    arr: np.ndarray,
    percentiles: tuple[float, float],
) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)

    lo, hi = np.percentile(a, percentiles)

    if hi <= lo:
        return np.zeros_like(a)

    a = np.clip(a, lo, hi)
    a -= a.mean()
    return a


def _hann_window_2d(shape: tuple[int, int]) -> np.ndarray:
    wy = np.hanning(shape[0])
    wx = np.hanning(shape[1])
    return np.outer(wy, wx)


def _parabolic_subpixel(values: np.ndarray, peak_index: int) -> float:
    """
    Refine a 1D peak location with a 3-point parabolic fit.
    Returns a fractional correction in (-0.5, 0.5).
    """

    if peak_index <= 0 or peak_index >= len(values) - 1:
        return 0.0

    left, center, right = (
        values[peak_index - 1],
        values[peak_index],
        values[peak_index + 1],
    )

    denom = left - 2.0 * center + right

    if denom == 0.0:
        return 0.0

    return float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))


def overlap_ncc(
    ref: np.ndarray,
    mov: np.ndarray,
    dy_px: float,
    dx_px: float,
    *,
    min_overlap_px: int = 1024,
) -> float:
    """
    Validate a candidate shift by directly comparing the two frames over the
    overlap region it implies: Pearson correlation of ref against mov placed
    at (dy, dx). Returns -1.0 when the implied overlap is too small to judge.
    """

    dy = int(round(dy_px))
    dx = int(round(dx_px))

    h, w = ref.shape

    y0, y1 = max(0, dy), min(h, h + dy)
    x0, x1 = max(0, dx), min(w, w + dx)

    if (y1 - y0) * (x1 - x0) < min_overlap_px:
        return -1.0

    a = np.asarray(ref[y0:y1, x0:x1], dtype=np.float64).ravel()
    b = np.asarray(mov[y0 - dy : y1 - dy, x0 - dx : x1 - dx], dtype=np.float64).ravel()

    a = a - a.mean()
    b = b - b.mean()

    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))

    if denom < 1e-12:
        return 0.0

    return float(np.sum(a * b) / denom)


def phase_correlation_shift(
    ref: np.ndarray,
    mov: np.ndarray,
    *,
    percentiles: tuple[float, float] = (0.5, 99.8),
    min_overlap_px: int = 1024,
) -> ShiftEstimate:
    """
    Estimate the translation of `mov` relative to `ref` via phase correlation.

    Circular FFT correlation only determines the shift modulo the frame size,
    so each axis has two plausible branches (d and d - N). All four
    combinations are validated by direct overlap correlation and the best
    one wins — this makes stage moves larger than half the sensor size work.
    """

    if ref.shape != mov.shape:
        raise StitchError(
            f"Frames must have identical shapes; got {ref.shape} and {mov.shape}."
        )

    if ref.ndim != 2:
        raise StitchError(f"Expected 2D grayscale frames; got ndim={ref.ndim}.")

    window = _hann_window_2d(ref.shape)

    a = _normalize_for_registration(ref, percentiles) * window
    b = _normalize_for_registration(mov, percentiles) * window

    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)

    cross_power = fa * np.conj(fb)
    magnitude = np.abs(cross_power)

    # Avoid division by zero for empty spectral bins.
    magnitude[magnitude < 1e-15] = 1e-15

    correlation = np.fft.irfft2(cross_power / magnitude, s=ref.shape)

    peak_flat = int(np.argmax(correlation))
    peak_y, peak_x = np.unravel_index(peak_flat, correlation.shape)

    confidence = float(correlation[peak_y, peak_x])

    # Sub-pixel refinement along each axis (with wrap-around neighbors).
    row = correlation[peak_y, :]
    col = correlation[:, peak_x]

    frac_x = _parabolic_subpixel(np.roll(row, 1 - peak_x), 1) if row.size > 2 else 0.0
    frac_y = _parabolic_subpixel(np.roll(col, 1 - peak_y), 1) if col.size > 2 else 0.0

    # With mov(y, x) == ref(y + d, x + d'), the cross-power spectrum is
    # exp(-2*pi*i*k*d/N) and its inverse FFT peaks exactly at (d, d') —
    # the field-of-view (camera) displacement between the frames — but only
    # modulo the frame size. Disambiguate the wraparound branches by direct
    # overlap validation.
    dy_base = float(peak_y) + frac_y
    dx_base = float(peak_x) + frac_x

    h, w = ref.shape

    candidates = [
        (dy_cand, dx_cand)
        for dy_cand in (dy_base, dy_base - h)
        for dx_cand in (dx_base, dx_base - w)
    ]

    best_dy, best_dx, best_ncc = 0.0, 0.0, -np.inf

    for dy_cand, dx_cand in candidates:
        ncc = overlap_ncc(
            ref, mov, dy_cand, dx_cand, min_overlap_px=min_overlap_px
        )

        if ncc > best_ncc:
            best_dy, best_dx, best_ncc = dy_cand, dx_cand, ncc

    return ShiftEstimate(
        dy_px=best_dy,
        dx_px=best_dx,
        Confidence=confidence,
        OverlapNCC=best_ncc,
        Method="PhaseCorrelation",
    )


def opencv_feature_shift(
    ref: np.ndarray,
    mov: np.ndarray,
    *,
    percentiles: tuple[float, float] = (0.5, 99.8),
    max_features: int = 2000,
    min_overlap_px: int = 1024,
) -> ShiftEstimate:
    """
    Estimate a pure translation from ORB feature matches (OpenCV fallback).

    Raises StitchError if OpenCV is unavailable or matching fails.
    """

    try:
        import cv2
    except ImportError as ex:
        raise StitchError(
            "OpenCV is not installed; `pip install opencv-python` to enable "
            "the feature-based fallback."
        ) from ex

    def to_uint8(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float64)
        lo, hi = np.percentile(a, percentiles)

        if hi <= lo:
            hi = lo + 1.0

        a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
        return (a * 255.0).astype(np.uint8)

    img_a = to_uint8(ref)
    img_b = to_uint8(mov)

    orb = cv2.ORB_create(nfeatures=max_features)
    kp_a, desc_a = orb.detectAndCompute(img_a, None)
    kp_b, desc_b = orb.detectAndCompute(img_b, None)

    if desc_a is None or desc_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        raise StitchError("Not enough ORB features to match (smooth image?).")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc_a, desc_b)

    if len(matches) < 4:
        raise StitchError(f"Only {len(matches)} ORB matches; need at least 4.")

    matches = sorted(matches, key=lambda m: m.distance)[: max(8, len(matches) // 2)]

    # A feature at row/col p in frame A appears at p - d in frame B when the
    # camera moves by +d, so the camera displacement is kp_a - kp_b. This
    # matches the ShiftEstimate convention used by phase_correlation_shift.
    displacements = np.array(
        [
            (
                kp_a[m.queryIdx].pt[1] - kp_b[m.trainIdx].pt[1],  # dy
                kp_a[m.queryIdx].pt[0] - kp_b[m.trainIdx].pt[0],  # dx
            )
            for m in matches
        ]
    )

    dy, dx = np.median(displacements, axis=0)

    # Pseudo-confidence: fraction of matches that agree with the median
    # displacement to within 2 px.
    agreement = np.all(np.abs(displacements - [dy, dx]) <= 2.0, axis=1)
    confidence = float(np.mean(agreement))

    return ShiftEstimate(
        dy_px=float(dy),
        dx_px=float(dx),
        Confidence=confidence,
        OverlapNCC=overlap_ncc(ref, mov, dy, dx, min_overlap_px=min_overlap_px),
        Method="OpenCV-ORB",
    )


def estimate_shift(
    ref: np.ndarray,
    mov: np.ndarray,
    config: StitchConfig,
) -> ShiftEstimate:
    if config.Method == "phase":
        return phase_correlation_shift(
            ref,
            mov,
            percentiles=config.NormalizationPercentiles,
            min_overlap_px=config.MinOverlap_px,
        )

    if config.Method == "opencv":
        return opencv_feature_shift(
            ref,
            mov,
            percentiles=config.NormalizationPercentiles,
            min_overlap_px=config.MinOverlap_px,
        )

    if config.Method != "auto":
        raise StitchError(
            f"Unknown stitch method {config.Method!r}. "
            "Use 'phase', 'opencv', or 'auto'."
        )

    estimate = phase_correlation_shift(
        ref,
        mov,
        percentiles=config.NormalizationPercentiles,
        min_overlap_px=config.MinOverlap_px,
    )

    if estimate.OverlapNCC >= config.MinOverlapNCC:
        return estimate

    try:
        fallback = opencv_feature_shift(
            ref,
            mov,
            percentiles=config.NormalizationPercentiles,
            min_overlap_px=config.MinOverlap_px,
        )
    except StitchError:
        return estimate  # keep the phase result; nothing better available

    # Keep whichever estimate validates better against the actual pixels.
    return fallback if fallback.OverlapNCC > estimate.OverlapNCC else estimate


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------


def _feather_weights(shape: tuple[int, int]) -> np.ndarray:
    """
    Per-pixel blending weight: distance to the nearest frame edge (+1),
    so seams between overlapping frames fade smoothly.
    """

    rows = np.minimum(np.arange(shape[0]), np.arange(shape[0])[::-1])
    cols = np.minimum(np.arange(shape[1]), np.arange(shape[1])[::-1])
    return (np.minimum.outer(rows, cols) + 1.0).astype(np.float64)


def stitch_frames(
    frames: Sequence[np.ndarray],
    config: Optional[StitchConfig] = None,
) -> StitchResult:
    """
    Register consecutive frames pairwise and blend them onto one canvas.

    Frames must be 2D arrays of identical shape, ordered so that consecutive
    frames overlap (i.e. the order you saved them in while moving the stage).
    """

    config = config or StitchConfig()

    if len(frames) == 0:
        raise StitchError("No frames to stitch.")

    shapes = {np.asarray(f).shape for f in frames}

    if len(shapes) != 1:
        raise StitchError(f"All frames must share one shape; got {shapes}.")

    frames = [np.asarray(f, dtype=np.float64) for f in frames]

    pairwise: list[ShiftEstimate] = []
    offsets: list[tuple[float, float]] = [(0.0, 0.0)]

    for prev, curr in zip(frames[:-1], frames[1:]):
        shift = estimate_shift(prev, curr, config)
        pairwise.append(shift)

        last_dy, last_dx = offsets[-1]
        offsets.append((last_dy + shift.dy_px, last_dx + shift.dx_px))

    # Integer placement offsets, shifted so the canvas starts at (0, 0).
    int_offsets = [(int(round(dy)), int(round(dx))) for dy, dx in offsets]

    min_dy = min(dy for dy, _ in int_offsets)
    min_dx = min(dx for _, dx in int_offsets)

    placements = [(dy - min_dy, dx - min_dx) for dy, dx in int_offsets]

    frame_h, frame_w = frames[0].shape
    canvas_h = max(dy for dy, _ in placements) + frame_h
    canvas_w = max(dx for _, dx in placements) + frame_w

    accumulator = np.zeros((canvas_h, canvas_w), dtype=np.float64)
    weights = np.zeros((canvas_h, canvas_w), dtype=np.float64)
    feather = _feather_weights((frame_h, frame_w))

    for frame, (dy, dx) in zip(frames, placements):
        accumulator[dy : dy + frame_h, dx : dx + frame_w] += frame * feather
        weights[dy : dy + frame_h, dx : dx + frame_w] += feather

    composite = np.divide(
        accumulator,
        weights,
        out=np.zeros_like(accumulator),
        where=weights > 0,
    ).astype(np.float32)

    return StitchResult(
        OffsetsPx=offsets,
        PairwiseShifts=pairwise,
        Composite=composite,
        Coverage=weights.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Run-directory integration
# ---------------------------------------------------------------------------


def load_run_frames(run_dir: Path) -> list[tuple[Path, np.ndarray]]:
    """
    Load frame arrays from a dataset run directory, in acquisition order.

    Uses frames.jsonl (the manifest written by FLIRDatasetWriter) when
    present; otherwise falls back to sorted *.npy files.
    """

    run_dir = Path(run_dir)
    manifest_path = run_dir / "frames.jsonl"

    paths: list[Path] = []

    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            if not line.strip():
                continue

            record = json.loads(line)
            path = Path(record["Path"])

            # Manifests written on another machine may hold absolute paths
            # that no longer resolve; fall back to the local run dir.
            if not path.exists():
                path = run_dir / path.name

            if path.suffix == ".npy" and path.exists():
                paths.append(path)
    else:
        paths = sorted(run_dir.glob("*.npy"))

    # Never try to stitch a previous composite into itself.
    paths = [p for p in paths if not p.name.startswith("composite")]

    if not paths:
        raise StitchError(f"No .npy frames found in {run_dir}.")

    return [(p, np.load(p)) for p in paths]


def stitch_run_dir(
    run_dir: Path,
    config: Optional[StitchConfig] = None,
    *,
    output_stem: str = "composite",
) -> dict[str, Path]:
    """
    Stitch all frames in a run directory and write:

        <output_stem>.npy   float32 composite (source of truth)
        <output_stem>.png   normalized quick-look render
        <output_stem>_offsets.json   per-frame offsets + pairwise estimates

    Returns the paths of the written artifacts.
    """

    config = config or StitchConfig()
    run_dir = Path(run_dir)

    named_frames = load_run_frames(run_dir)
    frames = [arr for _, arr in named_frames]

    result = stitch_frames(frames, config)

    npy_path = run_dir / f"{output_stem}.npy"
    png_path = run_dir / f"{output_stem}.png"
    json_path = run_dir / f"{output_stem}_offsets.json"

    np.save(npy_path, result.Composite)
    save_composite_png(result.Composite, png_path, config)

    payload = result.to_jsonable()
    payload["Frames"] = [str(p) for p, _ in named_frames]
    payload["Config"] = asdict(config)

    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    return {"npy": npy_path, "png": png_path, "offsets": json_path}


def save_composite_png(
    composite: np.ndarray,
    path: Path,
    config: Optional[StitchConfig] = None,
) -> Path:
    # imsave writes directly to disk and does not require a GUI backend.
    import matplotlib.image as mpimg

    config = config or StitchConfig()

    lo, hi = np.percentile(composite, config.NormalizationPercentiles)

    if hi <= lo:
        hi = lo + 1.0

    mpimg.imsave(
        path,
        composite,
        cmap=config.Colormap,
        vmin=lo,
        vmax=hi,
    )

    return Path(path)