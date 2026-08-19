"""
Adaptive XY raster: grow the imaged rectangle until its edges are dark.

At each z-slice the beam's cross-section size is unknown a priori (small
near the axicon, large far away). Instead of rastering a fixed rectangle
sized for the worst case, this runner:

    1. Captures a single seed frame at the raster center (the calibration
       point, which is known to contain the beam).
    2. Repeatedly examines the frames on each edge of the current
       rectangle. If a frame's outer border strips contain signal above a
       background-referenced threshold, the beam extends past the frame,
       so the rectangle grows one grid step in that direction and only the
       new cells are captured.
    3. Stops when all four edges are dark, or when an edge hits the
       configured cap (the --x/--y ranges) — a cap stop is loudly logged
       and recorded, because it means coverage was truncated.

If the beam fits entirely inside the seed frame (all four border strips
dark), the raster is complete after ONE frame.

Border test:

A frame "has signal at its border" if ANY of its four border strips does,
and that one answer drives growth in all four directions. This is
orientation-independent — it does not matter how the camera's image axes
map to machine axes — at the cost of overshooting the beam extent by up
to ~1 frame per direction. Those surplus frames are dark on every strip,
so they are labeled and excluded from composites by default.

The runner is decoupled from the camera and gantry: it drives a capture
callback and reads back NumPy arrays, so it is unit tested against a
synthetic beam field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)

DIRECTIONS = ("+x", "-x", "+y", "-y")


class AdaptiveRasterError(RuntimeError):
    pass


@dataclass(frozen=True)
class AdaptiveRasterConfig:
    # Grid lattice: cell (i, j) sits at (CenterX + i*StepX, CenterY + j*StepY).
    CenterX_mm: float
    CenterY_mm: float
    StepX_mm: float
    StepY_mm: float

    # Hard caps on camera-center positions (mm, machine coords). The lattice
    # never grows past these.
    XMin_mm: float = float("-inf")
    XMax_mm: float = float("inf")
    YMin_mm: float = float("-inf")
    YMax_mm: float = float("inf")

    # A border strip "has signal" when at least MinSignalPixels pixels
    # exceed SignalThreshold_counts (typically background p99 + margin).
    SignalThreshold_counts: float = 13.0
    MinSignalPixels: int = 50

    # Border strip width as a fraction of the frame dimension.
    BorderStripFraction: float = 0.15


    # If the SEED frame contains no signal (e.g. the seed landed in the
    # dark interior of a ring beam), grow in all directions for up to this
    # many passes anyway, hunting for the beam. Once any captured cell has
    # signal, the normal edge rule takes over; if nothing is found the
    # raster stops and reports BeamFound=false.
    BlindProbePasses: int = 2


@dataclass
class CellResult:
    I: int
    J: int
    X_mm: float
    Y_mm: float
    Records: list
    BorderSignal: dict  # side -> bool
    AnySignal: bool


@dataclass
class AdaptiveRasterResult:
    Records: list
    Cells: list  # CellResult, in capture order
    Metadata: dict


# CaptureFn(x_mm, y_mm, i, j) -> (records, frame_array). The array is the
# frame used for the signal test (first shot if several were taken).
CaptureFn = Callable[[float, float, int, int], tuple[list, np.ndarray]]


def border_strips(
    arr: np.ndarray, fraction: float
) -> dict[str, np.ndarray]:
    """
    The four border strips of a frame in ARRAY orientation:
    'row0' (first rows), 'row1' (last rows), 'col0', 'col1'.
    """

    rows = max(1, int(round(arr.shape[0] * fraction)))
    cols = max(1, int(round(arr.shape[1] * fraction)))

    return {
        "row0": arr[:rows, :],
        "row1": arr[-rows:, :],
        "col0": arr[:, :cols],
        "col1": arr[:, -cols:],
    }


class AdaptiveRasterRunner:
    def __init__(
        self,
        config: AdaptiveRasterConfig,
        capture_fn: CaptureFn,
    ):
        self.config = config
        self.capture_fn = capture_fn

        self.cells: dict[tuple[int, int], CellResult] = {}
        self.capture_order: list[tuple[int, int]] = []
        self._strip_size_checked = False

    # -- lattice geometry ----------------------------------------------

    def cell_xy(self, i: int, j: int) -> tuple[float, float]:
        return (
            self.config.CenterX_mm + i * self.config.StepX_mm,
            self.config.CenterY_mm + j * self.config.StepY_mm,
        )

    def _index_caps(self) -> tuple[int, int, int, int]:
        """Largest i/j extents whose positions stay inside the caps."""

        c = self.config

        i_min = -int(np.floor((c.CenterX_mm - c.XMin_mm) / c.StepX_mm + 1e-9))
        i_max = int(np.floor((c.XMax_mm - c.CenterX_mm) / c.StepX_mm + 1e-9))
        j_min = -int(np.floor((c.CenterY_mm - c.YMin_mm) / c.StepY_mm + 1e-9))
        j_max = int(np.floor((c.YMax_mm - c.CenterY_mm) / c.StepY_mm + 1e-9))

        if i_min > 0 or i_max < 0 or j_min > 0 or j_max < 0:
            raise AdaptiveRasterError(
                "Raster center is outside the configured X/Y caps."
            )

        return i_min, i_max, j_min, j_max

    # -- signal test ---------------------------------------------------

    def _strip_has_signal(self, strip: np.ndarray) -> bool:
        above = int(np.sum(strip > self.config.SignalThreshold_counts))
        return above >= self.config.MinSignalPixels

    def _classify_borders(self, arr: np.ndarray) -> dict[str, bool]:
        """Per machine-side border-signal flags for one frame."""

        strips = border_strips(arr, self.config.BorderStripFraction)

        if not self._strip_size_checked:
            self._strip_size_checked = True
            largest = max(strip.size for strip in strips.values())

            if largest < self.config.MinSignalPixels:
                logger.warning(
                    f"MinSignalPixels ({self.config.MinSignalPixels}) exceeds "
                    f"every border-strip size (largest {largest} px): the "
                    "raster can NEVER grow. Lower MinSignalPixels or raise "
                    "BorderStripFraction."
                )
        strip_signal = {
            name: self._strip_has_signal(strip) for name, strip in strips.items()
        }

        any_signal = any(strip_signal.values())
        return {side: any_signal for side in DIRECTIONS}

    # -- capture -------------------------------------------------------

    def _capture_cell(self, i: int, j: int) -> CellResult:
        x_mm, y_mm = self.cell_xy(i, j)
        records, arr = self.capture_fn(x_mm, y_mm, i, j)

        cell = CellResult(
            I=i,
            J=j,
            X_mm=x_mm,
            Y_mm=y_mm,
            Records=records,
            BorderSignal=self._classify_borders(np.asarray(arr)),
            AnySignal=bool(
                np.sum(np.asarray(arr) > self.config.SignalThreshold_counts)
                >= self.config.MinSignalPixels
            ),
        )

        self.cells[(i, j)] = cell
        self.capture_order.append((i, j))
        return cell

    # -- growth loop ---------------------------------------------------

    @staticmethod
    def _edge_cells(rect: list[int], side: str) -> list[tuple[int, int]]:
        i0, i1, j0, j1 = rect

        if side == "+x":
            return [(i1, j) for j in range(j0, j1 + 1)]
        if side == "-x":
            return [(i0, j) for j in range(j0, j1 + 1)]
        if side == "+y":
            return [(i, j1) for i in range(i0, i1 + 1)]
        return [(i, j0) for i in range(i0, i1 + 1)]

    @staticmethod
    def _new_cells_after_growth(rect: list[int], side: str) -> list[tuple[int, int]]:
        i0, i1, j0, j1 = rect

        if side == "+x":
            return [(i1 + 1, j) for j in range(j0, j1 + 1)]
        if side == "-x":
            return [(i0 - 1, j) for j in range(j0, j1 + 1)]
        if side == "+y":
            return [(i, j1 + 1) for i in range(i0, i1 + 1)]
        return [(i, j0 - 1) for i in range(i0, i1 + 1)]

    @staticmethod
    def _grow(rect: list[int], side: str) -> None:
        if side == "+x":
            rect[1] += 1
        elif side == "-x":
            rect[0] -= 1
        elif side == "+y":
            rect[3] += 1
        else:
            rect[2] -= 1

    def _at_cap(self, rect: list[int], side: str, caps) -> bool:
        i_min, i_max, j_min, j_max = caps
        i0, i1, j0, j1 = rect

        return {
            "+x": i1 >= i_max,
            "-x": i0 <= i_min,
            "+y": j1 >= j_max,
            "-y": j0 <= j_min,
        }[side]

    def run(self) -> AdaptiveRasterResult:
        caps = self._index_caps()
        rect = [0, 0, 0, 0]  # i0, i1, j0, j1

        self._capture_cell(0, 0)

        iterations: list[dict[str, Any]] = []
        edge_stops: dict[str, str] = {}
        blind_passes_used = 0

        while True:
            beam_found = any(cell.AnySignal for cell in self.cells.values())
            blind_probe = (
                not beam_found and blind_passes_used < self.config.BlindProbePasses
            )

            if not beam_found and not blind_probe:
                # Seed was dark and the probe budget is spent: nothing here.
                edge_stops = {side: "dark" for side in DIRECTIONS}
                logger.warning(
                    "Adaptive raster found NO beam signal: the seed frame "
                    f"and {blind_passes_used} blind probe pass(es) around it "
                    "are all dark. Check the calibration/seed position."
                )
                break

            grew_any = False
            pass_report: dict[str, Any] = {
                "Pass": len(iterations) + 1,
                "BlindProbe": blind_probe,
                "Grew": [],
                "Stopped": {},
                "CellsAdded": [],
            }

            for side in DIRECTIONS:
                edge = self._edge_cells(rect, side)
                edge_signal = any(
                    self.cells[cell].BorderSignal[side] for cell in edge
                )

                # Blind probe: grow outward regardless of (dark) borders,
                # hunting for a beam the seed missed.
                if not edge_signal and not blind_probe:
                    pass_report["Stopped"][side] = "dark"
                    continue

                if self._at_cap(rect, side, caps):
                    pass_report["Stopped"][side] = "cap"
                    continue

                new_cells = self._new_cells_after_growth(rect, side)
                self._grow(rect, side)

                for i, j in new_cells:
                    if (i, j) not in self.cells:
                        cell = self._capture_cell(i, j)
                        pass_report["CellsAdded"].append(
                            [i, j, cell.X_mm, cell.Y_mm]
                        )

                pass_report["Grew"].append(side)
                grew_any = True

            if blind_probe:
                blind_passes_used += 1

            iterations.append(pass_report)

            if not grew_any:
                edge_stops = dict(pass_report["Stopped"])
                break

        # -- summarize --------------------------------------------------

        truncated_sides = [s for s, why in edge_stops.items() if why == "cap"]

        for side in truncated_sides:
            logger.warning(
                f"ADAPTIVE RASTER TRUNCATED on {side}: the beam still had "
                "signal at the configured X/Y cap. Widen the cap (or accept "
                "reduced coverage)."
            )

        i0, i1, j0, j1 = rect
        i_min, i_max, j_min, j_max = caps
        fixed_grid_cells = (i_max - i_min + 1) * (j_max - j_min + 1)

        cells_in_order = [self.cells[key] for key in self.capture_order]
        records = [r for cell in cells_in_order for r in cell.Records]

        beam_found = any(cell.AnySignal for cell in cells_in_order)
        single_frame = len(cells_in_order) == 1 and beam_found

        if single_frame:
            logger.info(
                "Beam fits entirely within a single camera frame at this "
                "z-slice; raster complete after 1 position."
            )

        x0, y0 = self.cell_xy(i0, j0)
        x1, y1 = self.cell_xy(i1, j1)

        metadata = {
            "RasterMode": "adaptive",
            "Center_mm": [self.config.CenterX_mm, self.config.CenterY_mm],
            "Step_mm": [self.config.StepX_mm, self.config.StepY_mm],
            "Cap_mm": {
                "XMin": self.config.XMin_mm,
                "XMax": self.config.XMax_mm,
                "YMin": self.config.YMin_mm,
                "YMax": self.config.YMax_mm,
            },
            "SignalThreshold_counts": self.config.SignalThreshold_counts,
            "MinSignalPixels": self.config.MinSignalPixels,
            "BorderStripFraction": self.config.BorderStripFraction,
            "Iterations": iterations,
            "EdgeStops": edge_stops,
            "TruncatedSides": truncated_sides,
            "GridShape": [i1 - i0 + 1, j1 - j0 + 1],
            "FinalRect_mm": {"XMin": x0, "XMax": x1, "YMin": y0, "YMax": y1},
            "CellsCaptured": len(cells_in_order),
            "FixedGridCells": fixed_grid_cells,
            "BeamFitsInSingleFrame": single_frame,
            "BeamFound": beam_found,
            "BlindProbePassesUsed": blind_passes_used,
            "Cells": [
                {
                    "I": cell.I,
                    "J": cell.J,
                    "X_mm": cell.X_mm,
                    "Y_mm": cell.Y_mm,
                    "AnySignal": cell.AnySignal,
                    "BorderSignal": cell.BorderSignal,
                    "Paths": [
                        getattr(r, "Path", None) for r in cell.Records
                    ],
                }
                for cell in cells_in_order
            ],
        }

        return AdaptiveRasterResult(
            Records=records, Cells=cells_in_order, Metadata=metadata
        )
