from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Vec3D:
    x_mm: float
    y_mm: float
    z_mm: float


@dataclass(frozen=True)
class AxisRange:
    start_mm: float
    stop_mm: float
    step_mm: float

    def values(self) -> list[float]:
        """
        Positions from start to stop inclusive. Decreasing ranges
        (stop < start) walk downward; step_mm's sign is ignored — the
        direction comes from start/stop. (Descending Y scans let a
        placement bootstrap near the optic, where a diverging beam is
        smallest and brightest, then track it outward.)
        """

        step = abs(self.step_mm)
        if step <= 0:
            return [round(self.start_mm, 6)]

        direction = 1.0 if self.stop_mm >= self.start_mm else -1.0

        vals = []
        x = self.start_mm
        while direction * (self.stop_mm - x) >= -1e-9:
            vals.append(round(x, 6))
            x += direction * step

        return vals


@dataclass(frozen=True)
class Bounds3D:
    """
    Full safe travel range of the CNC gantry, in gantry-local coordinates.
    """
    x_min_mm: float
    x_max_mm: float
    y_min_mm: float
    y_max_mm: float
    z_min_mm: float
    z_max_mm: float

    def contains(self, p: Vec3D) -> bool:
        return (
            self.x_min_mm <= p.x_mm <= self.x_max_mm
            and self.y_min_mm <= p.y_mm <= self.y_max_mm
            and self.z_min_mm <= p.z_mm <= self.z_max_mm
        )


@dataclass(frozen=True)
class ScanPoint:
    """
    One camera acquisition point.

    GantryPosition_mm is used for stepper motor movement.
    TablePosition_mm is used for beam profile reconstruction.
    """

    PlacementID: str
    GantryPosition_mm: Vec3D
    TablePosition_mm: Vec3D
    NShots: int = 1
    Metadata: dict[str, Any] = field(default_factory=dict)