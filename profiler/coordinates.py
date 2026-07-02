from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class Vec2D:
    x_mm: float
    y_mm: float


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
        vals = []
        x = self.start_mm

        # Handles increasing ranges.
        # TODO? Add decreasing later if needed.
        while x <= self.stop_mm + 1e-9:
            vals.append(round(x, 6))
            x += self.step_mm

        return vals


@dataclass(frozen=True)
class Bounds2D:
    """
    A rectangular region of interest in gantry-local coordinates.
    """
    x_min_mm: float
    x_max_mm: float
    y_min_mm: float
    y_max_mm: float

    def contains(self, p: Vec2D) -> bool:
        return (
            self.x_min_mm <= p.x_mm <= self.x_max_mm
            and self.y_min_mm <= p.y_mm <= self.y_max_mm
        )


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
class GantryPlacement:
    """
    One physical placement of the CNC gantry on the optics table.

    These values are manually entered after moving the gantry to a new position and measuring where the gantry-local (0, 0, 0) point lands in the lab/beamline coordinate frame.
    """

    PlacementID: str

    # Where gantry-local (0, 0, 0) lands in your lab/beamline coordinate frame.
    TableOrigin_mm: Vec3D

    # Human-readable notes:
    # e.g. "gantry moved downstream, front left corner aligned with tape mark B", positioning square used / not used, etc.
    Notes: str = ""

    def gantry_to_table(self, p: Vec3D) -> Vec3D:
        return Vec3D(
            x_mm=self.TableOrigin_mm.x_mm + p.x_mm,
            y_mm=self.TableOrigin_mm.y_mm + p.y_mm,
            z_mm=self.TableOrigin_mm.z_mm + p.z_mm,
        )


@dataclass(frozen=True)
class XYCrossSectionPlan:
    """
    One XY image cross-section at one local gantry Z.
    """

    Placement: GantryPlacement
    MachineLimits: Bounds3D
    # gantry ROI, not camera ROI.
    ROI: Bounds2D

    # Local Z in the gantry frame.
    GantryZ_mm: float

    X: AxisRange
    Y: AxisRange

    NShots: int = 1
    Metadata: dict[str, Any] = field(default_factory=dict)

    def generate_points(self) -> list["ScanPoint"]:
        points: list[ScanPoint] = []

        for y_mm in self.Y.values():
            for x_mm in self.X.values():
                gantry_position = Vec3D(x_mm=x_mm, y_mm=y_mm, z_mm=self.GantryZ_mm)

                if not self.MachineLimits.contains(gantry_position):
                    raise ValueError(f"Point outside machine limits: {gantry_position}")

                if not self.ROI.contains(Vec2D(x_mm=x_mm, y_mm=y_mm)):
                    continue

                table_position = self.Placement.gantry_to_table(gantry_position)

                points.append(
                    ScanPoint(
                        PlacementID=self.Placement.PlacementID,
                        GantryPosition_mm=gantry_position,
                        TablePosition_mm=table_position,
                        NShots=self.NShots,
                        Metadata=self.Metadata,
                    )
                )

        return points


@dataclass(frozen=True)
class ZStackPlan:
    """
    Multiple XY cross-sections within one physical gantry placement.

    This scans the full XY cross-section first, then moves to the next Z.
    """

    Placement: GantryPlacement
    MachineLimits: Bounds3D
    ROI: Bounds2D

    X: AxisRange
    Y: AxisRange
    Z: AxisRange

    NShots: int = 1
    Metadata: dict[str, Any] = field(default_factory=dict)

    def generate_points(self) -> list["ScanPoint"]:
        points: list[ScanPoint] = []

        # Outer loop is Z; inner loops are Y/X.
        # This means: complete an entire XY cross-section before moving Z.
        for z_mm in self.Z.values():
            section = XYCrossSectionPlan(
                Placement=self.Placement,
                MachineLimits=self.MachineLimits,
                ROI=self.ROI,
                GantryZ_mm=z_mm,
                X=self.X,
                Y=self.Y,
                NShots=self.NShots,
                Metadata={
                    **self.Metadata,
                    "ScanKind": "ZStack",
                },
            )
            points.extend(section.generate_points())

        return points


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