"""
Parametric camera mount for FLIR Blackfly S BFS-PGE-31S4 (29 x 29 x 30 mm case)
on the XYZ CNC gantry - laser beam imaging.

Parts generated:
  1. camera_cradle - U-channel cradle. The camera body (29 x 29 mm, known from
     the datasheet) registers laterally between the walls; it bolts down with
     2-3x M3x6 socket-head screws through a SLOT along the optical axis, so the
     exact spacing of FLIR's bottom M3 holes doesn't matter (they are on the
     case centerline). Flange has M4 slots matching the endstop-mount pattern
     philosophy: T-nuts, tapped holes, or through-bolts, with adjustability.
  2. tilt_wedge - optional shim that pitches the camera by TILT_DEG to kill
     etalon fringes from the sensor cover glass (coherent light). Print and
     insert under the cradle flange only if you see fringes.

Design notes:
 - Camera front face sits flush with the cradle front, so C-mount ND filter
   adapters (e.g. SM1, ~Ø30.5 mm) overhang in free air - no collision.
 - No front lip: the C-mount ring spans nearly the full 29 mm face, so any
   lip would collide with it.
 - Rear is open for the RJ45 + M8 GPIO connectors; rear flange strip has
   ziptie slots for ethernet strain relief (don't let cable flex torque the
   camera).
 - Side walls have windows and the top is open: the camera dissipates up to
   3 W - do not fully enclose it.
 - Ziptie slots near the wall tops let you strap the camera down as a
   backup / vibration damper.

All dimensions in mm. Re-run after edits:
    pip install build123d
    python camera_mount.py
"""

import math
import os
from build123d import (
    Box, Cylinder, Pos, Rot, SlotOverall, extrude, export_stl, export_step,
    Polyline, make_face, Plane,
)

# ----------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------

# --- Camera (Blackfly S 29 mm case, from datasheet) ---
CAM_W    = 29.0    # body width
CAM_L    = 30.0    # body length (front face to rear, excl. connectors)
CAM_FIT  = 0.4     # lateral clearance added to channel width
CAM_SLOT_W   = 3.4     # bolt slot width (M3 clearance)
CAM_SLOT_LEN = 24.0    # bolt slot length along optical axis

# --- Frame fastening (matches endstop mounts) ---
SLOT_W   = 4.4     # M4 clearance (3.4 for M3, 5.4 for M5)
SLOT_LEN = 16.0

# --- Gantry carriage interface (from the linear module drawing) ---
CARRIAGE_PITCH   = 18.0   # 4x M3 on an 18 x 18 mm square
CARRIAGE_HOLE_D  = 3.4    # M3 clearance
CARRIAGE_CB_D    = 6.5    # counterbore dia for M3 socket heads
CARRIAGE_CB_DEEP = 4.0    # counterbore depth (from camera floor, heads sit 1 mm below camera)

# --- Cradle geometry ---
FL_W     = 70.0    # flange width  (X, across the camera)
FL_L     = 40.0    # flange length (Y, along optical axis; front at -Y)
FL_TH    = 5.0     # flange thickness
WALL     = 5.0     # cradle wall thickness
WALL_H   = 18.0    # wall height above camera floor
WALL_LEN = 34.0    # wall length (front flush with flange front)
FLOOR_Z  = 9.0     # camera floor height (leaves room for screw heads below)
CBORE_W  = 7.0     # counterbore slot width for M3 socket heads (5.5 head + room)
CBORE_D  = 5.0     # counterbore depth from bottom (leaves 4 mm web -> use M3x6)

# --- Tilt wedge ---
TILT_DEG = 3.0     # pitch angle to kill sensor-window etalon fringes
WEDGE_T  = 3.0     # minimum wedge thickness (thin end)

ZIPTIE_W = 5.0
ZIPTIE_T = 2.0

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

CH_IN  = CAM_W + CAM_FIT                # channel inner width
BLK_W  = CH_IN + 2 * WALL               # cradle block outer width
WALL_X = CH_IN / 2 + WALL / 2           # wall center x
WALL_CY = -FL_L / 2 + WALL_LEN / 2      # wall center y (front flush)
M4_X   = BLK_W / 2 + (FL_W - BLK_W) / 4 # frame slot center x


def camera_cradle():
    # Flange
    part = Pos(0, 0, FL_TH / 2) * Box(FL_W, FL_L, FL_TH)

    # Raised camera floor block (front flush with flange front edge)
    part += Pos(0, WALL_CY, FLOOR_Z / 2) * Box(BLK_W, WALL_LEN, FLOOR_Z)

    # Side walls
    for wx in (-WALL_X, WALL_X):
        part += Pos(wx, WALL_CY, FLOOR_Z + WALL_H / 2) * Box(WALL, WALL_LEN, WALL_H)

    # Wall lightening / ventilation windows
    for wx in (-WALL_X, WALL_X):
        part -= Pos(wx, WALL_CY, FLOOR_Z + WALL_H / 2) * Box(WALL + 2, WALL_LEN - 14, WALL_H - 9)

    # Camera bolt slot (M3 clearance) through the floor, along Y
    part -= Pos(0, WALL_CY, FLOOR_Z / 2) * Rot(0, 0, 90) * extrude(
        SlotOverall(CAM_SLOT_LEN, CAM_SLOT_W), amount=FLOOR_Z / 2 + 1, both=True
    )
    # Counterbore slot for socket heads, from below
    part -= Pos(0, WALL_CY, CBORE_D / 2 - 0.01) * Rot(0, 0, 90) * extrude(
        SlotOverall(CAM_SLOT_LEN, CBORE_W), amount=CBORE_D / 2 + 0.01, both=True
    )

    # Carriage bolt pattern: 4x M3 clearance on 18 x 18 mm square, through
    # the camera floor, counterbored from the top so the heads sit below the
    # camera. Install these screws BEFORE dropping the camera in.
    p = CARRIAGE_PITCH / 2
    for hx in (-p, p):
        for hy in (WALL_CY - p, WALL_CY + p):
            part -= Pos(hx, hy, FLOOR_Z / 2) * Cylinder(CARRIAGE_HOLE_D / 2, FLOOR_Z + 2)
            part -= Pos(hx, hy, FLOOR_Z - CARRIAGE_CB_DEEP / 2 + 0.01) * Cylinder(
                CARRIAGE_CB_D / 2, CARRIAGE_CB_DEEP
            )

    # Frame slots (M4) in the side flanges, along Y
    for sx in (-M4_X, M4_X):
        part -= Pos(sx, 0, FL_TH / 2) * Rot(0, 0, 90) * extrude(
            SlotOverall(SLOT_LEN, SLOT_W), amount=FL_TH / 2 + 1, both=True
        )

    # Ziptie slots near wall tops (strap over the camera body).
    # Located in the solid end posts, clear of the ventilation windows.
    zt_z = FLOOR_Z + WALL_H - 3.0
    for wx in (-WALL_X, WALL_X):
        for zy in (WALL_CY - 13.0, WALL_CY + 13.0):
            part -= Pos(wx, zy, zt_z) * Box(WALL + 2, ZIPTIE_W, ZIPTIE_T)

    # Ziptie slots in the rear flange strip (ethernet strain relief)
    rear_y = FL_L / 2 - 3.0
    for zx in (-8.0, 8.0):
        part -= Pos(zx, rear_y, FL_TH / 2) * Box(ZIPTIE_W, ZIPTIE_T, FL_TH + 2)

    return part


def carriage_adapter():
    """90-degree bracket: bolts to the carriage's 18 x 18 M3 pattern when the
    carriage face is VERTICAL, providing a horizontal shelf that the camera
    cradle's M4 flange slots bolt onto (M4 screws + nuts/washers underneath).
    Carriage face is at y = 0; shelf extends toward +Y (away from the module,
    so the moving carriage never fouls the bracket)."""
    LEG_W, LEG_H, LEG_T = 40.0, 40.0, 6.0
    SHELF_W, SHELF_D, SHELF_T = 66.0, 28.0, 6.0

    # Vertical leg against the carriage
    part = Pos(0, LEG_T / 2, LEG_H / 2) * Box(LEG_W, LEG_T, LEG_H)
    # Horizontal shelf, top flush with leg top
    part += Pos(0, SHELF_D / 2, LEG_H - SHELF_T / 2) * Box(SHELF_W, SHELF_D, SHELF_T)

    # Gussets under the shelf (clear of the 30 mm wide module and the M4 nuts)
    for gx in (-18.0, 15.0):
        tri = Plane.YZ * Polyline(
            (LEG_T, LEG_H - SHELF_T),
            (LEG_T, LEG_H - SHELF_T - 12.0),
            (LEG_T + 14.0, LEG_H - SHELF_T),
            (LEG_T, LEG_H - SHELF_T),
        )
        part += Pos(gx, 0, 0) * extrude(make_face(tri), amount=3.0)

    # Carriage bolt pattern (M3 clearance, horizontal, through the leg)
    p = CARRIAGE_PITCH / 2
    for hx in (-p, p):
        for hz in (LEG_H / 2 - SHELF_T - p + 2.0, LEG_H / 2 - SHELF_T + p + 2.0):
            part -= Pos(hx, LEG_T / 2, hz) * Rot(90, 0, 0) * Cylinder(
                CARRIAGE_HOLE_D / 2, LEG_T + 2
            )

    # M4 holes in the shelf, matching the cradle flange slot spacing
    for sx in (-M4_X, M4_X):
        part -= Pos(sx, SHELF_D / 2, LEG_H - SHELF_T / 2) * Cylinder(
            SLOT_W / 2, SHELF_T + 2
        )
    return part


def tilt_wedge():
    rise = FL_L * math.tan(math.radians(TILT_DEG))
    profile = Plane.YZ * Polyline(
        (-FL_L / 2, 0), (FL_L / 2, 0),
        (FL_L / 2, WEDGE_T + rise), (-FL_L / 2, WEDGE_T),
        (-FL_L / 2, 0),
    )
    part = extrude(make_face(profile), amount=FL_W / 2, both=True)

    # Matching M4 slots
    for sx in (-M4_X, M4_X):
        part -= Pos(sx, 0, (WEDGE_T + rise) / 2) * Rot(0, 0, 90) * extrude(
            SlotOverall(SLOT_LEN, SLOT_W), amount=WEDGE_T + rise, both=True
        )
    return part


if __name__ == "__main__":
    parts = {
        "camera_cradle": camera_cradle(),
        "carriage_adapter": carriage_adapter(),
        "tilt_wedge": tilt_wedge(),
    }
    for name, p in parts.items():
        export_stl(p, os.path.join(OUT, f"{name}.stl"))
        export_step(p, os.path.join(OUT, f"{name}.step"))
        print(f"exported {name}: volume={p.volume:.0f} mm^3, bbox={p.bounding_box().size}")

    # Dummy camera body for the assembly preview
    cam = Pos(0, WALL_CY + (WALL_LEN - CAM_L) / 2 - (WALL_LEN - CAM_L) / 2 * 0 - 2.0, FLOOR_Z + CAM_W / 2)
    dummy = Pos(0, -FL_L / 2 + CAM_L / 2, FLOOR_Z + CAM_W / 2) * Box(CAM_W, CAM_L, CAM_W)
    export_stl(dummy, os.path.join(OUT, "_camera_dummy.stl"))
    print("done")
