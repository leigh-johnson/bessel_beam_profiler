"""
Parametric endstop mounts for KW11-3Z-style lever microswitches
(e.g. DAOKI 20-pack) on a small CNC gantry.

Three parts are generated:
  1. L-mount      - switch stands vertical on a wall above a slotted base.
                    Most versatile: bolts to T-slot extrusion (T-nuts),
                    tapped holes, or through-bolts. Slots allow ~8 mm of
                    position adjustment.
  2. Flat mount   - switch lies flat on a plate; lower profile.
  3. Striker flag - adjustable L-bracket that bolts to the moving carriage
                    and trips the switch lever. Slotted for fine-tuning
                    the trigger point.

VERIFY WITH CALIPERS when your switches arrive, then re-run:
    pip install build123d
    python endstop_mounts.py

All dimensions in mm.
"""

from build123d import (
    Box, Cylinder, Pos, Rot, SlotOverall, extrude, export_stl, export_step,
    Polyline, make_face, Plane, Axis,
)
import os

# ----------------------------------------------------------------------
# PARAMETERS - edit these, then re-run the script
# ----------------------------------------------------------------------

# --- Switch (KW11-3Z defaults; MEASURE YOURS) ---
SW_LEN        = 20.0   # switch body length
SW_HT         = 10.3   # switch body height (bottom edge to top, excl. lever)
SW_TH         = 6.4    # switch body thickness
SW_HOLE_SPACE = 9.5    # mounting hole center-to-center spacing
SW_HOLE_UP    = 2.9    # hole center height above switch bottom edge
SW_PILOT_D    = 1.8    # pilot hole for M2 self-tapping screw
PIN_CLEAR     = 6.0    # free space below switch bottom edge for solder pins

# --- Frame fastening ---
SLOT_W        = 4.4    # base slot width: 4.4 fits M4 (use 3.4 for M3, 5.4 for M5)
SLOT_LEN      = 10.0   # overall slot length (adjustability)

# --- General ---
BASE_TH       = 4.0    # base plate thickness
WALL_TH       = 6.0    # switch wall thickness (L-mount)
ZIPTIE_W      = 4.0    # ziptie slot width  (fallback fastening + strain relief)
ZIPTIE_T      = 2.0    # ziptie slot thickness

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)


def base_slots(base_len, y=0.0, th=BASE_TH):
    """Two through-slots for frame screws, running along X."""
    cuts = []
    x_off = base_len / 2 - SLOT_LEN / 2 - 3.0
    for sx in (-x_off, x_off):
        cuts.append(
            Pos(sx, y, th / 2)
            * extrude(SlotOverall(SLOT_LEN, SLOT_W), amount=th / 2, both=True)
        )
    return cuts


# ----------------------------------------------------------------------
# PART 1: L-MOUNT
# ----------------------------------------------------------------------
def l_mount():
    base_len = SW_LEN + 16.0          # 36
    base_w   = 18.0
    wall_len = SW_LEN + 10.0          # 30
    wall_h   = BASE_TH + PIN_CLEAR + SW_HT + 1.0   # ~21.3

    # Base plate, top face at z = BASE_TH
    part = Pos(0, 0, BASE_TH / 2) * Box(base_len, base_w, BASE_TH)

    # Vertical wall along rear edge (front face at y = wall_y_front)
    wall_y = base_w / 2 - WALL_TH / 2
    part += Pos(0, wall_y, wall_h / 2) * Box(wall_len, WALL_TH, wall_h)

    # Base slots
    for c in base_slots(base_len):
        part -= c

    # Switch pilot holes (horizontal, through wall)
    sw_bottom = BASE_TH + PIN_CLEAR
    hole_z = sw_bottom + SW_HOLE_UP
    for hx in (-SW_HOLE_SPACE / 2, SW_HOLE_SPACE / 2):
        part -= Pos(hx, wall_y, hole_z) * Rot(90, 0, 0) * Cylinder(
            SW_PILOT_D / 2, WALL_TH + 2
        )

    # Ziptie slots flanking the switch position (vertical slots through wall)
    zt_z = sw_bottom + SW_HT / 2
    for zx in (-(SW_LEN / 2 + 3.0), SW_LEN / 2 + 3.0):
        part -= Pos(zx, wall_y, zt_z) * Box(ZIPTIE_T, WALL_TH + 2, ZIPTIE_W + 2)

    return part


# ----------------------------------------------------------------------
# PART 2: FLAT MOUNT
# ----------------------------------------------------------------------
def flat_mount():
    plate_th  = 5.0
    plate_len = SW_LEN + 16.0                # 36 (X)
    plate_w   = SW_TH + 2 * (SLOT_W + 8.0)   # room for slots each side (Y)

    part = Pos(0, 0, plate_th / 2) * Box(plate_len, plate_w, plate_th)

    # Switch lies flat with its pin end flush with the -X plate edge, so
    # the solder pins overhang in free air. Shallow locating recess, open
    # at the plate edge.
    recess_d = 1.2
    sw_cx = -(plate_len / 2) + SW_LEN / 2  # switch center x
    part -= Pos(sw_cx - 0.5, 0, plate_th - recess_d / 2 + 0.01) * Box(
        SW_LEN + 1.4, SW_TH + 0.4, recess_d
    )

    # Switch pilot holes (vertical, symmetric about switch body center)
    for hx in (sw_cx - SW_HOLE_SPACE / 2, sw_cx + SW_HOLE_SPACE / 2):
        part -= Pos(hx, 0, plate_th / 2) * Cylinder(SW_PILOT_D / 2, plate_th + 2)

    # Frame slots either side of the switch, running along X
    y_off = SW_TH / 2 + 0.2 + 4.0 + SLOT_W / 2
    for sy in (-y_off, y_off):
        part -= Pos(4.0, sy, plate_th / 2) * extrude(
            SlotOverall(SLOT_LEN + 6, SLOT_W), amount=plate_th / 2 + 1, both=True
        )

    # Ziptie slots across the switch body
    for zx in (sw_cx - 6.0, sw_cx + 6.0):
        part -= Pos(zx, SW_TH / 2 + 1.6, plate_th / 2) * Box(ZIPTIE_T, ZIPTIE_W - 1.0, plate_th + 2)
        part -= Pos(zx, -(SW_TH / 2 + 1.6), plate_th / 2) * Box(ZIPTIE_T, ZIPTIE_W - 1.0, plate_th + 2)

    return part


# ----------------------------------------------------------------------
# PART 3: STRIKER FLAG
# ----------------------------------------------------------------------
def striker_flag():
    th     = 4.0    # material thickness
    width  = 12.0   # extrusion width (Y)
    arm_h  = 26.0   # vertical arm height
    foot_l = 18.0   # horizontal foot length

    # L cross-section in XZ plane, extruded along Y
    profile = Plane.XZ * Polyline(
        (0, 0), (foot_l, 0), (foot_l, th), (th, th), (th, arm_h), (0, arm_h), (0, 0)
    )
    part = extrude(make_face(profile), amount=width / 2, both=True)

    # Slot in the foot (adjust along mounting screw)
    part -= Pos(foot_l / 2 + th / 2, 0, th / 2) * extrude(
        SlotOverall(foot_l - th - 6.0, SLOT_W), amount=th, both=True
    )

    # Slot in the vertical arm (height-adjust if bolted instead)
    part -= (
        Pos(th / 2, 0, arm_h / 2 + th / 2)
        * Rot(90, 0, 90)
        * extrude(SlotOverall(arm_h - th - 8.0, SLOT_W), amount=th, both=True)
    )

    return part


# ----------------------------------------------------------------------
# Build + export
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parts = {
        "endstop_L_mount": l_mount(),
        "endstop_flat_mount": flat_mount(),
        "striker_flag": striker_flag(),
    }
    for name, p in parts.items():
        stl = os.path.join(OUT, f"{name}.stl")
        step = os.path.join(OUT, f"{name}.step")
        export_stl(p, stl)
        export_step(p, step)
        print(f"exported {name}: volume={p.volume:.0f} mm^3, bbox={p.bounding_box().size}")
    print("done")
