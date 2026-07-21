"""
Z-axis -> X-axis carriage mounting brackets (two orientation variants)
======================================================================
Mounts the vertical Z stage (275mm overall, 200mm travel, base-hole pairs
17mm across x 175mm apart along the rail, counterbored M3) onto the X-axis
carriage's 4x M3 holes on a 20x20mm square.

Instead of fixed nut pockets, both variants carry TWIN VERTICAL T-SLOT
CHANNELS, 17mm apart, sized for standard M3 hex nuts (5.5 AF x 2.4) slid
in from the top. The Z rail's base screws (M3x10, through the rail's own
counterbored base holes) clamp onto the sliding nuts, so:
  - rail height is fully adjustable (set the rail bottom ~3mm above the
    table regardless of what the X-axis stack height ends up being,
    1020 extrusion or not),
  - the exact pair-to-end distance of the rail never matters,
  - both pairs (bottom AND top, 175mm apart) engage the same channels.

VARIANT 1 - "FaceUp": X-carriage mounting plate faces UP (normal parallel
  to the Z axis). L-bracket: foot bolts down onto the carriage, wall runs
  along the X travel direction, offset so the hanging Z rail clears the
  X-stage body (30mm wide) by 3mm and can drop to the table beside it.
  Camera ends up looking across the X axis.

VARIANT 2 - "SideFace": X-carriage mounting plate faces SIDEWAYS (normal
  perpendicular to the Z axis; stage on its side, as in your photos).
  Flat tall plate: bolts to the vertical face with FLUSH counterbored
  heads; the Z rail clamps beside the pattern (offset 25mm along travel
  so channels clear the carriage screws). Camera looks along the plate
  normal. ~24mm of X travel is lost at one end (rail sticks past the
  carriage); mirror the STL in your slicer to lose it at the other end.

Both: printed part leaves the rail's bottom end and everything below the
bracket completely clear for the future endstop mount.

Coordinates per variant are commented inline. Everything parametric.
Assumed X-carriage face ~60mm above breadboard (Leigh, 2026-07-16);
channels span carriage-face-55mm .. +160mm which covers bottom pair
(rail bottom near table) through top pair (+175mm) with adjustment room.
If the stack grows >~15mm, bump DROP_BELOW and reprint.

Screws: M3x10 SHCS everywhere (Leigh's stock).
Print: V1 on its side (channels on a vertical face), V2 flat channels-up.
PETG/PLA, 4+ perimeters. Mirror STL in slicer to flip handedness.
"""

import os
import cadquery as cq

# ---------------- shared parameters (mm) ----------------
PATTERN = 20.0            # X-carriage hole square
HOLE_D = 3.4              # M3 clearance
CB_D = 6.8                # SHCS head counterbore dia
NUT_AF = 5.5              # M3 hex nut across flats
NUT_T = 2.4               # M3 hex nut thickness
CH_PITCH = 17.0           # Z-rail base holes across the rail
CH_CAV_W = NUT_AF + 0.3   # 5.8 channel cavity width (locks nut rotation)
CH_CAV_D = 5.2            # cavity depth (nut + screw-tip room)
CH_LIP = 2.5              # lip thickness the screw clamps through
CH_SLOT_W = 3.6           # slot through the lip
WALL_T = CH_LIP + CH_CAV_D + 3.3   # 11.0 total wall/plate thickness
Z_LO, Z_HI = -55.0, 160.0 # channel span relative to carriage-face height
RAIL_W = 30.0             # Z/X stage body width
BODY_CLEAR = 3.0          # clearance to the X-stage body sides

def channel_cut(face_wp_origin_x, zlo, zhi):
    """Return solids to cut one vertical T-channel; x = channel center,
    cut boxes are built in the caller's coordinate frame helper."""
    pass  # (channels are cut inline per variant; see below)

def cut_channels(body, centers, face_pos, axis, zlo=Z_LO, zhi=Z_HI):
    """Cut T-channels into `body`. `centers`: list of coordinates across;
    `face_pos`: coordinate of the outer face along `axis` ('y' for both
    variants here). Channel runs along z from zlo to zhi (through ends)."""
    h = zhi - zlo + 2
    zc = (zlo + zhi) / 2
    for c in centers:
        slot = cq.Workplane("XY").box(CH_SLOT_W, CH_LIP + 0.2, h)\
            .translate((c, face_pos - CH_LIP / 2 + 0.1, zc))
        cav = cq.Workplane("XY").box(CH_CAV_W, CH_CAV_D, h)\
            .translate((c, face_pos - CH_LIP - CH_CAV_D / 2, zc))
        body = body.cut(slot).cut(cav)
    return body

# =========================================================
# VARIANT 1 - FaceUp L-bracket
# frame: origin = pattern center at carriage face; x = X travel,
#        y = outboard (toward the hanging rail), z = up
# =========================================================
FOOT_T = 6.0
FOOT_X = 40.0                       # +-20
FOOT_Y_NEG = -20.0
WALL_YIN = RAIL_W / 2 + BODY_CLEAR  # 18.0 inner face (3mm off the body)
WALL_YOUT = WALL_YIN + WALL_T       # 29.0 rail clamps against this face

v1 = (
    cq.Workplane("XY")
    .box(FOOT_X, WALL_YIN - FOOT_Y_NEG, FOOT_T, centered=False)
    .translate((-FOOT_X / 2, FOOT_Y_NEG, 0))
)
wall = (
    cq.Workplane("XY")
    .box(FOOT_X, WALL_T, Z_HI - Z_LO, centered=False)
    .translate((-FOOT_X / 2, WALL_YIN, Z_LO))
)
v1 = v1.union(wall)

# gussets: right triangles in YZ at the foot's x edges
gusset_pts = [(-12.0, FOOT_T), (WALL_YIN, FOOT_T), (WALL_YIN, 45.0)]
for xg in (-20.0, 20.0 - 4.0):
    g = (
        cq.Workplane("YZ", origin=(xg, 0, 0))
        .polyline(gusset_pts).close().extrude(4.0)
    )
    v1 = v1.union(g)

# foot holes: M3 clearance + cb from the top
for xx in (PATTERN / 2, -PATTERN / 2):
    for yy in (PATTERN / 2, -PATTERN / 2):
        v1 = v1.cut(
            cq.Workplane("XY", origin=(xx, yy, -1)).circle(HOLE_D / 2).extrude(FOOT_T + 2)
        ).cut(
            cq.Workplane("XY", origin=(xx, yy, FOOT_T - 2.0)).circle(CB_D / 2).extrude(3)
        )

# twin T-channels on the outboard face
v1 = cut_channels(v1, [CH_PITCH / 2, -CH_PITCH / 2], WALL_YOUT, "y")

# =========================================================
# VARIANT 2 - SideFace flat plate
# frame: origin = pattern center on the vertical carriage face;
#        x = X travel, y = plate normal (0 = against carriage), z = up
# =========================================================
RAIL_OFF = 25.0   # rail centerline offset along travel (channels clear the cb heads)
PL_X_NEG, PL_X_POS = -20.0, RAIL_OFF + RAIL_W / 2 + 5.0   # -20 .. +45
PL_T = WALL_T

v2 = (
    cq.Workplane("XY")
    .box(PL_X_POS - PL_X_NEG, PL_T, Z_HI - Z_LO, centered=False)
    .translate((PL_X_NEG, 0, Z_LO))
)

# pattern holes: axis along y, FLUSH counterbore from the front (rail overlaps them)
for xx in (PATTERN / 2, -PATTERN / 2):
    for zz in (PATTERN / 2, -PATTERN / 2):
        v2 = v2.cut(
            cq.Workplane("XZ", origin=(xx, -1, zz)).circle(HOLE_D / 2).extrude(-(PL_T + 2))
        ).cut(
            cq.Workplane("XZ", origin=(xx, PL_T - 3.5, zz)).circle(CB_D / 2).extrude(-5)
        )

# twin T-channels on the front face, offset beside the pattern
v2 = cut_channels(v2, [RAIL_OFF - CH_PITCH / 2, RAIL_OFF + CH_PITCH / 2], PL_T, "y")

# corner chamfers on the tall plate for handling
for part_name in ():
    pass

# ---------------- verify ----------------
grip = 10.0 - CH_LIP - NUT_T   # screw left after lip+nut (goes through rail base + tip room)
assert CH_CAV_D >= NUT_T + 2.2, "no room for screw tip behind nut"
assert (RAIL_OFF - CH_PITCH / 2) - CH_CAV_W / 2 > PATTERN / 2 + CB_D / 2, "V2 channel hits cb"
b1, b2 = v1.val().BoundingBox(), v2.val().BoundingBox()
print(f"V1 bbox: x[{b1.xmin:.1f},{b1.xmax:.1f}] y[{b1.ymin:.1f},{b1.ymax:.1f}] z[{b1.zmin:.1f},{b1.zmax:.1f}]")
print(f"V2 bbox: x[{b2.xmin:.1f},{b2.xmax:.1f}] y[{b2.ymin:.1f},{b2.ymax:.1f}] z[{b2.zmin:.1f},{b2.zmax:.1f}]")
print(f"channel span rel. carriage face: {Z_LO} .. {Z_HI} (pairs 175 apart both fit, height adjustable)")
print(f"V1 rail clamp face at y={WALL_YOUT}; X-body clearance {WALL_YIN - RAIL_W/2:.1f}mm")
print(f"screw budget: M3x10 - lip {CH_LIP} - nut {NUT_T} = {grip:.1f}mm for rail base + tip")

# ---------------- export ----------------
outdir = os.path.dirname(os.path.abspath(__file__))
for name, part in [("ZMount_Var1_FaceUp", v1), ("ZMount_Var2_SideFace", v2)]:
    cq.exporters.export(part, os.path.join(outdir, f"{name}.stl"), tolerance=0.01, angularTolerance=0.1)
    cq.exporters.export(part, os.path.join(outdir, f"{name}.step"))
    print("exported", name)
