"""
X-rail -> Y-carriage bridge adapter plate (print TWO, identical)
================================================================
Bolts the X-axis rail (275mm, base-hole pairs 17mm across x 175mm apart,
counterbored M3) across the two Y-axis carriages (each: 4x M3 on a 20x20mm
square, facing up). One plate per Y carriage; the X rail lies across both.

Problem it solves: the Y1-Y2 separation is quantized by the breadboard's
1" grid (175mm is NOT a multiple of 25.4 -- you'll be at ~177.8mm), so the
rail's fixed 175mm pair spacing can never line up with two fixed hole sets.
Each plate therefore carries twin T-SLOT CHANNELS running along the bridge
direction: M3 hex nuts (5.5 AF) drop into the entry mouths and slide under
the lips, giving each rail end +-12mm of travel along the bridge. That
absorbs the grid mismatch (Y separations ~165..185mm all work) and lets
you square the gantry before tightening.

Layout (per plate, origin = Y-carriage hole-pattern center, z up):
  - x = bridge (X-rail) direction, y = Y-rail travel direction
  - 4x M3x10 SHCS down into the carriage at (+-10, +-10), counterbored
    4.5mm so the heads sit flush BELOW the channel plane (the rail slides
    over one screw column).
  - Rail band offset +25mm in y so the channels (at y = 16.5 and 33.5,
    17mm apart) clear the counterbores. The bridge therefore sits 25mm
    off the Y-carriage centerline -- harmless; flip the plate 180deg at
    install to choose which side (the 20x20 pattern is symmetric, both
    plates must be flipped the same way).
  - Rail screws: M3x10 through the rail's own counterbored base holes
    into the sliding nuts (lip 2.5 + nut 2.4 -> same budget as the
    Z-mount brackets; M3x12 if the rail base is >4mm under the head).

Assembly: bolt both plates to the Y carriages -> drop 2 nuts per plate
into the mouths, slide under the lips -> set the X rail on top, start its
4 base screws into the nuts (move the X carriage along its travel to
reach them) -> square the bridge against the Y rails -> tighten.

Print: flat, channels up, no supports. PETG/PLA, 4+ perimeters. 2 copies.
"""

import os
import cadquery as cq

# ---------------- parameters (mm) ----------------
PATTERN = 20.0
HOLE_D = 3.4
CB_D = 6.8
CB_DEPTH = 4.5            # flush heads (SHCS head 3.0) + 4.5mm thread into carriage
NUT_AF = 5.5
NUT_T = 2.4
CH_PITCH = 17.0           # rail base pair, across the rail
RAIL_OFF = 25.0           # rail band offset in y (channels clear the counterbores)
CH_CAV_W = NUT_AF + 0.3   # 5.8
CH_CAV_D = 5.2
CH_LIP = 2.5
CH_SLOT_W = 3.6
PLATE_T = CH_LIP + CH_CAV_D + 2.3   # 10.0
LIP_HALF = 14.0           # lip span x -14..+14 (screw travel ~ +-12.5)
MOUTH = 4.0               # nut entry mouth length past the lip
PL_X = 40.0               # +-20
PL_Y_NEG, PL_Y_POS = -20.0, RAIL_OFF + CH_PITCH / 2 + 11.5   # -20 .. +45

# ---------------- build ----------------
p = (
    cq.Workplane("XY")
    .box(PL_X, PL_Y_POS - PL_Y_NEG, PLATE_T, centered=False)
    .translate((-PL_X / 2, PL_Y_NEG, 0))
)

# carriage screw holes, counterbored flush from the top
for xx in (PATTERN / 2, -PATTERN / 2):
    for yy in (PATTERN / 2, -PATTERN / 2):
        p = p.cut(
            cq.Workplane("XY", origin=(xx, yy, -1)).circle(HOLE_D / 2).extrude(PLATE_T + 2)
        ).cut(
            cq.Workplane("XY", origin=(xx, yy, PLATE_T - CB_DEPTH)).circle(CB_D / 2).extrude(CB_DEPTH + 1)
        )

# twin T-slot channels along the bridge direction (x), on the top face
z_cav_lo = PLATE_T - CH_LIP - CH_CAV_D      # 2.3
for yc in (RAIL_OFF - CH_PITCH / 2, RAIL_OFF + CH_PITCH / 2):   # 16.5, 33.5
    # slot through the lip
    p = p.cut(
        cq.Workplane("XY")
        .box(2 * LIP_HALF, CH_SLOT_W, CH_LIP + 0.2, centered=False)
        .translate((-LIP_HALF, yc - CH_SLOT_W / 2, PLATE_T - CH_LIP - 0.1))
    )
    # nut cavity (extends into the mouth)
    p = p.cut(
        cq.Workplane("XY")
        .box(LIP_HALF + LIP_HALF + MOUTH, CH_CAV_W, CH_CAV_D, centered=False)
        .translate((-LIP_HALF, yc - CH_CAV_W / 2, z_cav_lo))
    )
    # entry mouth: open to the top (no lip) at +x end
    p = p.cut(
        cq.Workplane("XY")
        .box(MOUTH, CH_CAV_W, CH_LIP + CH_CAV_D + 0.2, centered=False)
        .translate((LIP_HALF, yc - CH_CAV_W / 2, z_cav_lo))
    )

# ---------------- verify ----------------
ch_edge = (RAIL_OFF - CH_PITCH / 2) - CH_CAV_W / 2
cb_edge = PATTERN / 2 + CB_D / 2
assert ch_edge > cb_edge, f"channel cavity ({ch_edge}) hits counterbore ({cb_edge})"
assert PLATE_T - CH_LIP - CH_CAV_D >= 2.0, "floor under cavity too thin"
bb = p.val().BoundingBox()
print(f"bbox: x[{bb.xmin:.1f},{bb.xmax:.1f}] y[{bb.ymin:.1f},{bb.ymax:.1f}] z[{bb.zmin:.1f},{bb.zmax:.1f}]")
print(f"channels at y = {RAIL_OFF - CH_PITCH/2}, {RAIL_OFF + CH_PITCH/2}; screw travel x ~ +-{LIP_HALF - 1.5}")
print(f"channel-to-counterbore clearance: {ch_edge - cb_edge:.1f}mm")
print(f"carriage screw engagement (M3x10, {PLATE_T} plate, cb {CB_DEPTH}): {10 - (PLATE_T - CB_DEPTH):.1f}mm")
print(f"rail screw budget: M3x10 - lip {CH_LIP} - nut {NUT_T} = {10 - CH_LIP - NUT_T:.1f}mm for rail base + tip")

# ---------------- export ----------------
outdir = os.path.dirname(os.path.abspath(__file__))
cq.exporters.export(p, os.path.join(outdir, "XtoY_bridge_adapter.stl"), tolerance=0.01, angularTolerance=0.1)
cq.exporters.export(p, os.path.join(outdir, "XtoY_bridge_adapter.step"))
print("exported XtoY_bridge_adapter (print 2 copies)")
