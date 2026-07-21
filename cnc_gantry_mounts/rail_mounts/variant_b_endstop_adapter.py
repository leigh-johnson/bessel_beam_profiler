"""
VariantB rail->1020 adapter WITH integrated KW12-3 endstop tower (Y axes).

The plain VariantB_rail_to_1020_adapter (rail_mounts.py, 2026-07-16) plus the
switch tower from endstop_clamp_kw12 grafted onto one side, so the Y-axis
limit switches mount identically to the X/Z clamps -- same KW12-3 pocket,
same M2x10 screws in vertical slots, same sliding-nut channel, same
hinge-up / roller-down orientation. The adapter already lives at the rail's
base-hole pair near each end, so the switch lands exactly at the travel end
with no extra attachment hardware.

Frame (same as rail_mounts.py): X across the rail / along the 1020,
Y along the rail, Z up; rail bottom sits at z = core top = 4.0.
Carriage approaches from +Y; its face band is z 19..30 (rail bottom +15..26).
Roller tip lands at z ~21.6, mid-band. Tower inner face at x = 15.6
(0.6 mm clear of the 30 mm carriage envelope), verified crash order:
roller -> click -> switch body -> tower face.

At the rail's OTHER end, rotate the whole adapter 180 deg (tower flips to
the far side of the rail, switch faces the other way -- correct). Use the
_mirrored STL only if both Y switches must sit on the SAME side of a rail.

Print exactly like the plain adapter: flat, wings down, no supports; the
tower rises clean. Reprint 2 (or 4) of these to replace the plain adapters
at the ends where you want switches.

BOM per adapter: KW12-3 + 2x M2x10 + 2x M2 nut (rail hardware unchanged).
"""

import os
import cadquery as cq

# ---------------- plain adapter (verbatim from rail_mounts.py) ----------------
M3_CLEAR = 3.4
Q20_SLOT_W = 7.0
RAIL_W = 30.0
RAIL_HOLE_DX = 17.0
CHAN_CLR = 0.6
EXT_W, EXT_H, EXT_CLR = 20.0, 10.0, 0.3
RIDGE_H, RIDGE_W = 2.0, 2.5
B_AD_W = B_AD_L = 38.0
B_CORE_T = 4.0
SKIRT_H = 3.0

def variant_b_adapter():
    ch_half = (RAIL_W + CHAN_CLR) / 2
    ex_half = (EXT_W + EXT_CLR) / 2
    body = (cq.Workplane("XY")
            .box(B_AD_W, B_AD_L, B_CORE_T, centered=(True, True, False)))
    for sx in (+1, -1):
        ridge = (cq.Workplane("XY")
                 .center(sx * (ch_half + RIDGE_W / 2), 0)
                 .box(RIDGE_W, B_AD_L, RIDGE_H, centered=(True, True, False))
                 .translate((0, 0, B_CORE_T)))
        body = body.union(ridge)
    for sy in (+1, -1):
        wing_w = B_AD_L / 2 - ex_half
        wing = (cq.Workplane("XY")
                .center(0, sy * (ex_half + wing_w / 2))
                .box(B_AD_W, wing_w, SKIRT_H, centered=(True, True, False))
                .translate((0, 0, -SKIRT_H)))
        body = body.union(wing)
    for sx in (+1, -1):
        x = sx * RAIL_HOLE_DX / 2
        body = body.cut(cq.Workplane("XY", origin=(x, 0, -SKIRT_H - 1))
                        .circle(M3_CLEAR / 2)
                        .extrude(B_CORE_T + RIDGE_H + SKIRT_H + 2))
    try:
        body = body.edges("|Z and (>X or <X)").fillet(3.0)
    except Exception:
        pass
    return body

# ---------------- endstop tower (ported from endstop_mount_kw12.py) ----------
# clamp frame -> adapter frame: x_c -> y (carriage from +Y), y_c -> x
# (tower at +X), z_c -> z - RAIL_Z0.
RAIL_Z0   = B_CORE_T          # 4.0  rail bottom height in this frame
CARR_CLR  = 0.6
TOWER_T   = 4.0
TW_XIN    = RAIL_W / 2 + CARR_CLR          # 15.6
TW_XOUT   = TW_XIN + TOWER_T               # 19.6
TW_Y0, TW_Y1 = -8.0, 6.8                   # along rail (lever face side +Y)
SW_BOT    = 16.5 + RAIL_Z0                 # 20.5 switch body bottom
TW_Z1     = SW_BOT + 20.0 + 3.55 + 1.0     # 45.05 tower top
RECESS_D  = 0.5
SW_PINS_Y, SW_BTN_Y = -2.9, 7.3
ADJ       = 3.55
M2_SLOT_D = 2.3
M2NUT_AF, M2NUT_T = 4.0, 1.6
HOLE_Z    = (SW_BOT + 5.25, SW_BOT + 5.25 + 9.5)   # 25.75, 35.25

def _box(x0, x1, y0, y1, z0, z1):
    return cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False) \
        .translate((x0, y0, z0))

def add_tower(body):
    body = body.union(_box(TW_XIN, TW_XOUT, TW_Y0, TW_Y1, 0, TW_Z1))
    # switch registration recess on the inner (rail-side) face, open at top
    body = body.cut(_box(TW_XIN, TW_XIN + RECESS_D,
                         SW_PINS_Y - 0.3, SW_BTN_Y + 0.3, RAIL_Z0 + 11.5, TW_Z1 + 1))
    # M2 vertical stadium slots through the tower at y = 0
    for zc in HOLE_Z:
        s = _box(TW_XIN - 1, TW_XOUT + 1, -M2_SLOT_D / 2, M2_SLOT_D / 2,
                 zc - ADJ, zc + ADJ)
        body = body.cut(s)
        for z in (zc - ADJ, zc + ADJ):
            body = body.cut(cq.Workplane("YZ", origin=(TW_XIN - 1, 0, z))
                            .circle(M2_SLOT_D / 2).extrude(TOWER_T + 2))
    # M2 nut channel on the outer face (nuts drop in from the top)
    nutch_x0 = TW_XOUT - (M2NUT_T + 0.15)
    body = body.cut(_box(nutch_x0, TW_XOUT + 1,
                         -(M2NUT_AF + 0.2) / 2, (M2NUT_AF + 0.2) / 2,
                         HOLE_Z[0] - ADJ - 1.3, TW_Z1 + 1))
    return body

part = add_tower(variant_b_adapter())

# ---------------- verify ----------------
bb = part.val().BoundingBox()
print(f"bbox: x[{bb.xmin:.1f},{bb.xmax:.1f}] y[{bb.ymin:.1f},{bb.ymax:.1f}] z[{bb.zmin:.1f},{bb.zmax:.1f}]")
print(f"rail bottom z={RAIL_Z0}, carriage band z {RAIL_Z0+15}..{RAIL_Z0+26}")
print(f"switch bottom z={SW_BOT}, roller tip ~z={SW_BOT+1.1:.1f} (mid-band), slots z {HOLE_Z[0]-ADJ:.1f}..{HOLE_Z[0]+ADJ:.1f} / {HOLE_Z[1]-ADJ:.1f}..{HOLE_Z[1]+ADJ:.1f}")
print(f"tower inner face x={TW_XIN} (carriage envelope +-15); rail M3 holes at x=+-8.5 clear")

# ---------------- export ----------------
outdir = os.path.dirname(os.path.abspath(__file__))
cq.exporters.export(part, os.path.join(outdir, "VariantB_adapter_endstop.stl"),
                    tolerance=0.01, angularTolerance=0.1)
cq.exporters.export(part, os.path.join(outdir, "VariantB_adapter_endstop.step"))
cq.exporters.export(part.mirror("YZ"), os.path.join(outdir, "VariantB_adapter_endstop_mirrored.stl"),
                    tolerance=0.01, angularTolerance=0.1)
print("exported VariantB_adapter_endstop (+mirrored, +step)")
