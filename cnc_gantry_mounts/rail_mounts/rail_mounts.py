"""
Bessel-beam rig: 3D-printed mounts for a 300 mm linear rail stage
(MB1218-style, 30 mm wide body, base holes = 2x M3 counterbored,
17 mm apart across the rail, one pair near each end ~275 mm apart)
onto an imperial optical breadboard (1/4-20 on 1" grid).

Variant A: two identical feet, rail bolts directly to breadboard.
Variant B: rail -> adapter plates -> 1020 aluminum extrusion,
           extrusion held to breadboard by toe clamps.

All units mm.  Coordinate conventions per part:
  X = across the rail, Y = along the rail, Z = up.
Author: generated with CadQuery 2.8
"""

import cadquery as cq

# ----------------------------------------------------------------------------
# Shared parameters
# ----------------------------------------------------------------------------
M3_CLEAR = 3.4            # M3 clearance hole
HEX_AC = 6.70             # M3 nut pocket, across-corners (5.5 AF nut + 0.3 fit)
HEX_DEPTH = 2.7           # M3 nut pocket depth (nut is 2.4)
Q20_SLOT_W = 7.0          # 1/4-20 clearance slot width (bolt = 6.35)
GRID = 25.4               # breadboard hole pitch

RAIL_W = 30.0             # rail body width
RAIL_HOLE_DX = 17.0       # across-rail spacing of the M3 base holes
CHAN_CLR = 0.6            # locating-channel clearance on width

EXT_W = 20.0              # 1020 extrusion: width (lying flat)
EXT_H = 10.0              # 1020 extrusion: height
EXT_CLR = 0.3

def slot(wp, x1, x2, y, r, depth_or_through, z_top, solid):
    """Cut a slot (hull of two circles) along X from a top plane."""
    s = (cq.Workplane("XY", origin=(0, 0, z_top))
         .moveTo(x1, y).slot2D(abs(x2 - x1) + 2 * r, 2 * r, 0)
         )
    return s

# ----------------------------------------------------------------------------
# VARIANT A : direct-mount foot  (print 2)
# ----------------------------------------------------------------------------
A_W = 68.0          # across rail
A_L = 46.0          # along rail
A_T = 8.0           # plate thickness
RIDGE_H = 2.0
RIDGE_W = 2.5
A_SLOT_LEN = 33.0   # total slot length (along rail) incl. end radii
A_SLOT_X = GRID     # slot centers +/- 1 inch from rail centerline

def variant_a_foot():
    ch_half = (RAIL_W + CHAN_CLR) / 2          # 15.3
    body = (cq.Workplane("XY")
            .box(A_W, A_L, A_T, centered=(True, True, False)))
    # rail-locating ridges
    for sx in (+1, -1):
        ridge = (cq.Workplane("XY")
                 .center(sx * (ch_half + RIDGE_W / 2), 0)
                 .box(RIDGE_W, A_L, RIDGE_H, centered=(True, True, False))
                 .translate((0, 0, A_T)))
        body = body.union(ridge)
    # 1/4-20 slots, running along the rail (Y)
    for sx in (+1, -1):
        cut = (cq.Workplane("XY")
               .center(sx * A_SLOT_X, 0)
               .slot2D(A_SLOT_LEN, Q20_SLOT_W, 90)
               .extrude(A_T + RIDGE_H + 2))
        body = body.cut(cut)
    # M3 through-holes + hex nut pockets (open to top face; rail covers them)
    for sx in (+1, -1):
        x = sx * RAIL_HOLE_DX / 2
        body = body.cut(cq.Workplane("XY").center(x, 0)
                        .circle(M3_CLEAR / 2).extrude(A_T + 2))
        body = body.cut(cq.Workplane("XY", origin=(x, 0, A_T - HEX_DEPTH))
                        .polygon(6, HEX_AC).extrude(HEX_DEPTH + 1))
    # cosmetic: fillet outer vertical corners, chamfer top rim
    try:
        body = body.edges("|Z and (>X or <X)").fillet(4.0)
    except Exception:
        pass
    try:
        body = body.faces(">Z[1]").edges().chamfer(0.5)
    except Exception:
        pass
    return body

# ----------------------------------------------------------------------------
# VARIANT B part 1 : rail <-> 1020 squaring plate  (print 1 per crossing)
# ----------------------------------------------------------------------------
# The 1020 extrusions run PERPENDICULAR to the rails (gantry cross-members).
# The rail's M3 hole pair (17 mm apart, ACROSS the rail) therefore lies along
# the extrusion axis and lands directly over the extrusion's top T-slot: the
# rail's own M3 screws pass through this plate into T-nuts. The plate's job
# is squaring: top channel (along rail, Y) is keyed 90 deg to the bottom
# channel (along extrusion, X).
B_AD_W = 38.0        # X: along extrusion / across rail
B_AD_L = 38.0        # Y: along rail / across extrusion
B_CORE_T = 4.0       # core thickness (keep thin: screw must reach the T-nut)
SKIRT_H = 3.0        # bottom wing depth (keys onto the 20 mm extrusion)

def variant_b_adapter():
    ch_half = (RAIL_W + CHAN_CLR) / 2          # 15.3
    ex_half = (EXT_W + EXT_CLR) / 2            # 10.15
    body = (cq.Workplane("XY")
            .box(B_AD_W, B_AD_L, B_CORE_T, centered=(True, True, False)))
    # top ridges locating the rail (run along Y = rail axis)
    for sx in (+1, -1):
        ridge = (cq.Workplane("XY")
                 .center(sx * (ch_half + RIDGE_W / 2), 0)
                 .box(RIDGE_W, B_AD_L, RIDGE_H, centered=(True, True, False))
                 .translate((0, 0, B_CORE_T)))
        body = body.union(ridge)
    # bottom wings straddling the extrusion (run along X = extrusion axis)
    # prints flat; the 20.3 mm underside channel is an easy bridge
    for sy in (+1, -1):
        wing_w = B_AD_L / 2 - ex_half            # 8.85
        wing = (cq.Workplane("XY")
                .center(0, sy * (ex_half + wing_w / 2))
                .box(B_AD_W, wing_w, SKIRT_H, centered=(True, True, False))
                .translate((0, 0, -SKIRT_H)))
        body = body.union(wing)
    # M3 through-holes: rail screws pass straight through into slot T-nuts
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

# ----------------------------------------------------------------------------
# VARIANT B part 2 : extrusion toe clamp  (print 4)
# ----------------------------------------------------------------------------
# Cross-section in XZ, extruded 22 mm along Y.
# x=10 is the side face of the extrusion; nose reaches over its top face.
CL_Y = 22.0
CL_H = 16.0
CL_X0, CL_X1 = 10.0, 44.0     # body footprint
NOSE_X = 4.0                  # nose tip (stays clear of the 6 mm T-slot)
NOSE_Z = EXT_H - 0.2          # 9.8: bites 0.2 below extrusion top for preload
RELIEF = 0.5                  # bottom relief so heel + nose take the load
HEEL_X = 32.0
CL_SLOT_C1, CL_SLOT_C2 = 17.4, 33.6   # 1/4-20 slot center range
CL_CB_R = 5.6                 # counterbore slot radius (SHCS head 9.5)
CL_CB_FLOOR = 9.0             # counterbore floor height

def variant_b_clamp():
    pts = [(NOSE_X, CL_H), (CL_X1, CL_H), (CL_X1, 0), (HEEL_X, 0),
           (HEEL_X, RELIEF), (CL_X0, RELIEF), (CL_X0, NOSE_Z),
           (NOSE_X, NOSE_Z)]
    body = (cq.Workplane("XZ", origin=(0, CL_Y / 2, 0))
            .polyline(pts).close().extrude(CL_Y))
    mid = (CL_SLOT_C1 + CL_SLOT_C2) / 2
    ln = CL_SLOT_C2 - CL_SLOT_C1
    # through slot for 1/4-20
    body = body.cut(cq.Workplane("XY", origin=(mid, 0, -1))
                    .slot2D(ln + Q20_SLOT_W, Q20_SLOT_W, 0)
                    .extrude(CL_H + 2))
    # counterbore slot for the socket head
    body = body.cut(cq.Workplane("XY", origin=(mid, 0, CL_CB_FLOOR))
                    .slot2D(ln + 2 * CL_CB_R, 2 * CL_CB_R, 0)
                    .extrude(CL_H))
    # chamfer nose tip underside for printability when printed flat (optional)
    try:
        body = body.edges("|Y and <X and <Z").chamfer(1.0)
    except Exception:
        pass
    return body

# ----------------------------------------------------------------------------
# Build + export
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parts = {
        "VariantA_rail_foot": variant_a_foot(),
        "VariantB_rail_to_1020_adapter": variant_b_adapter(),
        "VariantB_1020_toe_clamp": variant_b_clamp(),
    }
    for name, p in parts.items():
        assert p.val().isValid(), name
        bb = p.val().BoundingBox()
        print(f"{name}: {bb.xlen:.1f} x {bb.ylen:.1f} x {bb.zlen:.1f} mm, "
              f"vol {p.val().Volume()/1000:.1f} cm^3")
        cq.exporters.export(p, f"{name}.stl", tolerance=0.01,
                            angularTolerance=0.1)
        cq.exporters.export(p, f"{name}.step")
    print("done")
