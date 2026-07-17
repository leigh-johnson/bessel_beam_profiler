"""
FLIR Blackfly S (BFS-PGE-31S4M-C) -> linear-stage carriage bracket
==================================================================
Mounts the 29x29x30 mm Blackfly S GigE camera to the vertical stage's
carriage (4x M3 tapped holes on a 20x20 mm square), lens horizontal,
pointing AWAY from the rail. The camera's bottom face bolts to a vertical
"wing" that runs alongside the camera, so the whole back face (RJ45 +
GPIO) stays open toward the rail with ETH_GAP mm of free space to
plug/unplug the Ethernet cable.

Coordinate system (as installed on the vertical rail):
  x : horizontal, away from the rail face (x=0 is the carriage face)
  y : horizontal, across the rail (y=0 is the carriage hole-pattern center)
  z : vertical, along rail travel (z=0 is the pattern center)

The part is symmetric in z, so the SAME print works with the wing on
either side of the camera: flip it 180 deg about the x axis to mirror.
(The camera ends up upside down in one of the two mountings; harmless.)

Screws (Leigh's M3x10 SHCS stock works everywhere):
  - 4x M3x10 into the carriage, counterbored CB_DEPTH so ~6 mm of thread
    engages the carriage.  Deepen CB_DEPTH if the carriage taps are shallow.
  - 3x M3x10 into the camera bottom face through the 7.6 mm wing:
    engagement 2.4 mm vs FLIR max depths 2.8/2.5 mm -- do NOT use longer
    screws or thinner wing, the camera threads bottom out.

Camera bottom-face hole map (FLIR BFS-PGE dimensional drawing):
  - 2x M3x0.5 depth 2.8, 20.0 mm apart, 3.0 mm behind the FRONT face
  - 1x M3x0.5 depth 2.5, on centerline, 26.7 mm behind the front face
  (M2 holes unused.)

Print: stand upright as modeled (L-footprint on bed, 32 mm tall).
No supports needed; counterbores bridge. PETG or PLA, >=4 perimeters.
"""

import cadquery as cq

# ---------------- parameters (mm) ----------------
# rail carriage
PATTERN = 20.0          # carriage hole square
RAIL_HOLE_D = 3.4       # M3 clearance
CB_D = 6.8              # SHCS head counterbore dia (head 5.5)
CB_DEPTH = 2.0          # counterbore depth (bump up if carriage taps are shallow)
PLATE_T = 6.0           # base plate thickness
PLATE_H = 32.0          # plate height along travel (carriage is 32 long)

# camera (FLIR BFS-PGE 29x29x30 case)
CAM_W = 29.0
CAM_LEN = 30.0
CAM_FRONT_PAIR_SETBACK = 3.0    # M3 pair behind front face
CAM_FRONT_PAIR_SPACING = 20.0
CAM_REAR_CENTER_SETBACK = 26.7  # single M3 behind front face, on centerline
CAM_HOLE_D = 3.2                # snug M3 clearance (locates the camera)

# layout
ETH_GAP = 42.0          # free space between carriage face and camera back
WING_T = 7.6            # wing thickness = M3x10 minus 2.4 mm engagement
WING_TIP_EXTRA = 0.6    # wing extends this far past camera front face
LENS_RELIEF = 1.5       # inner-face rebate near tip so fat lenses clear
LENS_RELIEF_LEN = 1.1   # rebate length back from the camera front face
CORNER_FILLET = 5.0     # inside corner plate<->wing
PLATE_Y_NEG = 18.0      # plate reach on the far side of the pattern

# ---------------- derived ----------------
cam_back_x = PLATE_T + ETH_GAP                 # 48.0
cam_front_x = cam_back_x + CAM_LEN             # 78.0
wing_tip_x = cam_front_x + WING_TIP_EXTRA      # 78.6
wing_yin = PATTERN / 2 + 8.0                   # wing inner face y=+18 (camera sits against it)
wing_yout = wing_yin + WING_T                  # +25.6
front_pair_x = cam_front_x - CAM_FRONT_PAIR_SETBACK    # 75.0
rear_center_x = cam_front_x - CAM_REAR_CENTER_SETBACK  # 51.3

# ---------------- build ----------------
# L-shaped extrusion, 32 tall, z-symmetric
profile = (
    cq.Sketch()
    .polygon([
        (0, -PLATE_Y_NEG),
        (PLATE_T, -PLATE_Y_NEG),
        (PLATE_T, wing_yin),
        (wing_tip_x, wing_yin),
        (wing_tip_x, wing_yout),
        (0, wing_yout),
        (0, -PLATE_Y_NEG),
    ])
)
body = (
    cq.Workplane("XY")
    .placeSketch(profile)
    .extrude(PLATE_H / 2, both=True)
)

# fillet the inside corner (plate front face meets wing inner face)
body = body.edges(
    cq.selectors.NearestToPointSelector((PLATE_T, wing_yin, 0))
).fillet(CORNER_FILLET)

# rail mounting holes: 4x M3 clearance + counterbore, axis along x
for yy in (PATTERN / 2, -PATTERN / 2):
    for zz in (PATTERN / 2, -PATTERN / 2):
        body = (
            body.cut(
                cq.Workplane("YZ", origin=(0, yy, zz))
                .circle(RAIL_HOLE_D / 2).extrude(PLATE_T)
            ).cut(
                cq.Workplane("YZ", origin=(PLATE_T - CB_DEPTH, yy, zz))
                .circle(CB_D / 2).extrude(CB_DEPTH + 1)
            )
        )

# camera holes through the wing, axis along y
cam_holes = [
    (front_pair_x, +CAM_FRONT_PAIR_SPACING / 2),
    (front_pair_x, -CAM_FRONT_PAIR_SPACING / 2),
    (rear_center_x, 0.0),
]
for xx, zz in cam_holes:
    body = body.cut(
        cq.Workplane("XZ", origin=(xx, wing_yin - 1, zz))
        .circle(CAM_HOLE_D / 2).extrude(-(WING_T + 2))
    )

# lens-clearance rebate on the wing's inner face at the tip
body = body.cut(
    cq.Workplane("XY")
    .box(
        wing_tip_x - (cam_front_x - LENS_RELIEF_LEN),
        LENS_RELIEF,
        PLATE_H + 2,
        centered=False,
    )
    .translate((cam_front_x - LENS_RELIEF_LEN, wing_yin, -(PLATE_H + 2) / 2))
)

# soften the wing tip outer corner
try:
    body = body.edges(
        cq.selectors.NearestToPointSelector((wing_tip_x, wing_yout, 0))
    ).chamfer(1.5)
except Exception as e:
    print(f"(cosmetic tip chamfer skipped: {e})")

# ---------------- verify ----------------
eng_cam = 10.0 - WING_T
assert eng_cam <= 2.5, "camera screws would bottom out"
bb = body.val().BoundingBox()
print(f"bracket bbox: x[{bb.xmin:.1f},{bb.xmax:.1f}] y[{bb.ymin:.1f},{bb.ymax:.1f}] z[{bb.zmin:.1f},{bb.zmax:.1f}]")
print(f"ethernet gap (carriage face -> camera back): {ETH_GAP:.1f} mm")
print(f"camera screw engagement (M3x10 through {WING_T} wing): {eng_cam:.1f} mm (FLIR max 2.8/2.5)")
print(f"rail screw engagement (M3x10, {PLATE_T} plate, {CB_DEPTH} cb): {10 - (PLATE_T - CB_DEPTH):.1f} mm into carriage")
print(f"camera axis: y={wing_yin - CAM_W/2 + 0:.1f}... center of cam body y={(wing_yin + (wing_yin - CAM_W))/2:.1f}, z=0 (pattern center)")
print(f"camera hole xs: front pair {front_pair_x}, rear center {rear_center_x}; cam back at {cam_back_x}, front at {cam_front_x}")

# ---------------- export ----------------
import os
outdir = os.path.dirname(os.path.abspath(__file__))
cq.exporters.export(body, os.path.join(outdir, "FLIR_BFS_rail_bracket.stl"), tolerance=0.01, angularTolerance=0.1)
cq.exporters.export(body, os.path.join(outdir, "FLIR_BFS_rail_bracket.step"))
print("exported STL + STEP")
