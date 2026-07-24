"""
Bolt-on end-stop mount for the KW12-3 roller-lever micro switch — Z axis
(MB1218-style linear stage, Pi2Su26 scan rig).

Replaces the clamp-on `endstop_clamp_kw12` on Z, which slid/rotated on the
rail under homing taps + handling (only friction from one M3 set screw).
This version bolts into the stage's OWN tapped mounting holes on the back
(base) face — the free pair nearest the stepper motor:

  vendor drawing (all strokes): hole pair 17 mm apart across the rail,
  20 mm from the motor-end plate's inner face; drawing labels them 4-M4
  but Leigh confirmed these take an M3 thread. Plenty of clearance behind
  the back face (>10 mm) — Z rail top end overhangs the Var2 bracket plate.

Positive registration: homing force (carriage moving UP toward the motor)
goes straight into two steel screws. Nothing to slip, nothing to knock.

Switch interface is IDENTICAL to the verified clamp design
(endstop_mount_kw12.py, 2026-07-18): KW12-3 vertical in a 0.5 recess,
hinge up / roller down, lever toward the carriage, 2x M2x10 through
vertical stadium slots (+-3.55) into M2 nuts sliding in an open-top
channel behind a 4.0 mm web. Same crash order, same +-height adjustment.

Geometry / frame (stage frame, orientation-independent):
  X along rail, x=0 at the motor-end plate's INNER face (the hard-stop
    face). Carriage approaches from +X moving toward 0 (i.e. homing up).
  Y across the rail (+-15 = rail edges). Tower on +Y; mirror for -Y.
  Z out of the back/base face: z=0 back face, rail body top 13,
    carriage band 13..24, end plate & motor block up to 28.

Trip plane at x = TRIP_X (roller contact): carriage leading face stops
~TRIP_X from the hard stop. Stage stroke is 200 mm and the machine only
uses 90, so sacrificing ~24 mm at the top costs nothing. NOTE: machine
zero (mpos 3 at trip) will shift a little vs the old clamp position —
re-home and jog carefully to Z min once after installing.
"""

import math
import cadquery as cq

# ------------------ stage facts (vendor drawing + measured) ------------------
RAIL_W       = 30.0
RAIL_BODY_H  = 13.0      # back face -> rail body top
SLOT_W       = 9.0       # central leadscrew slot width in the back face
CARRIAGE_BOT = 13.0      # carriage envelope 13..24, full rail width
CARRIAGE_TOP = 24.0
CARRIAGE_L   = 32.0
CARRIAGE_CLR = 0.6
EPLATE_T     = 5.0       # end plate x -5..0, full width, z 0..28
MOTOR_L      = 30.0      # motor block x -35..-5, full width, z 0..28
BLOCK_H      = 28.0
HOLE_X       = 20.0      # tapped M3 pair: 20 from plate inner face...
HOLE_DY      = 8.5       # ...17 mm apart across the rail

# ------------------ switch facts (KW12-3, unchanged) -------------------------
SW_L      = 20.0
SW_H      = 10.2
SW_W      = 6.4
SW_HOLE_SPACING   = 9.5
SW_HOLE_FROM_PINS = 2.9
HOLE_LO_FROM_BOT_MID = (3.2 + 7.3) / 2       # 5.25 (drawing-ambiguity midpoint)
LEVER_LEN = 18.8
LEVER_REST_DEG = 18.0
ROLLER_D, ROLLER_W = 4.8, 3.0

# ------------------ printed-part parameters ----------------------------------
TRIP_X     = 24.0        # roller contact plane (carriage face here at trip)
PLATE_T    = 6.0         # back plate thickness (z -6..0)
PLATE_X0, PLATE_X1 = 10.0, 30.0     # covers holes at x=20 with 10mm ring
PLATE_Y0   = -13.0       # far side just past the -y hole
SPINE_T    = 6.0         # side wall thickness (thicker than the old 4mm tower)
TOWER_T    = 4.0         # web thickness in the M2 screw zone (verified stack)
RELIEF_HW  = 6.0         # half-width of back-face relief over the slot
RELIEF_D   = 1.5

SW_BOT_NOM = 16.5        # nominal switch bottom -> roller lands mid-band
ADJ        = 3.55        # M2 slot half-travel (covers KW12 hole ambiguity)
RECESS_D   = 0.5
RECESS_CLR = 0.3

M2_SLOT_D  = 2.3
M2NUT_AF   = 4.0
M2NUT_T    = 1.6
M3_CLEAR_D = 3.4
M3_CB_D    = 6.2         # SHCS head 5.5 + clearance
M3_CB_DEPTH = 3.2        # head sits 0.2 sub-flush

# ------------------ derived --------------------------------------------------
tower_in  = RAIL_W / 2 + CARRIAGE_CLR        # 15.60
tower_out = tower_in + TOWER_T               # 19.60 (M2 web outer face)
spine_out = tower_in + SPINE_T               # 21.60

seat_y    = tower_in + RECESS_D              # 16.10 switch seating plane

# lever geometry -> roller contact face offset from the M2 hole line
_a        = math.radians(LEVER_REST_DEG)
_hinge_x  = 7.3 + 0.3                        # hinge just proud of body face
_tip_x    = _hinge_x + LEVER_LEN * math.sin(_a)
_roller_face_local = _tip_x + ROLLER_D / 2 * math.cos(_a) + ROLLER_D / 2  # 18.09

dx        = TRIP_X - _roller_face_local      # shift so roller face = TRIP_X
x_h       = dx                               # M2 hole line
sw_pins_x = x_h - SW_HOLE_FROM_PINS
sw_btn_x  = sw_pins_x + SW_H                 # lever-face plane
tower_x1  = sw_btn_x - 0.5
tower_x0  = x_h - 8.0

hole_z_lo = SW_BOT_NOM + HOLE_LO_FROM_BOT_MID            # 21.75
hole_z_hi = hole_z_lo + SW_HOLE_SPACING                  # 31.25
TOWER_TOP = SW_BOT_NOM + SW_L + ADJ + 1.0                # 41.9

# ------------------ build ----------------------------------------------------
def _box(x0, x1, y0, y1, z0, z1):
    return cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False) \
        .translate((x0, y0, z0))

# back plate
part = _box(PLATE_X0, PLATE_X1, PLATE_Y0, spine_out, -PLATE_T, 0)
# relief over the leadscrew slot -> plate bears only on the solid side rails
part = part.cut(_box(PLATE_X0 - 1, PLATE_X1 + 1, -RELIEF_HW, RELIEF_HW,
                     -RELIEF_D, 0.5))
# spine: full-height side wall from tower far end to plate end
part = part.union(_box(tower_x0, PLATE_X1, tower_in, spine_out,
                       -PLATE_T, TOWER_TOP))

# round the outer vertical corners now, while the solid is still simple —
# later cuts leave their own (sharp) edges, which is fine
for _r in (1.2, 0.8, 0.5):
    try:
        part = part.edges("|Z").fillet(_r)
        break
    except Exception as e:
        print(f"WARN: perimeter fillet r={_r} failed: {e}")

# nut window: locally thin the spine to the verified 4.0mm web in the
# M2 screw zone so M2x10 still reaches the sliding nuts
part = part.cut(_box(x_h - 4.6, x_h + 4.6, tower_out, spine_out + 1,
                     hole_z_lo - ADJ - 2.0, TOWER_TOP + 1))

# switch registration recess (vertical channel, open at top)
part = part.cut(_box(sw_pins_x - RECESS_CLR, sw_btn_x + RECESS_CLR,
                     tower_in, seat_y, 11.5, TOWER_TOP + 1))

# M2 vertical stadium slots through the web at x = x_h
def _vslot(zc, half_travel, d, y0, y1):
    s = _box(x_h - d / 2, x_h + d / 2, y0, y1, zc - half_travel, zc + half_travel)
    for z in (zc - half_travel, zc + half_travel):
        s = s.union(cq.Workplane("XZ").workplane(offset=-y1)
                    .center(x_h, z).circle(d / 2).extrude(y1 - y0))
    return s

for zc in (hole_z_lo, hole_z_hi):
    part = part.cut(_vslot(zc, ADJ, M2_SLOT_D, tower_in - 1, spine_out + 1))

# M2 nut channel behind the web (nuts drop in from the top)
part = part.cut(_box(x_h - (M2NUT_AF + 0.2) / 2, x_h + (M2NUT_AF + 0.2) / 2,
                     tower_out - (M2NUT_T + 0.15), spine_out + 1,
                     hole_z_lo - ADJ - 1.3, TOWER_TOP + 1))

try:
    part = part.faces(">Z").edges().chamfer(0.6)
    part = part.faces("<Z").edges().chamfer(0.6)
except Exception as e:
    print(f"WARN: chamfer skipped: {e}")

# M3 clearance holes + counterbores, cut LAST (round |Z holes break the
# perimeter fillet if present when it runs)
for sy in (HOLE_DY, -HOLE_DY):
    part = part.cut(cq.Workplane("XY").workplane(offset=-PLATE_T - 1)
                    .center(HOLE_X, sy).circle(M3_CLEAR_D / 2)
                    .extrude(PLATE_T + 2))
    part = part.cut(cq.Workplane("XY").workplane(offset=-PLATE_T)
                    .center(HOLE_X, sy).circle(M3_CB_D / 2)
                    .extrude(M3_CB_DEPTH))

result = part

# ------------------ mocks (renders + checks) ---------------------------------
def switch_mock(z_bot=SW_BOT_NOM, lever_angle_deg=LEVER_REST_DEG):
    body = _box(sw_pins_x, sw_btn_x, seat_y - SW_W, seat_y, z_bot, z_bot + SW_L)
    a = math.radians(lever_angle_deg)
    hx, hz = sw_btn_x + 0.3, z_bot + SW_L - 1.0
    tipx = hx + LEVER_LEN * math.sin(a)
    tipz = hz - LEVER_LEN * math.cos(a)
    lever = (cq.Workplane("XZ", origin=(0, seat_y - SW_W / 2 + 2.0, 0))
             .polyline([(hx, hz), (tipx, tipz), (tipx + 0.5, tipz + 0.5),
                        (hx + 0.5, hz + 0.5)]).close().extrude(4.0))
    roller = (cq.Workplane("XZ", origin=(0, seat_y - SW_W / 2 + ROLLER_W / 2 + 0.5, 0))
              .center(tipx + ROLLER_D / 2 * math.cos(a),
                      tipz + ROLLER_D / 2 * math.sin(a))
              .circle(ROLLER_D / 2).extrude(ROLLER_W))
    roller_face_x = tipx + ROLLER_D / 2 * math.cos(a) + ROLLER_D / 2
    return body.union(lever).union(roller), (roller_face_x, tipz)

def rail_mock(length=120.0):
    rail = _box(0, length, -RAIL_W / 2, RAIL_W / 2, 0, RAIL_BODY_H)
    rail = rail.cut(_box(-1, length + 1, -SLOT_W / 2, SLOT_W / 2,
                         -1, RAIL_BODY_H - 4))          # leadscrew slot
    block = _box(-EPLATE_T - MOTOR_L, 0, -RAIL_W / 2, RAIL_W / 2, 0, BLOCK_H)
    return rail.union(block)

def carriage_mock(face_x=TRIP_X):
    return _box(face_x, face_x + CARRIAGE_L, -RAIL_W / 2, RAIL_W / 2,
                CARRIAGE_BOT, CARRIAGE_TOP)

def _vol(shape):
    try:
        return shape.val().Volume()
    except Exception:
        return sum(s.Volume() for s in shape.vals())

if __name__ == "__main__":
    cq.exporters.export(result, "endstop_bolton_kw12.step")
    cq.exporters.export(result, "endstop_bolton_kw12.stl",
                        tolerance=0.01, angularTolerance=0.1)
    cq.exporters.export(result.mirror("XZ"), "endstop_bolton_kw12_mirrored.stl",
                        tolerance=0.01, angularTolerance=0.1)
    sw, (rx, rz) = switch_mock()
    cq.exporters.export(sw, "switch_mock.stl", tolerance=0.02)
    cq.exporters.export(rail_mock(), "rail_mock.stl", tolerance=0.05)
    cq.exporters.export(carriage_mock(), "carriage_mock.stl", tolerance=0.05)

    # ---------------- verification ----------------
    ok = True
    checks = [
        ("part vs rail+block", result.intersect(rail_mock()), 0.001),
        ("part vs carriage@trip", result.intersect(carriage_mock(TRIP_X)), 0.001),
        ("part vs carriage@hardstop", result.intersect(carriage_mock(0.0)), 0.001),
        ("switch vs rail+block", sw.intersect(rail_mock()), 0.001),
        ("switch vs carriage@trip", sw.intersect(carriage_mock(TRIP_X)), 0.001),
    ]
    for name, inter, tol in checks:
        v = _vol(inter)
        flag = "OK " if v < tol else "FAIL"
        if v >= tol:
            ok = False
        print(f"[{flag}] {name}: intersection {v:.4f} mm^3")

    band_ok = CARRIAGE_BOT + 1 < rz < CARRIAGE_TOP - 1
    print(f"[{'OK ' if band_ok else 'FAIL'}] roller z={rz:.1f} in carriage band "
          f"{CARRIAGE_BOT}..{CARRIAGE_TOP} (with 1mm margin)")
    ok = ok and band_ok

    print(f"trip: carriage face at x={rx:.1f}; hard stop (end plate) at x=0 "
          f"-> {rx:.1f} mm stop margin; click ~2 mm after contact")
    print(f"crash order: roller x={rx:.1f} > switch body x={sw_btn_x:.1f} > "
          f"tower x={tower_x1:.1f} (>= {rx - sw_btn_x:.1f} mm lever throw before "
          f"body contact)")
    print(f"M2x10 stack: {SW_W - RECESS_D:.1f} switch + "
          f"{TOWER_T - (M2NUT_T + 0.15):.2f} web + {M2NUT_T} nut = "
          f"{SW_W - RECESS_D + TOWER_T - 0.15:.2f} mm")
    eng10 = 10 - (PLATE_T - M3_CB_DEPTH)
    eng12 = 12 - (PLATE_T - M3_CB_DEPTH)
    print(f"thread engagement into rail: M3x10 -> {eng10:.1f} mm, "
          f"M3x12 -> {eng12:.1f} mm (side rail is ~13 solid)")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
