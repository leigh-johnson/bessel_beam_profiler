# Linear rail → 1/4-20 optical breadboard mounts (PLA prototype)

3D-printed mounts for the 300 mm MB1218-style linear rail stage (30 mm body,
base holes = M3 counterbored pairs, **17 mm apart across the rail**, one pair
near each end ~275 mm apart) on an imperial breadboard (1/4-20 on 1" grid).

All parts are parametric — dimensions live at the top of `rail_mounts.py`
(CadQuery). Re-run it to regenerate STL/STEP after edits.

---

## Variant A — direct mount

**Print 2× `VariantA_rail_foot`** (one per hole pair), flat side down, no supports.

Per rail:

| Hardware | Qty |
|---|---|
| M3×10 socket head (through rail base into nut) | 4 |
| M3 hex nut (standard, 5.5 AF × 2.4) | 4 |
| 1/4-20 socket head, 5/8"–3/4" + into breadboard | 4 |

Assembly:
1. Drop an M3 nut into each hex pocket on the foot's top face (the rail traps them).
2. Set the rail into the channel between the ridges, start the M3×10 screws through the rail's counterbored holes — snug, don't tighten.
3. The 1/4-20 slots run **along** the rail axis (33 mm long) so the 275 mm hole-pair spacing never needs to match the 1" grid. Place the rail centerline directly over a row of breadboard holes (the wing slots sit exactly ±1" from centerline), bolt down with 1/4-20 screws, then torque the M3s.

Notes:
- M3×10 assumes the rail base is ~2–4 mm thick under its counterbore seat. Screw tips can protrude up to ~1 mm shy of the breadboard through the foot's through-holes; if a screw bottoms out, add a washer inside the rail counterbore.
- Socket-head 1/4-20 works directly; large fender washers will hit the locating ridge.

## Variant B — rails on perpendicular 1020 cross-members, clamped to breadboard

The 1020 extrusions run **perpendicular** to the rails (gantry cross-members:
each extrusion passes under both Y rails near their ends and squares them).
Because the rail's M3 pair is 17 mm apart *across* the rail, both holes land
directly over the extrusion's top T-slot — the rail's own screws go straight
into T-nuts. No hex nuts needed in this variant.

**Print 1× `VariantB_rail_to_1020_adapter` per rail/extrusion crossing**
(4 for a two-rail gantry; flat, no supports — the underside channel bridges)
and **2+× `VariantB_1020_toe_clamp` per extrusion** (print **on its side**, no
supports).

The plate is a squaring key: its top channel (fits the rail, ±0.3 mm) is
molded at exactly 90° to its bottom channel (fits the 20 mm extrusion). With
a plate at every crossing, the two rails come out parallel and square to the
cross-members.

Per two-rail gantry (2 extrusions):

| Hardware | Qty |
|---|---|
| M3×12 socket head (through rail base + plate, into T-nut; see note) | 8 |
| 10-series M3 T-nut (6 mm slot) | 8 |
| 1/4-20 socket head 5/8" or 3/4" (clamps; head recesses into clamp) | 4–6 |

Assembly:
1. Slide two T-nuts per crossing into each extrusion's top slot.
2. Set a squaring plate on the extrusion at each crossing (wings straddle the extrusion), seat the rail in the plate's top channel, and run the M3 screws through rail + plate into the T-nuts — snug, not tight.
3. Slide the rails along the slots to set the gantry width / rail spacing, check Y1‖Y2, then torque the M3s.
4. Clamp each extrusion to the breadboard with toe clamps: nose hooks over the extrusion's top edge (0.2 mm preload bite, clears the T-slot), 1/4-20 through the slotted counterbore into any hole 8–33 mm from the extrusion's side face. Place clamps away from the crossings (the clamp body is taller than the rail's underside). If a breadboard hole lands closer than ~8 mm to the extrusion face, clamp from the other side or nudge the extrusion.

Screw-length note: the stack is rail base seat (~2–4 mm) + 4 mm plate core,
and the T-nut threads start ~1–1.5 mm below the extrusion face. M3×12 gives
solid engagement; your existing M3×10 works if the rail seat is ≤3 mm. If a
screw bottoms out against the slot floor (~4.5 mm deep), add a washer under
its head inside the rail counterbore.

## Print settings (both variants)

- PLA or PETG, 0.2 mm layers, 4 perimeters, ≥40 % infill (clamps: 6 perimeters / 60 %+ — they see real bending load).
- Clearances are printed-in: M3 holes 3.4 mm, nut pockets 5.8 AF, 1/4-20 slots 7.0 mm, rail channel +0.6 mm, extrusion channel +0.3 mm. If your printer runs tight, scale X/Y by ~100.5 % or open `rail_mounts.py` and bump the `*_CLR` values.

## Files

- `VariantA_rail_foot.stl / .step`
- `VariantB_rail_to_1020_adapter.stl / .step`
- `VariantB_1020_toe_clamp.stl / .step`
- `rail_mounts.py` — parametric source (CadQuery)
- `assembly_VariantA.png`, `assembly_VariantB.png` — cross-section diagrams
- `preview_*.png` — 3-view renders of each part
