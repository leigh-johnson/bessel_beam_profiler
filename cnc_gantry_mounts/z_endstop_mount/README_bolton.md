# Bolt-on KW12-3 end-stop mount — Z axis (motor end)

Replaces `endstop_clamp_kw12` **on Z only**. The clamp held by friction from
one M3 set screw and was getting knocked around by homing taps and handling
(2026-07-23 video). This mount bolts into the stage's **own tapped M3 pair**
on the back face nearest the stepper motor — 17 mm apart across the rail,
20 mm from the motor-end plate (the pair the vendor drawing labels 4-M4;
these take M3). Homing force goes into two steel screws: positive
registration, nothing to slip.

The switch interface is unchanged from the verified clamp design: KW12-3
vertical in a 0.5 mm recess, **hinge up (toward motor) / roller down**, lever
toward the carriage, 2× M2×10 through vertical stadium slots (±3.55 mm
height adjust, covers the KW12 hole-position ambiguity) into M2 nuts sliding
in an open-top channel. The M2 web is the same proven 4.0 mm (stack 9.75 mm);
the rest of the wall is 6 mm for stiffness.

## Trip point / machine-zero shift — read before first homing

Roller contact plane sits **24 mm below the end-plate hard stop** (click
~2 mm later, ~10.8 mm of lever throw before any hard contact). The stage has
200 mm stroke and the machine uses 90, so the lost headroom is free — but
machine zero (mpos 3 at trip) will land at a slightly different physical
height than with the old clamp. After installing: `$H`, check `$Limits`,
then **jog slowly to Z min once** before trusting soft limits again.

## BOM (per mount)

1× KW12-3 · 2× M3×10 SHCS into the rail (M3×12 also fine — 7.2 / 9.2 mm
engagement) · 2× M2×10 + M2 nut. No set screw, no M3 nut — the clamp
hardware is retired.

## Which STL

The part is handed. Hold it against the back of the rail: the tower must land
on the **same side as the old clamp's switch** (the other side is taken by
the stage's stock sensor bracket). Print whichever of
`endstop_bolton_kw12.stl` / `_mirrored.stl` puts it there.

## Setup

1. Drop two M2 nuts into the channel from the top; bolt the switch —
   hinge up, roller down, pins toward the motor.
2. Seat the plate on the back face over the tapped pair (relief pocket
   spans the leadscrew slot — it bears only on the side rails);
   2× M3×10 through the counterbores. Snug, no need to gorilla them.
3. Wire C + NC as before; route down the outside of the wall.
4. Re-home and do the Z-min check above.

## Print

Back face down, no supports (counterbores print as first-layer holes,
M2 slots bridge fine). PETG/PLA, 4 perimeters.

## Verified (by `endstop_bolton_kw12.py`)

Zero intersection: part vs rail + motor block, part vs carriage at trip AND
at the hard stop, switch vs everything. Roller lands at z = 17.6, dead
center of the 13–24 carriage band. Crash order: roller (x 24.0) → click →
switch body (x 13.2) → tower (x 12.7).

## Files

`endstop_bolton_kw12.py` (parametric CadQuery source; run to re-export) ·
`endstop_bolton_kw12.stl/.step`, `endstop_bolton_kw12_mirrored.stl` ·
`preview_bolton_assembly.png`, `preview_bolton_back.png`,
`preview_bolton_switchside.png`, `preview_bolton_part.png` ·
`render_preview_bolton.py`
