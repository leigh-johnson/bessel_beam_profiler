# Z-axis → X-carriage mounting brackets (two orientation variants)

Mounts the vertical Z stage (275 mm overall, base-hole pairs 17 mm across ×
175 mm apart, counterbored M3) onto the X-carriage's 4× M3 holes (20×20 mm
square). Designed 2026-07-16. Parametric source: `z_mount_brackets.py`
(CadQuery). All screws M3×10 SHCS.

## The T-slot trick (why no fixed holes for the rail)

Both variants have **twin vertical T-slot channels, 17 mm apart**, sized for
standard M3 hex nuts (5.5 AF) slid in from the top. The Z rail's own base
screws clamp onto the sliding nuts, so:

- **Rail height is adjustable** — set the rail bottom ~3 mm above the table
  whatever the X-axis stack turns out to be (1020 extrusion or not), and
  keep the camera's 40–80 mm imaging band covered.
- The rail's pair-to-end distance never needed measuring.
- Both pairs (bottom and top) ride the same channels; channels span
  carriage-face −55 mm to +160 mm, which fits the 175 mm pair spacing with
  the current ~60 mm face height. If the stack grows more than ~15 mm, bump
  `Z_LO` (and reprint) or shim the Y feet.

## Variant 1 — `ZMount_Var1_FaceUp`

For the X-carriage plate facing **UP** (plate normal parallel to Z).
L-bracket: foot bolts down onto the carriage (4× M3×10, counterbored 2 mm),
wall runs along the travel direction, offset so the hanging Z rail clears
the 30 mm X-stage body by 3 mm and drops to the table beside it. Camera
looks across the X axis. The hanging wall sweeps alongside the X-stage
body — check it clears whatever supports the X-stage ends before running
full travel.

## Assembly

1. Bolt bracket to the X carriage (4× M3×10).
2. Drop 4 M3 hex nuts into the channels from the top (flats ride the
   channel walls; they can't spin).
3. Hold the Z rail against the clamp face, run its base screws (through
   the rail's counterbored base holes) into the nuts — move the Z carriage
   along its travel to reach each screw. Set height: rail bottom ~3 mm
   above the breadboard. Tighten.
4. Rail bottom end stays completely clear for the future endstop mount.

Screw budget: M3×10 − 2.5 lip − 2.4 nut = 5.1 mm for the rail base
thickness + tip room (channel cavity has 2.8 mm of tip space behind the
nut). If your rail base is thicker than ~4 mm under the screw head, use
M3×12.

## Printing

- V1: print **on its side** (either 40 mm-wide face down); channels end up
  on a vertical face and print cleanly. No supports.
- V2: print **flat, channels up**. No supports.
- PETG or PLA, 4+ perimeters. If the channel lips lift under clamping,
  reprint with more perimeters or 0.2 mm smaller `CH_SLOT_W`.

## Files

`z_mount_brackets.py` (parametric source) · `ZMount_Var*.stl/.step` ·
`render_zmount.py` + `zmount_v*_*.png` (assembly renders w/ dummy rail,
collision-checked) · `preview_ZMount_*.png` (part close-ups)
