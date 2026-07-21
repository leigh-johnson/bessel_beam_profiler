# X-rail → Y-carriage bridge adapter (print 2 copies, identical)

Bolts the X-axis rail (275 mm, base pairs 17 mm across × 175 mm apart)
across the Y1/Y2 carriages (4× M3 on 20×20 mm, facing up). One plate per
carriage. Designed 2026-07-16. Source: `xy_bridge_adapter.py` (CadQuery).
All screws M3×10 SHCS + 4× M3 hex nuts total.

## Why T-slots instead of fixed holes

Your Y1–Y2 separation is set by the breadboard's 1″ grid — 175 mm is not
a multiple of 25.4, so you'll be at ~177.8 mm and fixed holes can never
line up with the rail's 175 mm pair spacing. Each plate has twin T-slot
channels (17 mm apart, running along the bridge): drop an M3 hex nut in
each entry mouth, slide it under the lip, and each rail end gets ±12 mm
of travel along the bridge. Any separation from ~165 to ~185 mm works,
and you can square the gantry before tightening.

## Details

- 40 × 65 × 10 mm. Carriage screws counterbored 4.5 mm → heads flush
  (the rail band passes over one screw column), 4.5 mm thread into the
  carriage.
- Rail band offset 25 mm from the carriage centerline (needed so the
  channels clear the counterbores). Harmless — the whole bridge just
  shifts 25 mm along Y. Both plates must point the same way; rotate both
  180° at install to choose which side the offset goes.
- Rail screw budget: M3×10 − 2.5 lip − 2.4 nut = 5.1 mm for the rail base
  thickness + tip room. M3×12 if the rail base is >4 mm under the head.

## Assembly

1. Bolt both plates to the Y carriages (4× M3×10 each, flush heads).
2. Two hex nuts per plate into the mouths, slide under the lips.
3. Set the X rail across, start its 4 base screws into the nuts (move the
   X carriage along its travel to reach each screw through the rail).
4. Square the bridge against the Y rails (both Y carriages at the same
   position), then tighten everything.

## Printing

Flat, channels up, no supports. PETG/PLA, 4+ perimeters. **Print 2.**

Files: `xy_bridge_adapter.py` · `XtoY_bridge_adapter.stl/.step` ·
`render_xy_bridge.py` + `xy_bridge_iso/top.png` (assembly, collision-
checked at 177.8 mm separation) · `preview_XtoY_adapter.png`
