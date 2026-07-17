# FLIR Blackfly S → linear-stage camera bracket

3D-printable bracket mounting the FLIR BFS-PGE-31S4M-C (29×29×30 mm) to the
vertical stage carriage's 4× M3 holes (20×20 mm square). Lens horizontal,
pointing away from the rail; the camera's whole back face (RJ45 + GPIO)
faces the rail across a **42 mm open gap** for plugging/unplugging Ethernet.

Designed 2026-07-16. Parametric source: `flir_bracket.py` (CadQuery 2.8).

## How it works

An L-profile, 32 mm tall (same as the carriage length, so nothing overhangs
along travel — zero travel loss):

- **Base plate** (6 mm) bolts to the carriage with 4× M3×10 SHCS,
  counterbored 2 mm → ~6 mm thread into the carriage. If the carriage taps
  are shallower, deepen `CB_DEPTH` and reprint (or add washers).
- **Wing** (7.6 mm thick) runs alongside the camera; the camera's bottom
  face bolts to its inner face with 3× M3×10 SHCS using FLIR's bottom-face
  M3 holes (pair 20 mm apart, 3 mm behind the front face + one on the
  centerline 26.7 mm behind the front face). **7.6 mm thickness is load-
  bearing**: M3×10 then engages 2.4 mm, and FLIR's threads are only
  2.5/2.8 mm deep — longer screws or a thinner wing will bottom out /
  punch through.
- Camera optical axis lands exactly at the carriage hole-pattern center
  (z) and ~3.5 mm off the rail centerline (y).
- The part is symmetric top/bottom: flip it 180° to put the wing on the
  other side of the camera (camera rides upside down then — harmless for
  a mono sensor).
- Inner-face rebate at the wing tip clears lenses up to ~⌀32.

## Printing

- Stand upright as modeled (L footprint on the bed, 32 mm tall). No supports.
- PETG or PLA, 4+ perimeters, ~40% infill. Counterbores bridge fine.

## Assembly order

1. Bolt bracket to carriage (4× M3×10, heads recessed on the gap side).
2. Hold camera against the wing's inner face, insert 3× M3×10 from the
   wing's outer face. Snug only — tiny thread engagement in the camera.
3. Plug in Ethernet/GPIO through the open gap between camera back and rail.

## Files

- `flir_bracket.py` — parametric CadQuery source (all dims at the top)
- `FLIR_BFS_rail_bracket.stl` / `.step` — print/CAD exports
- `render_preview.py`, `preview_bracket.png`, `assembly_iso.png`,
  `assembly_top.png` — renders + collision checks (camera, ⌀28 lens,
  33 mm RJ45 plug, rail body: all clear)

Camera hole data from FLIR's BFS-PGE dimensional drawing
(softwareservices.flir.com, BFS-PGE-31S4).
