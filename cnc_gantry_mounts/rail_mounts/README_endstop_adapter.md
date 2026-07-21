# VariantB adapter + integrated KW12-3 endstop tower (Y axes)

The plain `VariantB_rail_to_1020_adapter` with the switch tower from
`endstop_clamp_kw12` grafted onto one side. Reprint this in place of the
plain adapter at each Y-rail end where you want a limit switch — the
adapter already sits at the rail's base-hole pair near the end, so the
switch lands right at the travel limit with **no extra attachment
hardware**. Rail mounting is unchanged (same M3 screws through to the
1020 T-nuts, same ridges, same wings, same toe clamps).

The switch interface is identical to the X/Z clamps, so every switch on
the machine mounts the same way: KW12-3 vertical in the 0.5 mm recess,
**lever hinge up / roller down, lever face toward the carriage, pins away
from travel**; 2× M2×10 through the vertical slots into M2 nuts dropped
into the channel on the tower's outer face from the top. Slots give
height adjustment and absorb the KW12 hole-position ambiguity. Roller
lands mid-face of the carriage; crash order roller → click → body →
tower, all collision-verified against rail/1020/carriage mocks.

## Which STL where

- `VariantB_adapter_endstop.stl` — at one end of the rail as printed;
  **rotate the whole adapter 180°** for the other end (the tower flips to
  the rail's other side, switch faces the other way — correct).
- `VariantB_adapter_endstop_mirrored.stl` — only if you want both Y
  switches on the SAME side of one rail (e.g., one cable run): use one
  standard + one mirrored per rail.
- Ends without a switch keep the plain adapter.

## Print

Exactly like the plain adapter: flat, wings down, no supports. The tower
(45 mm) rises clean; slots and nut channel print vertically. PETG/PLA,
4 perimeters.

## BOM per adapter

1× KW12-3 · 2× M2×10 + M2 nut · (rail/1020 hardware unchanged)

Source: `variant_b_endstop_adapter.py` (self-contained; adapter geometry
copied verbatim from `rail_mounts.py`, tower ported from
`endstop_mount_kw12.py`).
