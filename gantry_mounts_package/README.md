# Parametric Endstop + Camera Mounts — Bessel Beam Pi2 Gantry

Mounts for KW11-3Z-style lever microswitches (DAOKI 20-pack) on the XYZ CNC
gantry, driven by the Jackpot3 / FluidNC.

## Parts

| File | Use |
|---|---|
| `endstop_L_mount` | Switch stands vertical on a wall above a slotted base. Most versatile — bolts to T-slot extrusion (drop-in T-nuts), tapped holes, or through-bolts. Print 6 for min+max on all axes. |
| `endstop_flat_mount` | Switch lies flat on a low-profile plate. Use where the L-mount doesn't fit. |
| `striker_flag` | L-bracket that bolts to the moving carriage and trips the lever. Slots in both legs give you trigger-point adjustment in two directions. |

Both mount styles include ziptie slots as a fallback if the screw holes don't
line up with your actual switches, plus for wire strain relief.

## Before printing — verify with calipers

The models default to standard KW11-3Z dimensions. When your switches arrive,
check these against the `PARAMETERS` block at the top of `endstop_mounts.py`:

- `SW_HOLE_SPACE = 9.5` — mounting hole center-to-center spacing
- `SW_HOLE_UP = 2.9` — hole center height above the switch bottom edge
- `SW_LEN / SW_HT / SW_TH = 20.0 / 10.3 / 6.4` — body dimensions
- `SLOT_W = 4.4` — base slot width, sized for M4. Use 3.4 for M3, 5.4 for M5.

Then regenerate:

```bash
pip install build123d
python endstop_mounts.py     # writes STL + STEP to ./output
```

STEP files import cleanly into Fusion 360 / SolidWorks / FreeCAD if you'd
rather edit there.

## Print settings

- Material: PETG or ABS preferred (PLA creeps under constant screw preload);
  PLA is fine to prototype fitment.
- 4 perimeters, 40%+ infill, no supports needed — all parts print flat on
  their largest face (striker flag prints on its side).
- Layer height 0.2 mm.
- Fasteners: switch attaches with M2 self-tapping screws into the 1.8 mm
  pilot holes (M2×10 for the L-mount wall, M2×8 for the flat mount);
  base mounts with M4 + T-nuts or whatever your frame takes.

## Wiring reminder (FluidNC / Jackpot3)

Wire switches **NC** (COM + NC terminals) between the signal pin and GND, so a
broken/disconnected wire reads as "triggered" instead of silently disabling
the endstop. Use the 22 AWG wire from the BOM, twist the pair, and keep the
runs away from the stepper cables. In the FluidNC config, that pairs with the
default `:low` input logic; run `$Limits` / status report to confirm each
switch reads correctly before homing at full speed.

## Camera mount (`camera_mount.py`)

| File | Use |
|---|---|
| `camera_cradle` | U-channel cradle for the Blackfly S 29×29×30 mm case. Camera bolts down through a slot with 2–3× **M3×6 socket-head** screws into the bottom-face M3 holes. Flange has M4 slots (same fastening scheme as the endstop mounts). |
| `carriage_adapter` | 90° bracket for when the carriage face is vertical: bolts to the carriage's 18×18 M3 pattern, provides a horizontal shelf the cradle bolts onto. |
| `tilt_wedge` | Optional 3° shim under the cradle flange — use it if you see etalon fringes from the sensor cover glass under coherent light. |

### Gantry carriage interface (from the module drawing)

The linear module's carriage plate has **4× M3 tapped holes on an 18×18 mm
square**; the module body mounts via **4× M4 at 17×20 mm**.

- **Carriage face horizontal** (camera axis set by cradle): bolt the cradle
  straight down using the four counterbored holes in the camera floor —
  **M3×8 socket head**, installed *before* the camera drops in. The heads sit
  1 mm below the camera. The M4 flange slots then go unused (or add T-nut
  screws if you mount to something else).
- **Carriage face vertical** (camera axis horizontal): bolt the
  `carriage_adapter` to the carriage with **M3×10**, then bolt the cradle's
  flange slots to the adapter shelf with **M4×16 + nyloc nuts + washers**.
  The shelf extends away from the module so nothing fouls during travel.
- The **endstop mounts need no changes**: their base slots span 5.5–14.5 mm
  from center, which covers both the 17 mm and 20 mm spacings of the module
  body's M4 holes.

Design intent:

- **The bolt slot absorbs FLIR's unpublished hole spacing.** The camera's
  three bottom M3 holes lie on the case centerline; the slot accepts any
  colinear spacing, so no measurement needed. Screw length matters though:
  the floor web is 4 mm and FLIR's holes are only 2.5 mm deep — **M3×6, no
  longer**, or you'll bottom out.
- Camera front face mounts flush with the cradle front, so a C-mount→SM1
  adapter + ND filter stack overhangs in free air (no lip — the C-mount ring
  spans nearly the whole front face).
- Rear is open for the RJ45 and M8-GPIO connectors. Ziptie slots in the rear
  flange strip are for **ethernet strain relief** — anchor the cable there so
  gantry motion never torques the camera body.
- Side windows + open top: the camera dissipates up to 3 W. Don't enclose it.
- Ziptie slots at the wall tops let you strap the body down as a vibration
  backup.
- Print settings same as the endstop mounts; the cradle prints flat, the
  wedge prints thick-end down (or any way, it's a brick).

Extra hardware for this part: 2–3× M3×6 socket head cap screws.

## Trigger geometry tip

Aim the striker flag so it hits the lever mid-length and keep pressing past
the click point without bottoming out the lever — set your FluidNC pulloff
distance (`homing: mpos + pulloff`) to ~2–3 mm. Lever switches like these
repeat to roughly ±0.05–0.1 mm, which is well below the systematic error of
the gantry itself.
