# Automated gantry scans (FluidNC + `dataset auto`)

How the profiler talks to the Jackpot3/FluidNC controller and runs the
automated beam-stack dataset collection.

## Coordinate convention (one frame, used everywhere)

* **X** = horizontal transverse (perpendicular to the beam)
* **Y** = beam propagation direction down the table
* **Z** = vertical (perpendicular to the optics table)

The gantry's machine axes coincide with this frame after homing, so
`GantryPosition_mm` IS the machine coordinate. `TablePosition_mm` differs
only in Y, where the per-placement measurement anchors machine Y to the
beamline: `TableY = measured + BeamDirectionSign * (machineY − y-start)`.
A beam cross-section is an **X-Z raster** at fixed machine Y; the scan
**steps along Y**.

`--beam-direction` sets the propagation sign. **Verified on this rig 2026-07-22**
(preflight beam-direction check): machine +Y moves the camera TOWARD the
optic, so the default is `-y` — stepping +Y along the stack means beam
distance *decreases* from the measured start value. Re-verify (preflight
`--motion`) if the gantry is ever re-oriented on the table (for example, if the beam path is reflected at the END of the optics table).

## How G-code is sent to FluidNC

FluidNC (web ui: http://fluidnc-sr2.local) sends GRBL protocol over a raw TCP socket on port
23 (the Telnet server on `fluidnc-sr2.local` / 10.43.19.86 same host as the web ui). `profiler/fluidnc_stage.py` wraps this:

* Send a newline-terminated line (`$H`, `G53 G1 X60 Y80 Z-50 F400`), read
  `ok` / `error:N`.
* `ok` only means *queued*. Motion completion = poll the realtime `?` query
  and parse `<Idle|MPos:60.000,80.000,-50.000|...>` until state is `Idle`
  and MPos matches the target.
* All moves are absolute **machine** coordinates (`G53 G1`), same as
  `rangetest.gcode`.
* Targets are validated in Python against the safe envelope
  (X 5–115, Y 5–155, Z −120…−2) *before* sending, because tripping the
  firmware soft limit raises `ALARM:2` and forces a reset + `$H`.
* `FluidNCStageController` implements the `StageController` interface that
  `FLIRDatasetWriter.acquire_scan` already calls, so the writer's
  move → settle → trigger → save loop is unchanged.

## Sanity-check the connection first

```bash
python cli.py gantry status                  # <Idle|MPos:...> or Alarm
python cli.py gantry home                    # $H (confirms first; machine moves!)
python cli.py gantry move --x 60 --y 80 --z -60
python cli.py gantry send '$SS'              # any raw command
```

## Run a dataset collection

```bash
python cli.py dataset auto --dataset-root data \
    --y-start 10 --y-stop 150 --y-step 10 \
    --x-min 45 --x-max 75 --x-step 5 \
    --z-min -75 --z-max -45 --z-step 4
```
The FLIR 31S4M camera has a resolution 2048 x 1536 (4:3 aspect ratio), so it's recommended to use an `--x-step` `--z-step` with a similar ratio. All stop/start/step values are in mm, with the smallest accepted step size ~0.06mm.

Workflow per placement (repeats for as many placements as needed):

1. Prompts for optic configuration and placement ID, then homes (`$H`).
2. You measure the optic→sensor distance along the beam with the camera at
   machine Y = `--y-start` and enter it (cm). The along-beam position of
   every slice derives from it.
3. For each machine Y: the camera moves to the calibration point (center
   of the X-Z caps), exposure auto-calibrates headlessly (seeded from the
   previous slice; halve when saturated, proportional step up when dim,
   accept when max is in [60%, 100%) of the saturation threshold with zero
   saturated pixels), then **backgrounds** are captured (see below), then
   the X-Z raster runs.
4. At the end you can move the gantry to a new placement; it re-homes and
   asks for a fresh distance measurement, and a new run directory begins.

`Ctrl-C` sends a feed hold (`!`) to the gantry before exiting. To release the feed hold, send `~` command to FluidNC via web ui, UART, or telnet. 

### Raster modes (`--raster`)

* `adaptive` (default): each slice starts with ONE seed frame at the
  calibration point, then the rectangle grows one grid step at a time in
  any direction whose edge frames still show signal in their border
  strips, referenced to this slice's own off-axis background
  (p99 + `--signal-margin` counts, at least `--min-signal-pixels` pixels).

The raster stops when all four edges are dark, or at the specified `--x/--z` ranges (by default the entire range is explored). 

  **A beam that fits inside one camera frame takes exactly one frame** (all four border strips of the seed frame are dark). 
  
The dark frames are labeled with `-dark` in the filename and excluded from composites by default.

* `fixed`: always raster the specified XZ grid.

* **Find-beam sweep** (`--find-beam/--no-find-beam`): when no
  `--calibration-x/-z` is given (or a slice reports `BeamFound: false`),
  the camera sweeps a column of frames along Z at the calibration X —
  starting from the FAR extrema (away from the Z home switch) looking for CONTRAST (frame max − median >= 30
  counts).
  
* **Follow-beam** (`--follow-beam/--no-follow-beam`): after each slice,
  the calibration point and raster seed move to that slice's brightest
  measured cell.

Every slice folder gets a `raster_metadata.json` containing the grid,
final rectangle, cells captured vs. the full-grid count, per-cell border
signal flags and file paths, growth history per pass, why each edge
stopped (`dark`/`cap`), the signal threshold and where it came from, and
`BeamFitsInSingleFrame`.

### Background modes (`--background-mode`)

* `offaxis` (default): at each Y, right after calibration, the camera
  drives to an X/Z position outside the beam (default: the machine-limit
  X/Z corner farthest from the raster; override with
  `--background-x/--background-z`) and captures `--background-shots`
  frames at that slice's calibrated exposure.
  
  New background shots are taken only when exposure has changed by at least `--background-exposure-change`
  (default 10%) since the background was last captured. 
  set `--background-exposure-change=0` to capture at every slice.
  Slices that reuse an earlier background indicate this in
  `background_reference.json`
  
* `none` (or `--skip-background`): quick runs without background images

## Output layout

```
data/auto_scan-2026-07-22_14-01-05/        # one run dir per placement
    run_metadata.json
    auto_scan_setup.json                   # placement, ranges, convention
    camera_settings.json
    frames.jsonl                           # global manifest (all frames)
    y0100.00cm/                            # distance along the beam
        frames.jsonl                       # per-slice manifest (compositable)
        calibration_result.json            # exposure, max, converged
        calibrated_camera_settings.json
        raster_metadata.json               # grid, growth history, edge stops,
                                           # threshold, BeamFitsInSingleFrame
        background_reference.json          # which background frames apply to
                                           # this slice (captured or reused)
        placement-01-background-...-shot0000.npy      # offaxis mode: the
                                           # slice's exposure-matched background
        placement-01-...-tabley...-gantryx...y...z...-shot0000.npy  (+ .jpg)
    y0101.00cm/
        ...
    scan.log                               # timestamped log of the placement
                                           # (also shown on the console)
```

Every frame's manifest record carries `Exposure_us`, `MachineY_mm`,
`BeamY_mm`, `MeasuredFrom`, and `ScanKind`
(`AutoBeamStack` / `Background`; off-axis backgrounds also record
`BackgroundMode: OffAxisAmbient` and their gantry position).

## Logging

The CLI configures Python logging at INFO by default (`--log-level` on the
top-level `cli` group changes it). All scan progress is logged (slice headers,
calibration results, background capture/reuse decisions, adaptive-raster
growth and truncation warnings). The dataset
subcommands (`auto`, `static`, `manual`) write to a timestamped
`scan.log` in the run directory. For `dataset auto`, each placement's
run directory gets its own `scan.log`.

## WiFi drop-offs and auto-reconnect

The ESP32 rides the moving gantry on WiFi, so link drop-offs are
expected (FluidNC only accepts network requests while all motors are idle). The client automatically reconnects on a socket error or status
timeout (5 attempts, 2 s apart, configurable via
`ReconnectAttempts`/`ReconnectDelay_s`).

## Live preview (`--preview` / `dataset watch`)

`dataset auto --preview` opens a live viewer showing each frame as it is
saved (filename, peak counts, timestamp). It runs as a SEPARATE process
that tails the run directory.

```
python cli.py dataset watch                    # newest run under data/
python cli.py dataset watch <run_dir>          # a specific run
```

## Defaults worth knowing

* Feed 400 mm/min (machine max_rate is 500); 0.2 s settle after Idle.
* X steps 5 mm and Z steps 4 mm ≈ 25–30 % overlap for the BFS-PGE-31S4M's
  7.1 × 5.3 mm sensor, for stitching.
* `--skip-background` for quick alignment runs
* `--skip-homing` only if already homed this power-cycle *and* the gantry hasn't been touched.
