# Automated gantry scans (FluidNC + `dataset auto`)

How the profiler talks to the Jackpot3/FluidNC controller and runs the
automated Z-stack dataset collection.

## How G-code gets to FluidNC

FluidNC speaks the plain-text GRBL protocol over a raw TCP socket on port
23 (the Telnet server on `fluidnc-sr2.local` / 10.43.19.86 — same board the
WebUI uses). `profiler/fluidnc_stage.py` wraps this:

* Send a newline-terminated line (`$H`, `G53 G1 X60 Y80 Z-50 F400`), read
  `ok` / `error:N`.
* `ok` only means *queued*. Motion completion = poll the realtime `?` query
  and parse `<Idle|MPos:60.000,80.000,-50.000|...>` until state is `Idle`
  and MPos matches the target.
* All moves are absolute **machine** coordinates (`G53 G1`), same as
  `rangetest.gcode`, so `ScanPoint.GantryPosition_mm` == MPos.
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
python cli.py gantry move --x 60 --y 80 --z -50
python cli.py gantry send '$SS'              # any raw command
```

## Run a dataset collection

```bash
python cli.py dataset auto --dataset-root data \
    --z-start -120 --z-stop -5 --z-step 10 \
    --x-min 45 --x-max 75 --x-step 5 \
    --y-min 65 --y-max 95 --y-step 4
```

Workflow per placement (repeats for as many placements as you want):

1. Prompts for optic configuration and placement ID, then homes (`$H`).
2. You measure the optic→sensor distance with the camera at machine
   Z = `--z-start` and enter it (cm). Table z for every slice is derived
   from this: `table_z = measured + (machineZ − z_start)` (+Z is up, along
   the beam).
3. For each machine Z: the camera moves to the calibration point (raster
   center), exposure auto-calibrates headlessly (seeded from the previous
   z; halve when saturated, proportional step up when dim, accept when max
   is in [60%, 100%) of the saturation threshold with zero saturated
   pixels), then **backgrounds** are captured (see below), then the XY
   raster runs.
4. At the end you can move the gantry to a new placement; it re-homes and
   asks for a fresh distance measurement, and a new run directory begins.

### Background modes (`--background-mode`)

* `offaxis` (default): at each Z, right after calibration, the camera
  drives to an XY position outside the beam (default: the machine-limit
  corner farthest from the raster; override with `--background-x/-y`) and
  captures `--background-shots` frames at that slice's calibrated
  exposure. Exact exposure match per slice, tracks drift over the run, no
  manual beam blocking — it corrects for ambient light (plus whatever
  stray scatter reaches the off-axis position). **Validate once**: at the
  highest-exposure (dimmest) slice, compare a corner frame with the beam
  on vs. physically blocked; if they match, the corner is beam-free.
* `ladder`: once per placement, you block the beam when prompted and a
  log-spaced exposure ladder is captured (default 10 rungs, 25 µs → 100 ms,
  3 shots each) into `background/`. At analysis time subtract the rung
  nearest each frame's exposure (or interpolate — background is ~linear in
  exposure). Use this if the off-axis position can't be made beam-free.
* `none` (or `--skip-background`): quick alignment runs.

`Ctrl-C` sends a feed hold (`!`) to the gantry before exiting.

## Output layout

```
data/auto_scan-2026-07-22_14-01-05/        # one run dir per placement
    run_metadata.json
    auto_scan_setup.json                   # placement, ranges, ladder, optics
    camera_settings.json
    frames.jsonl                           # global manifest (all frames)
    background/                            # ladder mode only
        frames.jsonl
        placement-01-exp0000100.0us-...-shot0000.npy  (+ .jpg)
    z0100.00cm/                            # table z in the folder name
        frames.jsonl                       # per-z manifest (stitchable)
        calibration_result.json            # exposure, max, converged
        calibrated_camera_settings.json
        placement-01-background-...-shot0000.npy      # offaxis mode: the
                                           # slice's exposure-matched background
        placement-01-...-gantryx...y...z...-shot0000.npy  (+ .jpg)
    z0101.00cm/
        ...
```

Every frame's manifest record carries `Exposure_us`, `MachineZ_mm`,
`TableZ_mm`, `SensorZReference`, and `ScanKind`
(`AutoZStack` / `Background`; off-axis backgrounds also record
`BackgroundMode: OffAxisAmbient` and their gantry position). In offaxis
mode, subtraction is trivial: each z folder contains its own
exposure-matched background. In ladder mode, pick the rung nearest each
frame's exposure or interpolate (background is ~linear in exposure).

## Defaults worth knowing

* Feed 400 mm/min (machine max_rate is 500); 0.2 s settle after Idle.
* XY steps 5 mm × 4 mm ≈ 25–30 % overlap for the BFS-PGE-31S4M's
  7.1 × 5.3 mm sensor, for stitching.
* `--skip-background` for quick alignment runs; `--skip-homing` only if
  already homed this power-cycle *and* the gantry hasn't been touched.
* For measurement runs remember the config note: flip `idle_ms` to 255 so
  holding current stays on between moves.
