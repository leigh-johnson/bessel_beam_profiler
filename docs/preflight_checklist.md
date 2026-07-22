# Morning preflight checklist — first hardware test of `dataset auto`

Work top to bottom. Steps marked ⚙ are automated by `python cli.py
preflight`; the rest are hands-on. Convention reminder: **X** horizontal
transverse, **Y** along the beam, **Z** vertical; slices are X-Z rasters
stepped along Y.

## 0. Before touching anything

- [ ] `cd bessel_beam_simulator/profiler`, activate the Python env that
      has the Spinnaker (PySpin) wheel installed.
- [ ] Close **SpinView** (it holds the GigE camera) and close the FluidNC
      **WebUI** tab (two clients interleaving commands is asking for
      trouble). You can reopen SpinView briefly in step 4.
- [ ] FluidNC config: `idle_ms` → **255** (hold torque between moves;
      currently 250). Power-cycle after changing.
- [ ] Cables: follow the camera's Ethernet + power through the full X/Y/Z
      travel by hand — nothing snags at the extremes.

## 1. ⚙ Static checks (nothing moves)

```bash
python cli.py preflight --dataset-root data
```

Checks: Python env + PySpin import, profiler modules, disk space, DNS/TCP
to `fluidnc-sr2.local` (falls back: `--host 10.43.19.86`), FluidNC status
(Alarm is EXPECTED before homing), firmware info, camera model/serial,
exposure limits, the `AcquisitionFrameRate=3` throttle, and 3 timed test
frames (target: well under ~250 ms/frame; slower → check the frame-rate
warning and the wired-GigE note).

## 2. ⚙ Motion checks (gantry moves; confirms before each)

```bash
python cli.py preflight --motion
```

- [ ] Homing: Z retracts up first, then X and Y auto-square. Watch it.
- [ ] Beam-direction check: it moves to X60 Y20 Z-60, then +30 mm in Y,
      and asks whether the camera moved AWAY from the optic. Note the
      answer — if "no", add `--beam-direction -y` to every `dataset auto`
      call today.

## 3. Find the beam (hands-on)

- [ ] Beam on. Open SpinView (temporarily) or use
      `python cli.py calibrate ...` for a live preview.
- [ ] Jog with `python cli.py gantry move --x .. --y .. --z ..` until the
      beam core is on the sensor. Note machine **X** and **Z**.
- [ ] Close SpinView again.
- [ ] Choose caps centered on the beam, e.g. beam at (X 62, Z −58) →
      `--x-min 47 --x-max 77 --z-min -73 --z-max -43`. The calibration
      point is the center of these caps — it must contain the beam.

## 4. Measure the placement

- [ ] Move to the scan start: `gantry move --y 10` (or your `--y-start`).
- [ ] Tape-measure optic (axicon3) → camera sensor plane along the beam.
      You'll enter this in cm at the prompt.

## 5. Smoke scan (3 slices, ~5 min)

```bash
python cli.py dataset auto --dataset-root data \
    --y-start 10 --y-stop 30 --y-step 10 \
    --x-min 47 --x-max 77 --z-min -73 --z-max -43
```

Then inspect the run directory:

- [ ] `scan.log` exists and mirrors the console.
- [ ] `y####.##cm/` folder names match your tape measurement.
- [ ] `raster_metadata.json`: growth history sensible? `TruncatedSides`
      empty? (Truncated → widen the X/Z caps.) `BeamFitsInSingleFrame`
      plausible for how big the beam looked?
- [ ] `background_reference.json` + one `...-background-...` frame per
      captured slice; check a background `.jpg` looks like darkness, not
      beam.
- [ ] `calibration_result.json`: `Converged: true`, exposure sane, and the
      scan-frame `.jpg`s show the beam near the frame center.

## 6. Validate the off-axis background corner (once)

- [ ] At the dimmest slice's Y and exposure, compare a corner frame
      (beam ON) with one where you physically block the beam. If they
      match, the ambient-only assumption holds. If not, pick a different
      `--background-x/--background-z`, or fall back to
      `--background-mode ladder`.

## 7. Full placement

- [ ] Same command with `--y-stop 150`. Expect ~12–15 min of machine time
      per placement. Watch the first two slices, then let it run.
- [ ] If interrupted (Ctrl-C): the gantry is in feed-hold — `~` in the
      WebUI to resume manually, or just re-home and restart.

## Known limitations today

* No auto-resume after a WiFi drop or crash — restart the placement.
* Adaptive raster's `any` border test may overshoot the beam by ~1 frame
  per side (orientation-safe). After verifying the image↔machine axis
  mapping, `BorderTest="directional"` removes it.
* Stitching is a separate step (`dataset stitch <run>/y0100.00cm`) and
  still uses image registration, not commanded positions.
