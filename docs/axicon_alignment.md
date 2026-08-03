# Axicon alignment tool (`align`)

Live alignment feedback for the annulus after axicon 3, using the CNC
gantry as a roving eye. The ring (~9.5 mm+) is wider than the BFS
sensor (~7.1 x 5.3 mm), so the tool patrols stations around the ring
and refits it every lap instead of relying on any single frame.

```
python cli.py align --y 20                        # find -> orbit -> park -> stream
python cli.py align --y 20 --mode patrol          # orbit continuously (all metrics/lap)
python cli.py align --y 20 --y2 120 --mode patrol # + two-plane pointing tilt
python cli.py align --y 20 --ring-diameter 12     # sanity prior for bootstrap
python cli.py align --no-display --frames 100     # headless (SSH): log + PNGs only
```

## Two modes

**stream** (default): after the bootstrap and ONE orbit lap, the gantry
parks on the ring (brightest station, or `--park-azimuth`) and streams
single frames at a few Hz with no motion. Each frame's arc is fit with
the radius FIXED to the last orbit's value — the only well-posed
single-arc fit — giving live center drift and ring width while you
turn an adjuster. Press `o` anytime (or set `--orbit-every N`) for a
fresh full lap, which refreshes radius, roundness, and azimuthal
uniformity and re-parks; `f` re-runs the find-beam bootstrap.

**patrol**: orbit continuously; every lap refreshes ALL metrics
(~10–20 s feedback latency, full-ring truth every time). Required for
the two-plane tilt readout (`--y2`).

Stations sit ON the fitted ring, so by default the composite has a
blind spot in the middle — fine for a thin annulus, not for a broad
fringe field. `--cover disk` adds a half-radius inner ring of stations
plus one at the center; those interior frames fill the composite but
are EXCLUDED from the ring fit, so interior diffraction light cannot
drag the geometry.

## What it does

1. **Bootstrap** (reused from `dataset auto`): find-beam sweep locates
   structured light along a Z column at `--probe-x` (default: middle of
   the machine X envelope — make sure that crosses the ring), headless
   exposure calibration runs there, then one **off-axis background
   frame** is captured at the far machine corner (it anchors all signal
   thresholds — recaptured automatically when the exposure drifts
   >30%), and two vertical **survey columns** measure the ring's
   chords. Each column's chord is the ENVELOPE of its lit extent (a
   chord of the beam's outer boundary — valid for thin rings AND broad
   bands whose columns miss the dark hole); two chords pin down center
   and radius (`h^2 = r^2 - (x - cx)^2`), no prior needed. If both
   second columns at `+/---survey-dx` see nothing — a COMPACT pattern
   like the focused Bessel region after axicon 3 can be narrower than
   the default 5 mm spacing — the survey automatically retries with a
   dx scaled to the extent column 1 measured (~0.7x its half-extent).
   The first full lap's fit then replaces the seed unclamped; after
   that, estimate changes are clamped to `--max-shift` (default 3 mm)
   per lap.
2. **Patrol cycle**: `--stations` (default 8) points spaced around the
   ring estimate, one frame each. Lit pixels are mapped to machine
   coordinates (composite.py's verified FlipX/FlipZ orientation and
   3.45 um pixel scale) and collapsed to ring-locus points by angular
   bin. A cycle takes roughly `stations x (move + exposure)` — expect
   ~10–20 s per lap.
3. **Fit + metrics per cycle**, shown in the live window and appended
   to `alignment_log.jsonl`:
   - **center** — Kasa circle fit, machine X/Z (mm)
   - **offset** — center minus the reference (um). Press `r` to zero
     the readout at the current center, then adjust the optic and
     watch dX/dZ.
   - **round** — ellipse fit (Halir–Flusser): minor/major axis ratio
     and major-axis angle. Ellipticity is the axicon-tilt signature.
   - **azimuth** — ring brightness vs angle (per-bin peak, max-merged
     across stations): min/max ratio, CV, brightest/dimmest angle.
     Azimuthal nonuniformity is the input-beam-decenter signature; the
     dim side tells you which way.
   - **width** — mean radial FWHM of the ring (2.355 x the
     intensity-weighted radial std per angular bin); with the fitted
     radius this checks the spacing/cone-angle geometry. Note: pixels
     below the signal threshold are excluded, which truncates the
     wings — the readout runs ~20% below the true FWHM, so treat it as
     a RELATIVE metric (watch it change, don't quote it absolutely).
   - **tilt** (with `--y2`) — cycles alternate between the two machine-Y
     planes; the ring-center difference over the beam-path distance is
     the pointing tilt in mrad (sign follows `--beam-direction`,
     default `-y` as verified on this rig).
   - **coverage** — fraction of the ring the patrol actually saw.
     Stations outside the machine envelope are marked and skipped.

Angles are measured in the machine X-Z frame: 0 deg = +X, 90 deg = +Z
(up), CCW looking downstream.

Signal thresholds are referenced to the measured off-axis background
(background p99 + `--signal-margin`). If the background capture fails,
the fallback references each frame's own median — fine for beams that
cover a small fraction of the frame, but a broad beam that FILLS a
frame makes the median read as signal and dim arcs vanish (this
exactly happened on the wide axicon-2 band, 2026-07-27).

## The preview window

Left: the patrolled ring stitched in machine coordinates with the
fitted ellipse (cyan), fitted center (cyan +), reference (green +),
and station markers (filled = signal, open = dark, x = unreachable).
Right: metrics panel and a center-offset history plot per cycle.

Keys: `r` = set reference to the current fitted center, `o` = run an
orbit lap now (stream mode), `f` = re-run the find-beam bootstrap
(stream mode), `q` = quit (closing the window also quits).

Each `r` press also saves a `preview_r=<coords>.png` snapshot into the
run directory, named with the reference center it just set (e.g.
`preview_r=X60.123_Z-59.876.png`; two-plane runs get one
`Y<plane>_X<x>_Z<z>` group per plane) — a permanent record of each
reference-set moment, since `preview_latest.png` keeps being
overwritten.

## Fast feedback recipe (turning adjusters)

Patrol laps are for surveying; for a tight adjust-watch-adjust loop
use stream mode at ONE plane and cap the exposure:

```
python cli.py align --optic axicon2 --y 10 --probe-x 75 \
    --max-exposure 100000 --feed 1500
```

- stream (the default mode) parks after one orbit and updates center
  offset + width every frame — feedback latency = exposure + fit,
  fractions of a second, no motion.
- `--max-exposure` stops the dim-beam calibration from settling at
  300+ ms/frame: the background-referenced threshold tracks dim rings
  fine, and lap/stream rate scales directly with exposure.
- `--feed` (mm/min) dominates lap time at low values — 300 means 5 mm/s
  gantry moves and ~24 s per Y-plane change. Use the fastest feed the
  rig moves cleanly at.
- Press `r` to zero, turn the adjuster, watch dX/dZ live. Press `o`
  for a full lap (radius/roundness/uniformity) after each coarse
  adjustment. Save `--mode patrol --y2 --cover disk` for the
  before/after survey, not the live loop.

In the composite, GRAY pixels mean "no station frame imaged here"
(coverage gaps between the ring and fill bands) — only black/purple
through yellow is actual beam data. For a gap-free full image of the
plane, use `dataset auto`, which rasters the whole slice.

## Diverging cones (e.g. after axicon 1)

Each Y plane bootstraps and tracks its own radius, so a cone whose
diameter grows along the beam works out of the box. Two things help:

- Per-plane priors: `--ring-diameter <d at --y> --ring-diameter2
  <d at --y2>` (the prior is only a sanity gate on the chord survey;
  without `--ring-diameter2` the `--y` value gates both planes).
- The two-plane run reports **cone** (mrad, + = ring expands going
  downstream): the radius spread rate between the planes. After
  axicon 1 this measures the deflection angle directly (compare to
  (n-1)*alpha); after axicon 2 it should read ~0 (collimation check).
  Remember the radius is the band's intensity-weighted centroid and is
  exposure-sensitive — pin `--max-exposure` when comparing.

## Focused Bessel region (after axicon 3)

In the focused region the pattern is a compact spot (bright core +
close-in fringes), typically a few mm across or less — not a thin
annulus. The tool still tracks it: the envelope chords solve the
spot's outer boundary, and the per-angular-bin "ring locus" of a
filled spot still centers the circle fit, so **center**, **offset**,
and the two-plane **tilt** stay meaningful. Radius / width / roundness
read the spot's intensity-weighted envelope instead of a band
centroid. Parameter notes:

- The survey auto-tightens its column spacing for compact beams, but
  `--survey-dx 1` (or so) skips the two wasted full-Z columns at the
  default 5 mm spacing (~1 min each).
- DROP `--ring-diameter` here unless it's the spot's true envelope
  diameter at that plane (use the Jul 29 r–y map). A too-big prior
  (e.g. 10 mm for a ~2 mm spot) trips the sanity gate, which then
  REPLACES the measured radius with the prior — the first patrol lap
  orbits out in the dark and the ring is lost immediately.
- Fewer stations cover a tiny orbit fine (`--stations 8`), and
  `--cover ring` is enough: one frame already spans the whole spot.

## Other optics (e.g. after axicon 2)

The tool is optic-agnostic — it finds and fits whatever ring the
camera can reach. To check the annulus after axicon 2 (the relay feed
to axicon 3), tag the run with `--optic axicon2` and prefer
`--mode patrol --y2 <far plane>`:

- center drift between the two planes = pointing after axicon 2
- fitted **radius vs Y** (compare the two planes' cycles in
  `alignment_log.jsonl`) = collimation check on the 1<->2 spacing: a
  collimated relay has the same radius at both planes.

## Self-healing (inherited from the auto-scan stack)

- Exposure: a cycle-level servo halves exposure on saturation and
  doubles it when everything is dim; a camera reconnect (GigE -1024)
  re-applies the current exposure via the AutoScanSession restore hook.
- Ring lost for 2 consecutive cycles/frames → an orbit relocates it,
  and a lost orbit escalates to a full find-beam bootstrap. In stream
  mode a failed re-find (beam blocked) logs, backs off, and keeps
  streaming rather than dying — unblock the beam and it recovers.
- Ctrl-C sends a feed hold (`!`) to the gantry, same as `dataset auto`.

Tag runs with what you changed on the bench so the run directories are
interpretable later: `--optic axicon2 --notes "camera y2 at L_12=290mm"
--notes "walked input mirror +1/4 turn CW"` — every `--notes` string is
recorded in the run's `align_session.json` and echoed into `align.log`.

## Outputs

Each session makes one timestamped run directory under
`--dataset-root` (default `data/align-runs/`):

- `alignment_log.jsonl` — one line of metrics per cycle (the
  before/after record of the alignment session)
- `preview_latest.png` / `preview_final.png` — window snapshots
  (also written in `--no-display` mode, so a headless run can be
  watched by refreshing the PNG)
- `preview_r=<coords>.png` — one snapshot per `r` press, named with
  the reference center coordinates set at that moment
- `align.log` — full log, same format as scan.log

## Module map

- `align_axicon.py` — patrol session + all pure geometry (arc
  extraction, circle/ellipse fits, chord bootstrap, uniformity);
  wraps `AutoScanSession` for motion, find-beam, calibration, and
  reconnect handling.
- `align_preview.py` — matplotlib window; owns no hardware.
- `align_cli.py` — `align` command; connection/limit-validation flow
  mirrors `dataset auto`.
- `test_align_axicon.py` — synthetic-ring rig (fake camera renders
  whatever the fake gantry points at) + fit unit tests.
