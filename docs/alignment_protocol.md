# Staged alignment protocol

One optic at a time. Sign off each stage before placing the next;
after sign-off, never touch that stage's knobs again (if you must,
re-run its check). Per stage, three DOFs → three sensors, in order:

1. **Incidence (tip/tilt)**: retro-reflect the axicon's flat face back
   to an iris at the source. First-order in tilt (1 mrad = 2 mm at
   1 m). Do NOT dial tilt against ring ellipticity — it's second-order
   and hides degrees.
2. **Centering (XZ)**: azimuthal uniformity — first-order; the bright
   sector points along the decenter.
3. **Axis straightness**: center vs machine Y. The gantry Y travel is
   the reference axis; slope = tilt, ladder residuals = rail wiggle.

## Stage 1 — input beam (no optics on table)

```
python cli.py align --mode gaussian --optic input --y 130 --y-stop 10 --planes 5
```

Ladder loops until `q`. Steer far mirror (angle) + near mirror
(position); watch the slope readout. Sign-off: |slope| < 0.2 mrad
both axes, residuals flat; record w0 and astigmatism (the 2×2 mosaic
handles the beam overfilling the sensor).

## Stage 2 — axicon 1

Retro-reflect first. Then center, at a plane in the L12 region:

```
python cli.py align --mode patrol --optic axicon1 --y 20 --max-exposure 50000
```

Metrics refit live every station (~1–2 s). Dial uniformity modulation
down; press `r` to zero, `--notes` the result. Then verify
straightness + cone over two planes:

```
python cli.py align --mode patrol --optic axicon1 --y 130 --y2 10 \
    --ring-diameter <d@130> --ring-diameter2 <d@10>
```

Sign-off: modulation < ~10%, center straight vs Y, cone ≈ 40.1 mrad
(±few %).

## Stage 3 — axicon 2

Retro-reflect first. Center (same patrol command, `--optic axicon2`,
one plane). Exit criterion — the annulus leaves COLLIMATED:

```
python cli.py align --mode patrol --optic axicon2 --y 130 --y2 10
```

Sign-off: cone ≈ 0 (±0.5 mrad), radius ≈ θ·L12, center straight.
An uncollimated annulus here is what a railed kr reads like in
stage 4.

## Stage 4 — axicon 3

Retro-reflect first. Then core mode (default for axicon3):

```
python cli.py align --optic axicon3 --y <core plane> --probe-x <x crossing the core>
```

Jog Y (arrows) to map the zone. Sign-off: kx ≈ kz, kr within a few %
of ideal, rms/A_fit < 0.01, zone extent ≈ z_max = w0/θ.
**[FIT RAILED]** or rms/A ≫ 0.01 = the model doesn't apply at this
plane (wrong plane / annulus light) — jog, don't turn knobs.

## Rules

- Pin exposure when comparing runs (`--max-exposure`).
- Ambiguous signature? Rotate the axicon 90°: rotates with it =
  manufacturing defect; fixed in lab frame = alignment.
- Scorecard at every sign-off: `--notes` + `r` snapshot. "Aligned" =
  thresholds recorded and passed.
- Motion is firmware-capped at 500 mm/min, accel 100 mm/s²
  (config.yaml, 2026-08-05; NEMA 11 motors — don't raise run_amps).
  The CLI's `--feed 1500` default just means "run at the cap".
