"""
Print an ASCII map of per-frame max pixel value over the X-Z grid of one
slice folder, straight from frames.jsonl — no stitching needed. Use after
a bounds-survey scan to see where the beam ring is and pick raster caps.

    python beam_bounds.py data/auto_scan-*/y0150.00cm [threshold]

Rows are Z (top = up), columns are X. Cells at/below `threshold` counts
(default 12) print '.', brighter cells ramp through :-=+*#%@.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

RAMP = " .:-=+*#%@"


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    slice_dir = Path(sys.argv[1])
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0

    records = [
        json.loads(line)
        for line in (slice_dir / "frames.jsonl").read_text().splitlines()
    ]
    records = [
        r for r in records if r["Extra"].get("ScanKind") == "AutoBeamStack"
    ]

    if not records:
        print("No scan frames in this folder.")
        raise SystemExit(1)

    xs = sorted({r["GantryPosition_mm"]["x_mm"] for r in records})
    zs = sorted({r["GantryPosition_mm"]["z_mm"] for r in records}, reverse=True)

    grid = {
        (r["GantryPosition_mm"]["x_mm"], r["GantryPosition_mm"]["z_mm"]): r["Max"]
        for r in records
    }

    peak = max(grid.values()) or 1

    def glyph(value) -> str:
        if value is None:
            return " "
        if value <= threshold:
            return "."
        idx = 1 + int((len(RAMP) - 2) * (value - threshold) / max(peak - threshold, 1))
        return RAMP[min(idx, len(RAMP) - 1)]

    print(f"{slice_dir}  (peak max {peak}, threshold {threshold:g})")
    print(f"     X: {xs[0]:g} .. {xs[-1]:g} mm ({len(xs)} cols)\n")

    for z in zs:
        row = "".join(glyph(grid.get((x, z))) for x in xs)
        print(f"Z {z:8.1f} |{row}|")

    # Bounding box of cells above threshold.
    lit = [(x, z) for (x, z), v in grid.items() if v is not None and v > threshold]

    if lit:
        lit_x = sorted(x for x, _ in lit)
        lit_z = sorted(z for _, z in lit)
        print(
            f"\nBeam extent above threshold: "
            f"X {lit_x[0]:g}..{lit_x[-1]:g} mm, Z {lit_z[0]:g}..{lit_z[-1]:g} mm"
        )
        print(
            f"Suggested caps (one grid step of margin): "
            f"--x-min {lit_x[0] - 5:g} --x-max {lit_x[-1] + 5:g} "
            f"--z-min {lit_z[0] - 4:g} --z-max {lit_z[-1] + 4:g}"
        )
    else:
        print("\nNo cells above threshold — beam not found in this survey.")


if __name__ == "__main__":
    main()
