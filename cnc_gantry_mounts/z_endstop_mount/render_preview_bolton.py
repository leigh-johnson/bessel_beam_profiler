"""Preview renders for the bolt-on KW12 endstop mount (matplotlib, no GPU)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh as stlmesh
import numpy as np


def add(ax, path, color, alpha=1.0):
    m = stlmesh.Mesh.from_file(path)
    coll = Poly3DCollection(m.vectors, alpha=alpha)
    coll.set_facecolor(color)
    coll.set_edgecolor("none")
    ax.add_collection3d(coll)
    return m.vectors.reshape(-1, 3)


def render(parts, out, elev, azim, title):
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    pts = np.vstack([add(ax, p, c, a) for p, c, a in parts])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    ctr, rng = (lo + hi) / 2, (hi - lo).max() / 2
    for f, c, r in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), ctr, [rng] * 3):
        f(c - r, c + r)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=110, facecolor="white")
    plt.close(fig)
    print("wrote", out)


PART = ("endstop_bolton_kw12.stl", "#2c5f8a", 1.0)          # Berkeley-ish blue
SW   = ("switch_mock.stl", "#c4342b", 1.0)                  # red switch
RAIL = ("rail_mock.stl", "#555555", 0.35)
CARR = ("carriage_mock.stl", "#999999", 0.55)

render([RAIL, CARR, PART, SW], "preview_bolton_assembly.png",
       elev=28, azim=-55, title="bolt-on KW12 mount — carriage at trip (gray), rail+motor block (dark)")
render([RAIL, CARR, PART, SW], "preview_bolton_back.png",
       elev=-30, azim=-125, title="back face — M3 screws into the rail's tapped pair")
render([PART], "preview_bolton_part.png",
       elev=25, azim=-40, title="printed part alone")
