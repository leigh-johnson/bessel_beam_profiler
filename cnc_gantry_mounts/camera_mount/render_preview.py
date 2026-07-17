"""Render preview PNGs of the bracket alone and with dummy camera/rail."""
import numpy as np, trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def load(p):
    m = trimesh.load(p)
    return m

def boxmesh(xmin, xmax, ymin, ymax, zmin, zmax):
    b = trimesh.creation.box(extents=[xmax-xmin, ymax-ymin, zmax-zmin])
    b.apply_translation([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2])
    return b

def cylmesh(r, h, center, axis="x"):
    c = trimesh.creation.cylinder(radius=r, height=h, sections=48)
    if axis == "x":
        c.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0,1,0]))
    c.apply_translation(center)
    return c

def draw(ax, mesh, color, alpha=1.0):
    tri = mesh.vertices[mesh.faces]
    pc = Poly3DCollection(tri, alpha=alpha, linewidths=0.05)
    pc.set_facecolor(color)
    pc.set_edgecolor((0.2,0.2,0.2,0.15))
    ax.add_collection3d(pc)

bracket = load("FLIR_BFS_rail_bracket.stl")

# dummy geometry
cam   = boxmesh(48, 78, -11, 17.98, -14.5, 14.5)                 # camera body (0.02 contact offset)
lens  = cylmesh(14, 11.8, [78+11.8/2, 3.5, 0])                   # C-mount barrel
plug  = boxmesh(48-33, 48, 3.5-6.5, 3.5+6.5, -12, 1)             # RJ45 plug+boot in the gap
carr  = boxmesh(-13, 0, -15, 15, -16, 16)                        # carriage block
rail  = boxmesh(-30, -13, -15, 15, -80, 80)                      # rail body

views = [
    ("preview_bracket.png",  [("bracket", bracket, "#f4b53f", 1.0)], (22, -35)),
    ("assembly_iso.png", [
        ("rail", rail, "#3a3a3a", 1.0), ("carr", carr, "#555555", 1.0),
        ("bracket", bracket, "#f4b53f", 1.0), ("cam", cam, "#20242a", 1.0),
        ("lens", lens, "#66707c", 1.0), ("plug", plug, "#2b6cb0", 1.0),
    ], (18, -40)),
    ("assembly_top.png", [
        ("rail", rail, "#3a3a3a", 1.0), ("carr", carr, "#555555", 1.0),
        ("bracket", bracket, "#f4b53f", 1.0), ("cam", cam, "#20242a", 1.0),
        ("lens", lens, "#66707c", 1.0), ("plug", plug, "#2b6cb0", 1.0),
    ], (88, -90)),
]

for fname, items, (elev, azim) in views:
    fig = plt.figure(figsize=(9, 7), dpi=130)
    ax = fig.add_subplot(111, projection="3d")
    allv = []
    for _, m, c, a in items:
        draw(ax, m, c, a)
        allv.append(m.vertices)
    v = np.vstack(allv)
    ctr = (v.max(0) + v.min(0)) / 2
    r = (v.max(0) - v.min(0)).max() / 2
    ax.set_xlim(ctr[0]-r, ctr[0]+r); ax.set_ylim(ctr[1]-r, ctr[1]+r); ax.set_zlim(ctr[2]-r, ctr[2]+r)
    ax.set_box_aspect([1,1,1]); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    ax.set_title(fname.replace(".png","").replace("_"," ") + "  (gold=bracket, blue=RJ45 plug)", fontsize=9)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)

# interference check: bracket vs camera/lens/plug dummies
mgr = trimesh.collision.CollisionManager()
mgr.add_object("bracket", bracket)
for name, m in [("cam", cam), ("lens", lens), ("plug", plug), ("rail", rail)]:
    hit, names = mgr.in_collision_single(m, return_names=True)
    print(f"collision bracket vs {name}: {hit}")
