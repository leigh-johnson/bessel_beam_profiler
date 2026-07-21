"""Assembly render + clearance checks: X rail bridging Y1/Y2 via two adapter plates."""
import numpy as np, trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def boxm(xmin,xmax,ymin,ymax,zmin,zmax):
    b = trimesh.creation.box(extents=[xmax-xmin,ymax-ymin,zmax-zmin])
    b.apply_translation([(xmin+xmax)/2,(ymin+ymax)/2,(zmin+zmax)/2]); return b

def draw(ax, mesh, color, alpha=1.0):
    pc = Poly3DCollection(mesh.vertices[mesh.faces], alpha=alpha, linewidths=0.05)
    pc.set_facecolor(color); pc.set_edgecolor((0.2,0.2,0.2,0.12)); ax.add_collection3d(pc)

plate = trimesh.load("XtoY_bridge_adapter.stl")

# world: x = bridge direction, y = Y-travel, z up
SEP = 177.8                      # Y1-Y2 pattern separation (7 inches)
FACE = 42.0                      # Y-carriage face height above breadboard (foot 8 + stage ~34)
pl1 = plate.copy(); pl1.apply_translation([-SEP/2, 0, FACE])
pl2 = plate.copy(); pl2.apply_translation([+SEP/2, 0, FACE])

y1 = boxm(-SEP/2-15, -SEP/2+15, -187.5, 187.5, 8, FACE-0.02)     # Y rail bodies (on 8mm feet)
y2 = boxm(+SEP/2-15, +SEP/2+15, -187.5, 187.5, 8, FACE-0.02)
# X rail across, base on plate tops (FACE+10), band offset +25 in y, screws at pair 175 apart
xrail = boxm(-137.5, 137.5, 25-15, 25+15, FACE+10.02, FACE+10.02+28)
xmotor = boxm(-137.5-0, -137.5+30, 25-15, 25+15, FACE+10.02, FACE+10.02+34)
table = boxm(-200, 200, -200, 200, -4, -0.5)

items = [(table,"#8a8f96",0.3),(y1,"#3a3a3a",1),(y2,"#3a3a3a",1),
         (pl1,"#f4b53f",1),(pl2,"#f4b53f",1),(xrail,"#20242a",0.95),(xmotor,"#444",0.95)]
def scene(fname, elev, azim, title):
    fig = plt.figure(figsize=(9,7), dpi=130); ax = fig.add_subplot(111, projection="3d")
    allv=[]
    for m,c,a in items: draw(ax,m,c,a); allv.append(m.vertices)
    v=np.vstack(allv); ctr=(v.max(0)+v.min(0))/2; r=(v.max(0)-v.min(0)).max()/2
    ax.set_xlim(ctr[0]-r,ctr[0]+r); ax.set_ylim(ctr[1]-r,ctr[1]+r); ax.set_zlim(ctr[2]-r,ctr[2]+r)
    ax.set_box_aspect([1,1,1]); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    ax.set_title(title, fontsize=9); fig.tight_layout(); fig.savefig(fname, bbox_inches="tight"); plt.close(fig)
    print("wrote", fname)

scene("xy_bridge_iso.png", 24, -55, "X rail bridging Y1/Y2 on two adapter plates (gold), 177.8mm separation")
scene("xy_bridge_top.png", 88, -90, "top view — rail band offset 25mm from carriage centers, pairs slide in channels")

# checks: screws at rail pair positions must be inside channel travel
for xc, plx in [(-87.5, -SEP/2), (87.5, +SEP/2)]:   # rail pairs at +-175/2
    off = xc - plx
    ok = abs(off) <= 12.5
    print(f"rail screw offset in channel at plate x={plx:+.1f}: {off:+.1f}mm (|off|<=12.5: {ok})")
    assert ok

mgr = trimesh.collision.CollisionManager(); mgr.add_object("p1", pl1); mgr.add_object("p2", pl2)
for name, m in [("Y1", y1), ("Y2", y2), ("X rail", xrail), ("X motor", xmotor)]:
    print(f"collision plates vs {name}: {mgr.in_collision_single(m)}")
mgr2 = trimesh.collision.CollisionManager(); mgr2.add_object("xrail", xrail); mgr2.add_object("xmotor", xmotor)
for name, m in [("Y1", y1), ("Y2", y2)]:
    print(f"collision X rail/motor vs {name}: {mgr2.in_collision_single(m)}")
