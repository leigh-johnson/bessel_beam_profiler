"""Preview renders + clearance checks for both Z-mount variants."""
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
    pc.set_facecolor(color); pc.set_edgecolor((0.2,0.2,0.2,0.12))
    ax.add_collection3d(pc)

def scene_png(fname, items, elev, azim, title):
    fig = plt.figure(figsize=(8.5,7), dpi=130)
    ax = fig.add_subplot(111, projection="3d")
    allv=[]
    for m,c,a in items: draw(ax,m,c,a); allv.append(m.vertices)
    v=np.vstack(allv); ctr=(v.max(0)+v.min(0))/2; r=(v.max(0)-v.min(0)).max()/2
    ax.set_xlim(ctr[0]-r,ctr[0]+r); ax.set_ylim(ctr[1]-r,ctr[1]+r); ax.set_zlim(ctr[2]-r,ctr[2]+r)
    ax.set_box_aspect([1,1,1]); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    ax.set_title(title, fontsize=9); fig.tight_layout(); fig.savefig(fname, bbox_inches="tight"); plt.close(fig)
    print("wrote", fname)

def check(bracket, dummies):
    mgr = trimesh.collision.CollisionManager(); mgr.add_object("bracket", bracket)
    for name, m in dummies:
        print(f"  collision bracket vs {name}: {mgr.in_collision_single(m)}")

TABLE = -60.0  # carriage face assumed 60mm above breadboard

# ---------- Variant 1: FaceUp ----------
v1 = trimesh.load("ZMount_Var1_FaceUp.stl")
xbody = boxm(-150,150,-15,15,-31,-3.02)
xcarr = boxm(-16,16,-15,15,-3.02,-0.02)
zrail = boxm(-15,15,29.02,57,TABLE+3,TABLE+3+275)     # base against clamp face, bottom 3mm above table
table = boxm(-160,160,-40,120,TABLE-4,TABLE-0.5)
items = [(table,"#8a8f96",0.35),(xbody,"#3a3a3a",1),(xcarr,"#555",1),(v1,"#f4b53f",1),(zrail,"#20242a",0.9)]
scene_png("zmount_v1_iso.png", items, 16, -50, "Variant 1 FaceUp — Z rail hangs beside X stage (gold=bracket)")
scene_png("zmount_v1_end.png", items, 4, -92, "Variant 1 — end view: 3mm body clearance, rail to table")
print("V1 checks:"); check(v1, [("X body",xbody),("X carriage",xcarr),("Z rail",zrail),("table",table)])
