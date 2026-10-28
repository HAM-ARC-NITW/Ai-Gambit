import json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STL = os.path.join(HERE, "meshes", "block1.stl")
SCENE_DIR = os.path.join(HERE, "generated")
os.makedirs(SCENE_DIR,exist_ok=True)
SCENE = os.path.join(SCENE_DIR, "stack_scene.xml")
N_BLOCKS = 4
RGBA = "0.8 0.1 0.5 1"
MASS = 0.02

def load_stl(path):
    raw = open(path, "rb").read()
    n = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
    if len(raw) == 84 + 50 * n:  # binary STL
        dt = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])
        return np.frombuffer(raw[84:], dtype=dt, count=n)["v"].reshape(-1, 3).astype(float)
    txt = raw.decode("ascii", "ignore")  # ascii STL
    return np.array(re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", txt), dtype=float)

v = load_stl(STL) * 0.001  # mm -> m
lo, hi = v.min(0), v.max(0)
size, ctr = hi - lo, (lo + hi) / 2
print("block size x,y,z (m):", size.round(4))
print("mesh centre offset  :", ctr.round(4), "(non-zero means your old XML put the block off-centre)")

m = MASS
I = [m/12*(size[1]**2+size[2]**2), m/12*(size[0]**2+size[2]**2), m/12*(size[0]**2+size[1]**2)]
hx, hy, hz = size / 2

bodies = ""
for i in range(N_BLOCKS):
    bodies += f"""
    <body name="bk{i}" pos="{0.2+0.08*i:.3f} 0 {hz+0.001:.4f}">
      <freejoint name="bk{i}_j"/>
      <inertial pos="0 0 0" mass="{m}" diaginertia="{I[0]:.3e} {I[1]:.3e} {I[2]:.3e}"/>
      <geom name="bk{i}_vis" type="mesh" mesh="bk_mesh" pos="{-ctr[0]:.5f} {-ctr[1]:.5f} {-ctr[2]:.5f}" contype="0" conaffinity="0" group="1" rgba="{RGBA}"/>
      <geom name="bk{i}_col" type="box" size="{hx:.5f} {hy:.5f} {hz:.5f}" friction="1.5 0.01 0.001" condim="4" group="3" rgba="0 0 0 0"/>
    </body>"""

xml = f"""<mujoco>
  <asset>
    <mesh name="bk_mesh" file="{STL}" scale="0.001 0.001 0.001"/>
  </asset>
  <worldbody>
    <camera name="topdown" pos="0.2 0 0.55" fovy="50"/>{bodies}
  </worldbody>
</mujoco>
"""
open(os.path.join(SCENE_DIR, "blocks.xml"), "w").write(xml)
json.dump({"size": size.tolist(), "n": N_BLOCKS}, open(os.path.join(HERE, "block_dims.json"), "w"))
print("wrote blocks.xml and block_dims.json")

if not os.path.exists(SCENE):
    from make_scene import build_scene
    build_scene()
    print("Refreshed the primary row scene from the measured block dimensions.")
    raise SystemExit(0)
txt = open(SCENE).read()
if "blocks.xml" not in txt:
    open(SCENE + ".bak", "w").write(txt)
    i = txt.rindex("</mujoco>")
    txt = txt[:i] + '  <include file="blocks.xml"/>\n' + txt[i:]
    open(SCENE, "w").write(txt)
    print("patched stack_scene.xml (backup: stack_scene.xml.bak)")
for ln in txt.splitlines():
    if re.search(r"block|cube", ln, re.I) and "blocks.xml" not in ln:
        print("WARNING old block-like line in scene, delete it if it is a block:", ln.strip())
