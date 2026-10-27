import numpy as np
import mujoco
from PIL import Image
from env import RowEnv

env = RowEnv()
m, d = env.model, env.data
site = m.site("gripperframe").id
pads = [m.geom("static_finger_pad").id, m.geom("moving_finger_pad").id]
jid = m.joint("gripper").id
adr = int(m.jnt_qposadr[jid])
lo, hi = m.jnt_range[jid]

results = {}
for label, val in (("gripper at its LOW end", lo), ("gripper at its HIGH end", hi)):
    d.qpos[adr] = val
    mujoco.mj_kinematics(m, d)
    R = d.site_xmat[site].reshape(3, 3)
    s = d.site_xpos[site].copy()
    dist = float(np.linalg.norm(d.geom_xpos[pads[0]] - d.geom_xpos[pads[1]]))
    results[label] = (val, dist)
    print(f"\n{label} (joint = {val:.2f}):")
    for name, g in zip(("static pad", "moving pad"), pads):
        rel = (d.geom_xpos[g] - s) @ R
        print(f"   {name}: {np.round(rel * 100, 1)} cm from the tip point (x, y, z in the tip frame)")
    print(f"   distance between the pads: {dist * 100:.1f} cm")

(lab_a, (va, da)), (lab_b, (vb, db)) = results.items()
closed_val, closed_d = (va, da) if da < db else (vb, db)
open_val, open_d = (vb, db) if da < db else (va, da)
print(f"\nClosed end ({closed_val:.2f}): pads {closed_d * 100:.1f} cm apart. "
      f"Open end ({open_val:.2f}): pads {open_d * 100:.1f} cm apart.")
print("GOOD if the closed distance is under about 1.5 cm and the open distance is a few cm more than the block.")

d.qpos[adr] = open_val
mujoco.mj_kinematics(m, d)
cam = mujoco.MjvCamera()
cam.lookat[:] = d.site_xpos[site]
cam.distance, cam.azimuth, cam.elevation = 0.14, 90, -25
env.renderer.update_scene(d, camera=cam)
Image.fromarray(env.renderer.render()).save("pads_open.png")
print("saved pads_open.png (pink pad = fixed finger, blue pad = moving finger)")    v