import os, glob
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
SO101 = os.path.join(HERE, "SO-ARM100", "Simulation", "SO101")

print("mujoco version:", mujoco.__version__)
print("looking in:", SO101)
if not os.path.isdir(SO101):
    raise SystemExit("Folder not found. Is check_setup.py in the same folder as SO-ARM100?")

print("\nXML files found:")
for f in sorted(glob.glob(os.path.join(SO101, "*.xml"))):
    print("   ", os.path.basename(f))

path = None
for name in ["so101_new_calib.xml", "scene.xml", "stack_scene.xml"]:
    if os.path.exists(os.path.join(SO101, name)):
        path = os.path.join(SO101, name)
        break
if path is None:
    raise SystemExit("Could not find a robot XML file. Send me the list printed above.")

print("\nLoading", os.path.basename(path))
model = mujoco.MjModel.from_xml_path(path)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print("\nJOINTS (name, min, max in radians):")
for j in range(model.njnt):
    print(f"   {model.joint(j).name:15s} {model.jnt_range[j][0]:7.3f} {model.jnt_range[j][1]:7.3f}")

print("\nACTUATORS (name, joint it drives, strength kp, force limit):")
for a in range(model.nu):
    jn = model.joint(int(model.actuator_trnid[a, 0])).name
    print(f"   {model.actuator(a).name:15s} {jn:15s} kp={model.actuator_gainprm[a, 0]:8.2f} "
          f"forcerange={model.actuator_forcerange[a]}")

print("\nSITES:", [model.site(s).name for s in range(model.nsite)])

print("\nGRIPPER GEOMS (name, body, friction):")
for g in range(model.ngeom):
    body = model.body(int(model.geom_bodyid[g])).name
    if "gripper" in body.lower() or "jaw" in body.lower():
        print(f"   {model.geom(g).name or 'geom' + str(g):20s} {body:20s} {np.round(model.geom_friction[g], 2)}")

# ---- how far can the gripper reach down onto the table? ----
site = model.site("gripperframe").id
wr = model.joint("wrist_roll").id
ax = data.xaxis[wr].copy()
if np.dot(ax, data.site_xpos[site] - data.xanchor[wr]) < 0:
    ax = -ax
APPR = data.site_xmat[site].reshape(3, 3).T @ ax
base = data.xanchor[model.joint("shoulder_pan").id].copy()
adr = lambda n: int(model.jnt_qposadr[model.joint(n).id])
rng = lambda n: model.jnt_range[model.joint(n).id]

pts = []
for a in np.linspace(*rng("shoulder_lift"), 31):
    for b in np.linspace(*rng("elbow_flex"), 31):
        for c in np.linspace(*rng("wrist_flex"), 31):
            data.qpos[adr("shoulder_lift")] = a
            data.qpos[adr("elbow_flex")] = b
            data.qpos[adr("wrist_flex")] = c
            mujoco.mj_kinematics(model, data)
            p = data.site_xpos[site]
            v = data.site_xmat[site].reshape(3, 3) @ APPR
            tilt = np.degrees(np.arccos(np.clip(-v[2], -1, 1)))
            if 0.005 < p[2] < 0.045 and tilt < 30:
                pts.append(p - base)
pts = np.array(pts)
print("\nARM BASE (world xyz):", np.round(base, 3))
if len(pts) == 0:
    print("The gripper never reached the table pointing down. Send me this whole output.")
else:
    r = np.hypot(pts[:, 0], pts[:, 1])
    ang = np.degrees(np.arctan2(pts[:, 1].mean(), pts[:, 0].mean()))
    print(f"REACH with gripper pointing down: {r.min():.3f} to {r.max():.3f} m from the base")
    print(f"FORWARD direction of the arm: {ang:.0f} degrees (0 = along +x axis)")