import numpy as np, mujoco
model = mujoco.MjModel.from_xml_path("SO-ARM100/Simulation/SO101/stack_scene.xml")
data = mujoco.MjData(model)
adr = lambda n: int(model.jnt_qposadr[model.joint(n).id])
rng = lambda n: model.jnt_range[model.joint(n).id]
site = model.site("gripperframe").id

mujoco.mj_forward(model, data)
wr = model.joint("wrist_roll").id
ax = data.xaxis[wr].copy()
if np.dot(ax, data.site_xpos[site] - data.xanchor[wr]) < 0:
    ax = -ax
APPROACH = data.site_xmat[site].reshape(3, 3).T @ ax

print("JOINT POSITIONS AT HOME (world xyz):")
for n in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]:
    print(f"  {n:14s}", np.round(data.xanchor[model.joint(n).id], 3))
print("  gripper tip   ", np.round(data.site_xpos[site], 3))

J = ["shoulder_lift", "elbow_flex", "wrist_flex"]
N = 45
pts = []
for a in np.linspace(*rng(J[0]), N):
    for b in np.linspace(*rng(J[1]), N):
        for c in np.linspace(*rng(J[2]), N):
            data.qpos[adr(J[0])], data.qpos[adr(J[1])], data.qpos[adr(J[2])] = a, b, c
            mujoco.mj_kinematics(model, data)
            p = data.site_xpos[site]
            v = data.site_xmat[site].reshape(3, 3) @ APPROACH
            tilt = np.degrees(np.arccos(np.clip(-v[2], -1, 1)))   # 0 = pointing straight down
            pts.append((p[0], p[2], tilt))
P = np.array(pts)

print("\nGRIPPER TIP AT TABLE HEIGHT (z 0.005-0.045 m), by how much tilt is allowed:")
for t in [10, 20, 30, 45, 60, 90, 180]:
    m = (P[:, 1] > 0.005) & (P[:, 1] < 0.045) & (P[:, 2] <= t)
    print(f"  tilt <= {t:3d} deg: " + (f"reach x from {P[m,0].min():.3f} to {P[m,0].max():.3f} m" if m.any() else "NOT reachable"))
print("\nHOVER HEIGHT (z 0.09-0.15 m):")
for t in [10, 30, 90]:
    m = (P[:, 1] > 0.09) & (P[:, 1] < 0.15) & (P[:, 2] <= t)
    print(f"  tilt <= {t:3d} deg: " + (f"reach x from {P[m,0].min():.3f} to {P[m,0].max():.3f} m" if m.any() else "NOT reachable"))
