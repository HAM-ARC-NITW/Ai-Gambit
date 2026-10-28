import time, numpy as np, mujoco, mujoco.viewer, imageio

from make_scene import HERE,ROBOT
ARM_DIR = str(HERE/"generated")
(HERE/"generated").mkdir(exist_ok=True)
ARM_XML = ROBOT.as_posix()
TOWER = (0.18, -0.16)          # fallback only; the real tower position is perceived
CUBE = 0.03
CUBES = {"cube1": "red", "cube2": "green", "cube3": "blue", "cube4": "purple"}
START = {"cube1": (0.18, 0.06), "cube2": (0.18, 0.14), "cube3": (0.24, 0.09), "cube4": (0.13, 0.10)}
RGBA = {"red": "0.9 0.15 0.15 1", "green": "0.15 0.75 0.25 1",
        "blue": "0.15 0.35 0.95 1", "purple": "0.65 0.2 0.9 1"}

bodies = "".join(
    f'<body name="{n}" pos="{START[n][0]} {START[n][1]} 0.015"><freejoint/>'
    f'<geom type="box" size="0.015 0.015 0.015" rgba="{RGBA[c]}" mass="0.02" friction="2 0.1 0.01" condim="4"/></body>'
    for n, c in CUBES.items())

SCENE = f"""
<mujoco model="stack_world">
  <include file="{ARM_XML}"/>
  <compiler meshdir="{(ROBOT.parent/'assets').as_posix()}"/>
  <option timestep="0.002"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.7 0.7 0.7"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.25 0.3 0.4" rgb2="0.02 0.02 0.04" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.30 0.32 0.35" rgb2="0.22 0.24 0.27" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0.2 0 1.5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="grid"/>
    <geom type="box" pos="{TOWER[0]} {TOWER[1]} 0.0005" size="0.035 0.035 0.0005" rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0"/>
    <camera name="topcam" pos="0.18 -0.02 0.6" fovy="60"/>
    {bodies}
  </worldbody>
</mujoco>
"""
scene_path = f"{ARM_DIR}/legacy_stack_scene.xml"
open(scene_path, "w").write(SCENE)
model = mujoco.MjModel.from_xml_path(scene_path)
data = mujoco.MjData(model)

# ---------------- arm setup (names from your inspect output) ----------------
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
jid = [model.joint(n).id for n in ARM]
qadr = [int(model.jnt_qposadr[j]) for j in jid]
dadr = [int(model.jnt_dofadr[j]) for j in jid]
lim = np.array([model.jnt_range[j] for j in jid])
act = [model.actuator(n).id for n in ARM]
gact = model.actuator("gripper").id
gq_adr = int(model.jnt_qposadr[model.joint("gripper").id])
site = model.site("gripperframe").id
HOME = model.qpos0[qadr].copy()

OPEN, CLOSED = 1.2, 0.0        # if the jaw closes when it should open, swap these
HOVER = 0.12                   # travel height of the gripper tip
GRASP_DZ = 0.008               # gripper-tip height above the cube centre when grasping

# which way the gripper points, in gripperframe coordinates (found automatically)
mujoco.mj_forward(model, data)
wr = model.joint("wrist_roll").id
roll_axis = data.xaxis[wr].copy()
if np.dot(roll_axis, data.site_xpos[site] - data.xanchor[wr]) < 0:
    roll_axis = -roll_axis
APPROACH = data.site_xmat[site].reshape(3, 3).T @ roll_axis
print("gripper points along (gripperframe coords):", np.round(APPROACH, 2))
DOWN = np.array([0, 0, -1.0])

def ik(target, iters=600):
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos
    jp = np.zeros((3, model.nv)); jr = np.zeros((3, model.nv))
    for _ in range(iters):
        mujoco.mj_forward(model, d)
        R = d.site_xmat[site].reshape(3, 3)
        err_p = target - d.site_xpos[site]
        err_o = np.cross(R @ APPROACH, DOWN)
        if np.linalg.norm(err_p) < 5e-4 and np.linalg.norm(err_o) < 0.02:
            break
        mujoco.mj_jacSite(model, d, jp, jr, site)
        J = np.vstack([jp[:, dadr], 0.5 * jr[:, dadr]])
        e = np.concatenate([err_p, 0.5 * err_o])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
        d.qpos[qadr] = np.clip(d.qpos[qadr] + 0.5 * dq, lim[:, 0], lim[:, 1])
    return d.qpos[qadr].copy(), np.linalg.norm(err_p)

# ---------------- motion ----------------
def tick(viewer, n=5):
    for _ in range(n):
        mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(model.opt.timestep * n)

def drive(viewer, q_goal, grip, secs):
    q0, g0 = data.ctrl[act].copy(), float(data.ctrl[gact])
    n = max(1, int(secs / 0.01))
    for i in range(1, n + 1):
        s = 0.5 - 0.5 * np.cos(np.pi * i / n)
        data.ctrl[act] = q0 + (q_goal - q0) * s
        data.ctrl[gact] = g0 + (grip - g0) * s
        tick(viewer)

def move_to(viewer, pos, grip, secs=1.5):
    pos = np.array(pos, dtype=float)
    q_goal, err = ik(pos)
    if err > 0.01:
        print(f"  WARNING: {np.round(pos, 3)} is not reachable (IK error {err * 100:.1f} cm)")
    drive(viewer, q_goal, grip, secs)
    for _ in range(2):                                   # correct for sag
        e = pos - data.site_xpos[site]
        if np.linalg.norm(e) < 0.002:
            break
        q_goal, _ = ik(pos + e)
        drive(viewer, q_goal, grip, 0.4)

def hold(viewer, grip, secs=0.6):
    data.ctrl[gact] = grip
    for _ in range(int(secs / 0.01)):
        tick(viewer)

def go_home(viewer, secs=1.5):
    drive(viewer, HOME, OPEN, secs)

# ---------------- perception: overhead camera, colour + depth ----------------
W, H, FOVY = 640, 480, 60.0
F = (H / 2) / np.tan(np.radians(FOVY) / 2)
cam_id = model.camera("topcam").id
renderer = mujoco.Renderer(model, H, W)

def hsv(rgb):
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx, mn = f.max(-1), f.min(-1)
    c = mx - mn + 1e-6
    h = np.where(mx == r, np.mod((g - b) / c, 6), np.where(mx == g, (b - r) / c + 2, (r - g) / c + 4)) * 60
    return h, (mx - mn) / (mx + 1e-6), mx

def hue_mask(h, col):
    if col == "red":
        return (h < 15) | (h > 345)
    lo, hi = {"green": (95, 160), "blue": (200, 250), "purple": (265, 310)}[col]
    return (h > lo) & (h < hi)

def perceive():
    """Returns {cube_name: [x, y, z_centre]} in world coordinates, from the camera only."""
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="topcam")
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="topcam")
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    h, s, v = hsv(rgb)
    colored = (s > 0.5) & (v > 0.25)
    vv, uu = np.mgrid[0:H, 0:W]
    R = data.cam_xmat[cam_id].reshape(3, 3)
    cpos = data.cam_xpos[cam_id]
    found, dbg = {}, rgb.copy()
    for name, col in CUBES.items():
        m = colored & hue_mask(h, col)
        if m.sum() < 25:
            continue
        u, vp, d = uu[m], vv[m], depth[m]
        pc = np.stack([(u - W / 2) / F * d, -(vp - H / 2) / F * d, -d], 1)
        pw = cpos + pc @ R.T
        ok = (pw[:, 2] < 0.15) & (pw[:, 0] > 0.08)       # ignore the arm
        if ok.sum() < 25:
            continue
        pw, u, vp = pw[ok], u[ok], vp[ok]
        top = pw[:, 2] > pw[:, 2].max() - 0.006          # top face only
        found[name] = np.array([pw[top, 0].mean(), pw[top, 1].mean(), np.median(pw[top, 2]) - CUBE / 2])
        cu, cv = int(u[top].mean()), int(vp[top].mean())
        dbg[max(cv - 6, 0):cv + 7, cu] = 255
        dbg[cv, max(cu - 6, 0):cu + 7] = 255
    imageio.imwrite("perception.png", dbg)
    return found

# ---------------- pick and place ----------------
CAND = [(0, 0), (0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01), (0.02, 0), (-0.02, 0), (0, 0.02), (0, -0.02)]

def offset_xy(xy, cand):
    rad = np.array(xy[:2], dtype=float)
    rad /= np.linalg.norm(rad)
    return rad * cand[0] + np.array([-rad[1], rad[0]]) * cand[1]

def grasp(viewer, name, cube, cand):
    o = offset_xy(cube, cand)
    gx, gy, gz = cube[0] + o[0], cube[1] + o[1], cube[2] + GRASP_DZ
    move_to(viewer, [gx, gy, HOVER], OPEN, 1.5)
    move_to(viewer, [gx, gy, gz], OPEN, 1.0)
    hold(viewer, CLOSED, 1.0)
    jaw = data.qpos[gq_adr]
    move_to(viewer, [gx, gy, HOVER], CLOSED, 1.2)
    lifted = data.body(name).xpos[2] > 0.06              # stands in for a grip sensor
    if not lifted:
        why = "jaw closed on nothing (missed)" if jaw < 0.1 else "cube between the jaws but slipped"
        print(f"  grasp failed: jaw closed to {jaw:.2f} rad -> {why}")
        hold(viewer, OPEN, 0.5)
    return lifted

def place(viewer, tower_xy, tower_top, cand):
    o = offset_xy(tower_xy, cand)
    px, py = tower_xy[0] + o[0], tower_xy[1] + o[1]
    carry = max(HOVER, tower_top + 0.10)
    pz = tower_top + CUBE / 2 + 0.004 + GRASP_DZ
    move_to(viewer, [px, py, carry], CLOSED, 1.5)
    move_to(viewer, [px, py, pz], CLOSED, 1.0)
    hold(viewer, OPEN, 0.8)
    move_to(viewer, [px, py, carry], OPEN, 1.0)

# ---------------- main ----------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 150, -35, 0.9
    viewer.cam.lookat[:] = [0.15, -0.02, 0.05]
    data.ctrl[act] = HOME
    data.ctrl[gact] = OPEN
    for _ in range(100):
        tick(viewer)
    go_home(viewer)

    seen = perceive()
    print("PERCEPTION CHECK (camera estimate vs true position):")
    for n in CUBES:
        if n in seen:
            print(f"  {n}: camera {np.round(seen[n][:2], 3)}  true {np.round(data.body(n).xpos[:2], 3)}")
        else:
            print(f"  {n}: NOT SEEN by the camera")

    placed, calib = [], None
    for name in CUBES:
        done = False
        cands = ([calib] if calib else []) + [c for c in CAND if c != calib]
        for cand in cands:
            go_home(viewer, 1.2)
            obs = perceive()
            if name not in obs:
                print(f"{name}: not visible to the camera")
                break
            cube = obs[name]
            tower_xy, tower_top = np.array(TOWER), 0.0
            pl = [obs[p] for p in placed if p in obs]
            if pl:
                tower_xy = np.mean([p[:2] for p in pl], axis=0)
                tower_top = max(p[2] for p in pl) + CUBE / 2
            print(f"{name}: camera sees it at {np.round(cube[:2], 3)}, tower top {tower_top:.3f} m, grasp offset {cand}")
            if grasp(viewer, name, cube, cand):
                place(viewer, tower_xy, tower_top, cand)
                placed.append(name)
                calib = cand
                done = True
                print(f"{name}: placed on the tower")
                break
            print(f"{name}: retrying with a different grasp offset")
        if not done:
            print(f"{name}: gave up")

    go_home(viewer)
    obs = perceive()
    print("FINAL (camera view of the tower):")
    for n in placed:
        if n in obs:
            print(f"  {n}: xy {np.round(obs[n][:2], 3)}  centre height {obs[n][2]:.3f} m")
    while viewer.is_running():
        tick(viewer)
