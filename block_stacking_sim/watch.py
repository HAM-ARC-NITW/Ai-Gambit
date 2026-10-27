import time
import numpy as np
import mujoco
import mujoco.viewer
from env import RowEnv, JOINTS

BLOCK = 1          # which block to pick (0 to 3)
HOVER_Z = 0.09     # fingertip hover height (m)
MAX_TILT = 40      # how far the gripper may lean from straight down while grasping (degrees)
PLACE_TILT = 60    # more lean is allowed while carrying to the square
SPEED = 1.0        # replay speed: 1.0 = real time, 0.5 = half speed
# one entry per grasp attempt: (change in grasp height, sideways shift of the jaws), in metres
ATTEMPTS = [(0.0, 0.0), (-0.005, 0.0), (-0.005, 0.006),
            (-0.005, -0.006), (0.005, 0.0), (-0.008, 0.0)]

env = RowEnv(render=False)
m, d = env.model, env.data
SITE = m.site("gripperframe").id
BH = 2 * m.geom_size[m.geom("bk0_geom").id][2]
qadr = {n: int(m.jnt_qposadr[m.joint(n).id]) for n in JOINTS}
dadr = {n: int(m.jnt_dofadr[m.joint(n).id]) for n in JOINTS}
rng_ = {n: m.jnt_range[m.joint(n).id] for n in JOINTS}
act = dict(zip(JOINTS, env.act))
IK_J = JOINTS[:4]
bid = env.blocks[BLOCK]
bgeom = m.geom(f"bk{BLOCK}_geom").id
sq_c = d.geom_xpos[env.sq].copy()
sq_half = float(m.geom_size[env.sq][0])
sq_top = float(sq_c[2] + m.geom_size[env.sq][2])

pan_id = m.joint("shoulder_pan").id
base = d.xanchor[pan_id].copy()
pan_sign = 1.0 if d.xaxis[pan_id][2] > 0 else -1.0
wr = m.joint("wrist_roll").id
ax = d.xaxis[wr].copy()
if np.dot(ax, d.site_xpos[SITE] - d.xanchor[wr]) < 0:
    ax = -ax
APPR = d.site_xmat[SITE].reshape(3, 3).T @ ax
APPR = APPR / np.linalg.norm(APPR)

fixed_g, moving_g = [], []
for g in range(m.ngeom):
    b = m.body(int(m.geom_bodyid[g])).name.lower()
    if "jaw" in b:
        moving_g.append(g)
    elif "gripper" in b:
        fixed_g.append(g)

# ---------- silent simulation that records every frame ----------
REC_EVERY = max(1, round(1 / (30 * m.opt.timestep)))     # about 30 frames per second
frames = []
step_count = 0


def step_sim():
    global step_count
    mujoco.mj_step(m, d)
    step_count += 1
    if step_count % REC_EVERY == 0:
        frames.append(d.qpos.copy())


def settle(sec):
    for _ in range(int(sec / m.opt.timestep)):
        step_sim()


def goto(q, T=1.0):
    start = {n: float(d.ctrl[act[n]]) for n in JOINTS}
    n = max(1, int(T / m.opt.timestep))
    for i in range(1, n + 1):
        s = i / n
        s = s * s * (3 - 2 * s)
        for k, v in q.items():
            d.ctrl[act[k]] = start[k] + s * (v - start[k])
        step_sim()


# ---------- geometry helpers ----------
def skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def verts(geoms, dd):
    out = []
    for g in geoms:
        xp, xm = dd.geom_xpos[g], dd.geom_xmat[g].reshape(3, 3)
        if m.geom_type[g] == int(mujoco.mjtGeom.mjGEOM_MESH):
            did = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[did]: m.mesh_vertadr[did] + m.mesh_vertnum[did]]
            out.append(xp + v @ xm.T)
        else:
            sz = m.geom_size[g].copy()
            sz[sz == 0] = sz[0]
            c = np.array([[a, b, c_] for a in (-1, 1) for b in (-1, 1) for c_ in (-1, 1)]) * sz
            out.append(xp + c @ xm.T)
    return np.vstack(out)


def tips(dd):
    """middle of the jaw gap, direction the jaws open in (horizontal),
    how far the jaws stick out past the tip point, and the gap width."""
    A = dd.site_xmat[SITE].reshape(3, 3) @ APPR
    Pf, Pm = verts(fixed_g, dd), verts(moving_g, dd)
    tf = Pf[Pf @ A > (Pf @ A).max() - 0.015]
    tm = Pm[Pm @ A > (Pm @ A).max() - 0.015]
    mid = 0.5 * (tf.mean(0) + tm.mean(0))
    dv = tm.mean(0) - tf.mean(0)
    U = dv.copy()
    U[2] = 0
    U = U / (np.linalg.norm(U) + 1e-9)
    protr = max((Pf @ A).max(), (Pm @ A).max()) - dd.site_xpos[SITE] @ A
    return mid, U, float(protr), float(np.linalg.norm(dv))


def ik(target, wo, rand, roll):
    dd = mujoco.MjData(m)
    dd.qpos[:] = d.qpos
    dd.qpos[qadr["wrist_roll"]] = roll
    rel = target[:2] - base[:2]
    pan = pan_sign * np.arctan2(rel[1], rel[0])
    dd.qpos[qadr["shoulder_pan"]] = np.clip(pan, *rng_["shoulder_pan"])
    if rand is not None:
        for n in IK_J[1:]:
            dd.qpos[qadr[n]] = rand.uniform(*rng_[n])
    dofs = [dadr[n] for n in IK_J]
    down = np.array([0.0, 0.0, -1.0])
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(300):
        mujoco.mj_forward(m, dd)
        a = dd.site_xmat[SITE].reshape(3, 3) @ APPR
        ep = target - dd.site_xpos[SITE]
        eo = down - a
        if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(eo) < 0.05:
            break
        mujoco.mj_jacSite(m, dd, jp, jr, SITE)
        J = np.vstack([jp[:, dofs], wo * (-skew(a) @ jr[:, dofs])])
        e = np.hstack([ep, wo * eo])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        nrm = np.linalg.norm(dq)
        if nrm > 0.15:
            dq *= 0.15 / nrm
        for n, x in zip(IK_J, dq):
            dd.qpos[qadr[n]] = np.clip(dd.qpos[qadr[n]] + x, *rng_[n])
    mujoco.mj_forward(m, dd)
    a = dd.site_xmat[SITE].reshape(3, 3) @ APPR
    tilt = float(np.degrees(np.arccos(np.clip(-a[2], -1, 1))))
    err = float(np.linalg.norm(target - dd.site_xpos[SITE]))
    return {n: float(dd.qpos[qadr[n]]) for n in IK_J}, err, tilt


def solve(target, roll, max_tilt):
    rand = np.random.default_rng(1)
    tries = [(0.4, None), (0.8, None), (0.15, None)]
    tries += [(w, rand) for w in (0.8, 0.4) for _ in range(15)]
    best = None
    for wo, r in tries:
        q, err, tilt = ik(target, wo, r, roll)
        if err < 0.01 and tilt < max_tilt:
            return q, err, tilt, True
        if best is None or err < best[1]:
            best = (q, err, tilt, False)
    return best


def pick_roll(q):
    """Wrist roll that makes the jaws open front-to-back (along x), so they
    never reach into the gaps between neighbouring blocks."""
    dd = mujoco.MjData(m)
    dd.qpos[:] = d.qpos
    for n, v in q.items():
        dd.qpos[qadr[n]] = v
    best = None
    for r in np.linspace(*rng_["wrist_roll"], 73):
        dd.qpos[qadr["wrist_roll"]] = r
        mujoco.mj_kinematics(m, dd)
        _, U, _, _ = tips(dd)
        ang = np.arctan2(U[1], U[0])
        ang = (ang + np.pi / 2) % np.pi - np.pi / 2
        cost = abs(ang) + 0.02 * abs(r)
        if best is None or cost < best[0]:
            best = (cost, float(r))
    return best[1]


def move_tip(target, T=1.2, max_tilt=MAX_TILT):
    target = np.asarray(target, float)
    q, err, tilt, ok = solve(target, 0.0, max_tilt)
    if ok:
        roll = pick_roll(q)
        q, err, tilt, ok = solve(target, roll, max_tilt)
    if not ok:
        print(f"   CANNOT REACH {np.round(target, 3)}: {err*100:.1f} cm off, tilt {tilt:.0f} deg")
        return False
    q["wrist_roll"] = float(np.clip(roll, *rng_["wrist_roll"]))
    goto(q, T)
    settle(0.2)
    return True


def touching():
    hits = set()
    for i in range(d.ncon):
        c = d.contact[i]
        if bgeom in (c.geom1, c.geom2):
            o = c.geom2 if c.geom1 == bgeom else c.geom1
            hits.add(f"{m.geom(o).name or 'geom' + str(o)}[{m.body(int(m.geom_bodyid[o])).name}]")
    return sorted(hits)


# ---------- which gripper value is OPEN? ----------
lo, hi = rng_["gripper"]
gaps = {}
for gv in (lo, hi):
    dd = mujoco.MjData(m)
    dd.qpos[:] = d.qpos
    dd.qpos[qadr["gripper"]] = gv
    mujoco.mj_kinematics(m, dd)
    gaps[gv] = tips(dd)[3]
open_end, closed_end = (hi, lo) if gaps[hi] >= gaps[lo] else (lo, hi)
g_open = closed_end + 0.95 * (open_end - closed_end)
g_closed = closed_end
print(f"gripper value {lo:.2f} -> gap {gaps[lo]*100:.1f} cm,  {hi:.2f} -> gap {gaps[hi]*100:.1f} cm")
print(f"using OPEN={g_open:.2f}, CLOSED={g_closed:.2f}.  Block is {BH*100:.1f} cm tall.")


def try_grasp(n, dz, du):
    """Returns (held, dzh). dzh = tip height above the block centre at the moment of grasping."""
    bxy = d.xpos[bid][:2].copy()
    goto({"gripper": g_open}, 0.8)
    if not move_tip([bxy[0], bxy[1], HOVER_Z], 2.0 if n == 0 else 1.5):
        return False, None
    for _ in range(4):                      # line the jaw gap up with the block
        mid, U, protr, gap = tips(d)
        s = d.site_xpos[SITE].copy()
        e = d.xpos[bid][:2] + du * U[:2] - mid[:2]
        if np.linalg.norm(e) < 0.002:
            break
        move_tip([s[0] + e[0], s[1] + e[1], s[2]], 0.8)
    mid, U, protr, gap = tips(d)
    err = np.linalg.norm(d.xpos[bid][:2] + du * U[:2] - mid[:2])
    print(f"   above block: gap centre {err*100:.1f} cm off, open gap {gap*100:.1f} cm")

    z_grasp = max(0.008, max(BH / 2, protr + 0.003) + dz)
    s = d.site_xpos[SITE].copy()
    print(f"   going down to tip height {z_grasp:.3f} (jaws stick out {protr*100:.1f} cm past the tip point)")
    if not move_tip([s[0], s[1], z_grasp], 1.5):
        return False, None
    settle(0.3)

    goto({"gripper": g_closed}, 0.8)
    settle(0.6)
    gj = d.qpos[qadr["gripper"]]
    dzh = float(d.site_xpos[SITE][2] - d.xpos[bid][2])
    print(f"   closed: gripper joint at {gj:.3f} (fully closed = {g_closed:.3f})")
    print(f"   touching block: {touching()}")

    s = d.site_xpos[SITE].copy()
    move_tip([s[0], s[1], HOVER_Z], 1.5)
    settle(0.8)
    b = d.xpos[bid]
    mid, _, _, _ = tips(d)
    held = bool(b[2] > BH / 2 + 0.03 and np.linalg.norm(b[:2] - mid[:2]) < 0.04)
    print(f"   after lift: block z = {b[2]:.3f}  ->  {'HELD' if held else 'MISSED'}")
    return held, dzh


# ================= 1) simulate silently =================
print("\nSimulating... (this takes a little while, no window yet)")
settle(1.0)
held, dzh = False, BH / 2
for n, (dz, du) in enumerate(ATTEMPTS):
    print(f"\nGRASP ATTEMPT {n + 1}/{len(ATTEMPTS)}  (height change {dz*1000:+.0f} mm, sideways {du*1000:+.0f} mm)")
    ok, dz_h = try_grasp(n, dz, du)
    if dz_h is not None:
        dzh = dz_h
    if ok:
        held = True
        break
    print("   no block in the gripper, trying again")

print("\n" + ("PLACING the block on the white square" if held else
              "All grasps failed. Going to the white square anyway to attempt the placement."))
mid, _, _, _ = tips(d)
s = d.site_xpos[SITE].copy()
off = (mid - s)[:2]                           # where the jaw gap sits relative to the tip point
z_over = sq_top + BH / 2 + 0.004 + dzh
tip_xy = sq_c[:2] - off
if move_tip([tip_xy[0], tip_xy[1], z_over + 0.05], 1.8, PLACE_TILT):
    if held:                                  # nudge until the BLOCK is centred over the square
        for _ in range(3):
            e = sq_c[:2] - d.xpos[bid][:2]
            if np.linalg.norm(e) < 0.003:
                break
            tip_xy = tip_xy + e
            move_tip([tip_xy[0], tip_xy[1], z_over + 0.05], 0.8, PLACE_TILT)
    move_tip([tip_xy[0], tip_xy[1], z_over], 1.0, PLACE_TILT)
    goto({"gripper": g_open}, 0.8)
    settle(0.5)
    move_tip([tip_xy[0], tip_xy[1], z_over + 0.06], 1.0, PLACE_TILT)
else:
    print("   could not reach the white square")

settle(1.0)
b = d.xpos[bid]
on = abs(b[0] - sq_c[0]) < sq_half and abs(b[1] - sq_c[1]) < sq_half
print(f"\nFINAL: block bk{BLOCK} is at {np.round(b, 3)}  ->  "
      f"{'ON THE WHITE SQUARE' if on else 'NOT on the white square'}")

arm_cols = [qadr[n] for n in JOINTS[:5]]
travel = float(np.abs(np.diff(np.array([f[arm_cols] for f in frames]), axis=0)).sum())
print(f"\nSUMMARY: recorded {len(frames)} frames ({len(frames) / 30:.0f} s of motion). "
      f"The arm joints moved {travel:.1f} rad in total.")
if travel < 1.0:
    print("WARNING: the arm barely moved. Look for 'CANNOT REACH' lines above and send them to me.")

# ================= 2) replay in the window =================
print("\nOpening the window and replaying. Close the window to exit.")
viewer = mujoco.viewer.launch_passive(m, d)
viewer.cam.lookat[:] = [0.2, 0.08, 0.03]
viewer.cam.distance = 0.85
viewer.cam.azimuth = 150
viewer.cam.elevation = -35
while viewer.is_running():
    for fq in frames:
        if not viewer.is_running():
            break
        with viewer.lock():
            d.qpos[:] = fq
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
        viewer.sync()
        time.sleep(1 / 30 / SPEED)
    time.sleep(1.5)