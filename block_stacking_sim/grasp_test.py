import numpy as np
import mujoco
from PIL import Image
from env import RowEnv, JOINTS

BLOCK = 1          # which block to try (0 to 3)
HOVER_Z = 0.09     # fingertip hover height (m)
MAX_TILT = 40      # how far the gripper may lean from straight down (degrees)

env = RowEnv()
m, d, rend = env.model, env.data, env.renderer
SITE = m.site("gripperframe").id
BH = 2 * m.geom("bk0_geom").size[2] if False else 2 * m.geom_size[m.geom("bk0_geom").id][2]
qadr = {n: int(m.jnt_qposadr[m.joint(n).id]) for n in JOINTS}
dadr = {n: int(m.jnt_dofadr[m.joint(n).id]) for n in JOINTS}
rng_ = {n: m.jnt_range[m.joint(n).id] for n in JOINTS}
act = dict(zip(JOINTS, env.act))
IK_J = JOINTS[:4]
bid = env.blocks[BLOCK]
bgeom = m.geom(f"bk{BLOCK}_geom").id

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

frames = []
cap_every = max(1, int(1 / (20 * m.opt.timestep)))
step_count = 0


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
    """Returns: middle of the jaw gap, direction the jaws open in (horizontal),
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


def capture():
    rend.update_scene(d, camera="overview")
    frames.append(rend.render()[::2, ::2].copy())


def step_sim():
    global step_count
    mujoco.mj_step(m, d)
    step_count += 1
    if step_count % cap_every == 0:
        capture()


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


def solve(target, roll):
    rand = np.random.default_rng(1)
    tries = [(0.4, None), (0.8, None), (0.15, None)]
    tries += [(w, rand) for w in (0.8, 0.4) for _ in range(15)]
    best = None
    for wo, r in tries:
        q, err, tilt = ik(target, wo, r, roll)
        if err < 0.01 and tilt < MAX_TILT:
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


def move_tip(target, T=1.2):
    target = np.asarray(target, float)
    q, err, tilt, ok = solve(target, 0.0)
    if ok:
        roll = pick_roll(q)
        q, err, tilt, ok = solve(target, roll)
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
            other = c.geom2 if c.geom1 == bgeom else c.geom1
            hits.add(m.geom(other).name or f"geom{other}")
    return sorted(hits)


def save_gif():
    if frames:
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save("grasp_test.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)
        print("saved grasp_test.gif")


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

try:
    bxy = d.xpos[bid][:2].copy()
    goto({"gripper": g_open}, 0.8)
    print(f"\n1) moving above block bk{BLOCK} at {np.round(bxy, 3)}")
    if not move_tip([bxy[0], bxy[1], HOVER_Z], 2.0):
        raise SystemExit

    for _ in range(4):                      # line the gap up with the block
        mid, U, protr, gap = tips(d)
        s = d.site_xpos[SITE].copy()
        e = d.xpos[bid][:2] - mid[:2]
        if np.linalg.norm(e) < 0.002:
            break
        move_tip([s[0] + e[0], s[1] + e[1], s[2]], 0.8)
    mid, U, protr, gap = tips(d)
    err = np.linalg.norm(d.xpos[bid][:2] - mid[:2])
    print(f"   gap centre is {err*100:.1f} cm off the block, open gap {gap*100:.1f} cm, "
          f"jaws open along ({U[0]:.2f}, {U[1]:.2f})")

    z_grasp = max(BH / 2, protr + 0.003)
    s = d.site_xpos[SITE].copy()
    print(f"2) going down: jaws stick out {protr*100:.1f} cm past the tip point, tip point to z={z_grasp:.3f}")
    if not move_tip([s[0], s[1], z_grasp], 1.5):
        raise SystemExit
    settle(0.3)

    print("3) closing the gripper")
    goto({"gripper": g_closed}, 0.8)
    settle(0.6)
    print(f"   gripper joint stopped at {d.qpos[qadr['gripper']]:.3f} (fully closed would be {g_closed:.3f})")
    print(f"   touching block: {touching()}")

    print("4) lifting")
    s = d.site_xpos[SITE].copy()
    move_tip([s[0], s[1], HOVER_Z], 1.5)
    settle(1.0)
    b = d.xpos[bid]
    mid, _, _, _ = tips(d)
    
    held = b[2] > BH / 2 + 0.03 and np.linalg.norm(b[:2] - mid[:2]) < 0.04
    print(f"   block z = {b[2]:.3f}   touching: {touching()}")
    print("\nRESULT:", "HELD - the block was lifted!" if held else "MISSED - the block stayed on the table.")
finally:
    save_gif()