import numpy as np
import mujoco
from env import JOINTS, CONTROL_HZ, PAD_HALF
from vision import TopCamera, detect

# (hover height, allowed lean in degrees): tried in order when getting above a block
HOVERS = ((0.07, 40), (0.055, 55), (0.045, 70))
PLACE_TILTS = (60, 75, 90)   # leans tried, in order, when carrying to the tower
HELD_MIN = 0.12      # gripper must stay at least this many rad open to count as holding a block
STACK_TOL = 0.03     # a block this close to the square centre counts as "on the tower"
MAX_FAILS = 8
# per attempt: (change in grasp height, sideways shift of the jaws), metres
ATTEMPTS = [(0.0, 0.0), (-0.005, 0.0), (-0.005, 0.006),
            (-0.005, -0.006), (0.005, 0.0), (-0.008, 0.0)]


def skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def wrap90(x):
    return (x + np.pi / 4) % (np.pi / 2) - np.pi / 4


class StackEngine:
    """Uses ONLY camera images and joint angles (plus the robot's own kinematics)."""

    def __init__(self, env):
        self.env, self.m = env, env.model
        m = self.m
        self.kd = mujoco.MjData(m)                       # scratch copy for kinematics
        self.cam = TopCamera(m)
        self.bw, self.bh = self.cam.bw, self.cam.bh
        self.qadr = {n: int(m.jnt_qposadr[m.joint(n).id]) for n in JOINTS}
        self.dadr = {n: int(m.jnt_dofadr[m.joint(n).id]) for n in JOINTS}
        self.rng = {n: m.jnt_range[m.joint(n).id].copy() for n in JOINTS}
        self.site = m.site("gripperframe").id
        self.IK = JOINTS[:4]
        self.q = env.get_obs()["qpos"].copy()            # current joint command
        self.home = self.q.copy()
        self.frames = []
        self.sq_xy = None
        self.stacked = 0                                 # blocks the engine has put on the tower
        self.sq_top = float(2 * m.geom_size[env.sq][2])
        try:
            self.pad = m.geom("static_finger_pad").id
        except KeyError:
            self.pad = None

        kd = self.kd
        kd.qpos[:] = env.data.qpos
        mujoco.mj_forward(m, kd)
        pan = m.joint("shoulder_pan").id
        self.base = kd.xanchor[pan].copy()
        self.pan_sign = 1.0 if kd.xaxis[pan][2] > 0 else -1.0
        wr = m.joint("wrist_roll").id
        ax = kd.xaxis[wr].copy()
        if np.dot(ax, kd.site_xpos[self.site] - kd.xanchor[wr]) < 0:
            ax = -ax
        self.appr = kd.site_xmat[self.site].reshape(3, 3).T @ ax
        self.appr /= np.linalg.norm(self.appr)
        self.fixed_g, self.moving_g = [], []
        for g in range(m.ngeom):
            if m.geom(g).name.endswith("_finger_pad"):
                continue
            b = m.body(int(m.geom_bodyid[g])).name.lower()
            if "jaw" in b:
                self.moving_g.append(g)
            elif "gripper" in b:
                self.fixed_g.append(g)

        lo, hi = self.rng["gripper"]
        gaps = {}
        for gv in (lo, hi):
            self._arm({"gripper": gv})
            gaps[gv] = self._tips()[3]
        open_end, self.g_closed = (hi, lo) if gaps[hi] >= gaps[lo] else (lo, hi)
        self.g_open = self.g_closed + 0.95 * (open_end - self.g_closed)
        self._arm({"gripper": self.g_open})
        self.protr = self._tips()[2]
        print(f"engine ready. gripper open={self.g_open:.2f} closed={self.g_closed:.2f}, "
              f"jaws stick out {self.protr*100:.1f} cm, block {self.bw*100:.1f} cm")

    # ---------- kinematics ----------
    def _arm(self, q):
        for i, n in enumerate(JOINTS):
            self.kd.qpos[self.qadr[n]] = q.get(n, self.q[i])
        mujoco.mj_kinematics(self.m, self.kd)

    def _verts(self, geoms):
        m, kd = self.m, self.kd
        out = []
        for g in geoms:
            xp, xm = kd.geom_xpos[g], kd.geom_xmat[g].reshape(3, 3)
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

    def _tips(self):
        """(middle of jaw gap, horizontal direction fixed->moving jaw, jaw protrusion, gap)"""
        kd = self.kd
        A = kd.site_xmat[self.site].reshape(3, 3) @ self.appr
        Pf, Pm = self._verts(self.fixed_g), self._verts(self.moving_g)
        tf = Pf[Pf @ A > (Pf @ A).max() - 0.015]
        tm = Pm[Pm @ A > (Pm @ A).max() - 0.015]
        mid = 0.5 * (tf.mean(0) + tm.mean(0))
        dv = tm.mean(0) - tf.mean(0)
        U = dv.copy()
        U[2] = 0
        U /= np.linalg.norm(U) + 1e-9
        protr = max((Pf @ A).max(), (Pm @ A).max()) - kd.site_xpos[self.site] @ A
        return mid, U, float(protr), float(np.linalg.norm(dv))

    def _hold_centre(self, mid, U):
        """Where a clamped block's centre sits: against the fixed pad, not in the middle of the gap."""
        if self.pad is None:
            return mid
        sp = self.kd.geom_xpos[self.pad]
        return mid + U * ((sp @ U + PAD_HALF + self.bw / 2) - mid @ U)

    def _ik(self, target, wo, rand, roll):
        m, kd = self.m, self.kd
        self._arm({"wrist_roll": roll})
        rel = target[:2] - self.base[:2]
        pan = self.pan_sign * np.arctan2(rel[1], rel[0])
        kd.qpos[self.qadr["shoulder_pan"]] = np.clip(pan, *self.rng["shoulder_pan"])
        if rand is not None:
            for n in self.IK[1:]:
                kd.qpos[self.qadr[n]] = rand.uniform(*self.rng[n])
        dofs = [self.dadr[n] for n in self.IK]
        down = np.array([0.0, 0.0, -1.0])
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(300):
            mujoco.mj_forward(m, kd)
            a = kd.site_xmat[self.site].reshape(3, 3) @ self.appr
            ep = target - kd.site_xpos[self.site]
            eo = down - a
            if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(eo) < 0.05:
                break
            mujoco.mj_jacSite(m, kd, jp, jr, self.site)
            J = np.vstack([jp[:, dofs], wo * (-skew(a) @ jr[:, dofs])])
            e = np.hstack([ep, wo * eo])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            nrm = np.linalg.norm(dq)
            if nrm > 0.15:
                dq *= 0.15 / nrm
            for n, x in zip(self.IK, dq):
                kd.qpos[self.qadr[n]] = np.clip(kd.qpos[self.qadr[n]] + x, *self.rng[n])
        mujoco.mj_forward(m, kd)
        a = kd.site_xmat[self.site].reshape(3, 3) @ self.appr
        tilt = float(np.degrees(np.arccos(np.clip(-a[2], -1, 1))))
        err = float(np.linalg.norm(target - kd.site_xpos[self.site]))
        return {n: float(kd.qpos[self.qadr[n]]) for n in self.IK}, err, tilt

    def _solve(self, target, roll, max_tilt):
        rand = np.random.default_rng(1)
        tries = [(0.4, None), (0.8, None), (0.15, None)]
        tries += [(w, rand) for w in (0.8, 0.4) for _ in range(15)]
        best = None
        for wo, r in tries:
            q, err, tilt = self._ik(target, wo, r, roll)
            if err < 0.01 and tilt < max_tilt:
                return q, err, tilt, True
            if best is None or err < best[1]:
                best = (q, err, tilt, False)
        return best

    def _pick_roll(self, q, yaw):
        """Wrist roll that makes the jaws close across the block's faces."""
        best = None
        for r in np.linspace(*self.rng["wrist_roll"], 73):
            self._arm({**q, "wrist_roll": r})
            U = self._tips()[1]
            cost = abs(wrap90(np.arctan2(U[1], U[0]) - yaw)) + 0.02 * abs(r)
            if best is None or cost < best[0]:
                best = (cost, float(r))
        return best[1]

    # ---------- motion ----------
    def goto(self, q, T=1.0):
        start, tgt = self.q.copy(), self.q.copy()
        for k, v in q.items():
            tgt[JOINTS.index(k)] = v
        n = max(1, int(T * CONTROL_HZ))
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)
            self.q = start + s * (tgt - start)
            self.env.step(self.q, render=False)
            self.frames.append(self.env.data.qpos.copy())

    def settle(self, sec):
        for _ in range(max(1, int(sec * CONTROL_HZ))):
            self.env.step(self.q, render=False)
            self.frames.append(self.env.data.qpos.copy())

    def move_gap(self, xy, z, yaw, T, tilt, du=0.0, hold=False, quiet=False):
        """Put the jaw gap (or a held block) centre over xy with the tip point at height z."""
        xy = np.asarray(xy, float)
        roll, tgt = 0.0, xy.copy()
        for _ in range(3):
            q, err, tl, ok = self._solve(np.array([tgt[0], tgt[1], z]), roll, tilt)
            if not ok:
                break
            roll = self._pick_roll(q, yaw)
            self._arm({**q, "wrist_roll": roll})
            mid, U, _, _ = self._tips()
            c = self._hold_centre(mid, U) if hold else mid
            off = c - self.kd.site_xpos[self.site]
            tgt = xy + du * U[:2] - off[:2]
        q, err, tl, ok = self._solve(np.array([tgt[0], tgt[1], z]), roll, tilt)
        if not ok:
            if not quiet:
                print(f"   cannot reach {np.round(xy, 3)} at z={z:.3f} (lean limit {tilt}): "
                      f"{err*100:.1f} cm off, lean {tl:.0f} deg")
            return False
        q["wrist_roll"] = float(np.clip(roll, *self.rng["wrist_roll"]))
        self.goto(q, T)
        self.settle(0.15)
        return True

    def move_gap_any(self, xy, z, yaw, T, tilts, hold):
        for t in tilts:
            if self.move_gap(xy, z, yaw, T, t, 0.0, hold, quiet=(t != tilts[-1])):
                return True
        return False

    def opening(self):
        return abs(float(self.env.data.qpos[self.qadr["gripper"]]) - self.g_closed)

    # ---------- perception ----------
    def park(self):
        q = dict(zip(JOINTS, self.home))
        q["shoulder_pan"] = self.pan_sign * -1.5          # swing out of the camera's way
        self.goto(q, 1.5)

    def look(self):
        self.park()
        self.settle(0.5)
        img = self.env.get_image("topdown")
        scene = detect(img, self.cam)
        from PIL import Image
        Image.fromarray(img).save("camera_last.png")
        print("   camera sees:", [(b["color"], tuple(np.round(b["xy"], 3))) for b in scene["blocks"]])
        return scene

    def on_tower(self, b):
        """Blocks higher up look shifted in the top-down image, so allow a bit more room."""
        h = self.stacked * self.bh
        lateral = np.linalg.norm(self.sq_xy - self.cam.pos[:2])
        allowed = STACK_TOL + lateral * h / max(self.cam.pos[2] - h, 0.3)
        return np.linalg.norm(b["xy"] - self.sq_xy) < allowed

    def loose(self, scene):
        return [b for b in scene["blocks"] if not self.on_tower(b)]

    # ---------- can the arm reach every tower level? ----------
    def tower_z(self, level, dzh):
        return self.sq_top + self.bh * (level + 0.5) + 0.002 + dzh

    def check_tower(self, n):
        dzh = max(0.008, max(self.bh / 2, self.protr + 0.003)) - self.bh / 2
        print("checking that the arm can reach each tower level:")
        bad = []
        for lv in range(n):
            z = self.tower_z(lv, dzh)
            ok = all(any(self._solve(np.array([*self.sq_xy, zz]), 0.0, t)[3] for t in PLACE_TILTS)
                     for zz in (z, z + 0.06))
            print(f"   level {lv + 1}: {'reachable' if ok else 'NOT REACHABLE'}")
            if not ok:
                bad.append(lv + 1)
        if bad:
            print("WARNING: the white square is too far or awkward for those levels.\n"
                  "         Run  python reach_check.py  to find a better spot for it.")
        return bad

    # ---------- actions ----------
    def grasp(self, xy, yaw, dz, du):
        """Returns (held, dzh, reached). reached=False means the arm could not get over the block."""
        self.goto({"gripper": self.g_open}, 0.6)
        tl = hov = None
        for hz, t in HOVERS:
            if self.move_gap(xy, hz, yaw, 2.0, t, du, quiet=True):
                tl, hov = t, hz
                break
        if tl is None:
            print(f"   the arm cannot get over the block at {np.round(xy, 3)} (tried hover heights "
                  f"{[h for h, _ in HOVERS]})")
            return False, 0.0, False
        z = max(0.008, max(self.bh / 2, self.protr + 0.003) + dz)
        dzh = z - self.bh / 2                              # tip height above the block centre
        print(f"   above the block (hover {hov:.3f}), descending to tip height {z:.3f}")
        if not self.move_gap(xy, z, yaw, 1.5, tl, du):
            return False, dzh, False
        self.settle(0.3)
        self.goto({"gripper": self.g_closed}, 0.8)
        self.settle(0.6)
        print(f"   gripper closed to {self.opening():.2f} rad open (needs > {HELD_MIN} to be holding a block)")
        if self.opening() <= HELD_MIN:
            self.goto({"gripper": self.g_open}, 0.6)
            return False, dzh, True
        self.move_gap(xy, hov, yaw, 1.5, tl, du)
        self.settle(0.5)
        held = self.opening() > HELD_MIN
        print(f"   after lifting: {'HOLDING the block' if held else 'block slipped out'} ({self.opening():.2f} rad)")
        if not held:
            self.goto({"gripper": self.g_open}, 0.6)
        return held, dzh, True

    def putback(self, xy, yaw, dzh, du):
        print("   cannot place the block, putting it back down")
        z = self.bh / 2 + dzh + 0.004
        self.move_gap(xy, HOVERS[0][0], yaw, 2.0, HOVERS[-1][1], du, True)
        self.move_gap(xy, z, yaw, 1.2, HOVERS[-1][1], du, True)
        self.goto({"gripper": self.g_open}, 0.8)
        self.settle(0.4)
        self.move_gap(xy, HOVERS[0][0], yaw, 1.0, HOVERS[-1][1], du, True)

    def place(self, level, dzh):
        """Carry to the tower and put the block on top. Never lets go unless it is over the tower."""
        z = self.tower_z(level, dzh)
        print(f"   carrying to tower level {level + 1}")
        if not self.move_gap_any(self.sq_xy, z + 0.06, 0.0, 2.5, PLACE_TILTS, True):
            return False
        if self.opening() <= HELD_MIN:
            print("   the block was dropped during the carry")
            return False
        if not self.move_gap_any(self.sq_xy, z, 0.0, 1.2, PLACE_TILTS, True):
            return False
        self.goto({"gripper": self.g_open}, 0.8)
        self.settle(0.5)
        self.move_gap(self.sq_xy, z + 0.06, 0.0, 1.0, PLACE_TILTS[0], 0.0, True, quiet=True)
        return True

    # ---------- the whole task ----------
    def run(self):
        scene = self.look()
        if scene["square"] is None:
            raise SystemExit("The camera could not find the white square.")
        self.sq_xy = scene["square"]["xy"]
        total = len(scene["blocks"])
        print(f"camera sees {total} blocks; white square at {np.round(self.sq_xy, 3)}")
        self.check_tower(total)
        fails = 0
        unreachable = []                     # blocks the arm could not get over (positions)
        while True:
            loose = self.loose(scene)
            print(f"\ntower height {self.stacked}, {len(loose)} blocks still loose")
            if not loose:
                print("ALL DETECTED BLOCKS ARE ON THE TOWER")
                break
            if fails >= MAX_FAILS:
                print(f"Stopping: {fails} failed grasps in a row.")
                break
            cand = [b for b in loose if not any(np.linalg.norm(b["xy"] - p) < 0.02 for p in unreachable)]
            if not cand:
                print("Stopping: the arm cannot reach any of the remaining blocks:",
                      [(b["color"], tuple(np.round(b["xy"], 3))) for b in loose])
                break
            b = min(cand, key=lambda b: np.linalg.norm(b["xy"] - self.sq_xy))
            dz, du = ATTEMPTS[fails % len(ATTEMPTS)]
            print(f"picking the {b['color']} block at {np.round(b['xy'], 3)} "
                  f"(variant {fails % len(ATTEMPTS) + 1}: {dz*1000:+.0f} mm height, {du*1000:+.0f} mm sideways)")
            held, dzh, reached = self.grasp(b["xy"], b["yaw"], dz, du)
            if not reached:
                unreachable.append(b["xy"].copy())
                print("   skipping this block, trying a different one")
                continue
            if held and not self.place(self.stacked, dzh):
                self.putback(b["xy"], b["yaw"], dzh, du)
            scene = self.look()
            if len(self.loose(scene)) < len(loose):
                self.stacked += 1
                fails = 0
                unreachable.clear()
                print(f"   STACKED. tower is now {self.stacked} high")
            else:
                fails += 1
                print("   the block is not on the tower, will retry")