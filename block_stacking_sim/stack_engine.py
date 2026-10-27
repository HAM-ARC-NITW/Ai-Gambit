import os, time, argparse
import numpy as np
import mujoco
from perception import DepthPerception, BH, BW, BD

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "SO-ARM100", "Simulation", "SO101", "stack_scene.xml")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
IK_JOINTS = JOINTS[:4]
N_BLOCKS = 4
HOVER = 0.06        # hover height above grasp/place point (m)
MAX_TILT = 35       # max degrees the gripper may lean away from straight down
TARGET_XY = None    # set e.g. [0.20, -0.05] to override the white-box auto-detect
MAX_TRIES = 4       # grasp attempts per block before giving up on it


def skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def wrap90(x):
    return (x + np.pi / 4) % (np.pi / 2) - np.pi / 4


class Sim:
    def __init__(self, headless, video):
        self.m = mujoco.MjModel.from_xml_path(SCENE)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.viewer = None
        if not headless:
            from mujoco import viewer as mjviewer
            self.viewer = mjviewer.launch_passive(self.m, self.d)
        self.every = max(1, round(1 / (30 * self.m.opt.timestep)))
        self.k, self.frames, self.rend, self.vcam = 0, [], None, None
        if video:
            self.rend = mujoco.Renderer(self.m, 480, 640)
            self.vcam = mujoco.MjvCamera()
            self.vcam.azimuth, self.vcam.elevation, self.vcam.distance = 140, -35, 1.0

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
            self.k += 1
            if self.k % self.every == 0:
                if self.viewer is not None:
                    if not self.viewer.is_running():
                        raise SystemExit
                    self.viewer.sync()
                    time.sleep(self.m.opt.timestep * self.every * 0.9)
                if self.rend is not None:
                    self.rend.update_scene(self.d, camera=self.vcam)
                    self.frames.append(self.rend.render().copy())

    def seconds(self, s):
        self.step(int(s / self.m.opt.timestep))


class Arm:
    def __init__(self, sim):
        self.s = sim
        self.picked = False          # True once a grasp has been verified (block lifted and held)
        m, d = sim.m, sim.d
        self.site = m.site("gripperframe").id
        self.jid = {n: m.joint(n).id for n in JOINTS}
        self.qa = {n: int(m.jnt_qposadr[self.jid[n]]) for n in JOINTS}
        self.da = {n: int(m.jnt_dofadr[self.jid[n]]) for n in JOINTS}
        self.rg = {n: m.jnt_range[self.jid[n]].copy() for n in JOINTS}
        self.ac = {}
        for a in range(m.nu):
            j = int(m.actuator_trnid[a, 0])
            for n in JOINTS:
                if self.jid[n] == j:
                    self.ac[n] = a
        missing = [n for n in JOINTS if n not in self.ac]
        if missing:
            raise SystemExit(f"No actuator found for {missing}. Run inspect_model.py and send me the actuator list.")
        mujoco.mj_forward(m, d)
        wr = self.jid["wrist_roll"]
        ax = d.xaxis[wr].copy()
        if np.dot(ax, d.site_xpos[self.site] - d.xanchor[wr]) < 0:
            ax = -ax
        self.appr = d.site_xmat[self.site].reshape(3, 3).T @ ax   # approach axis in gripper frame
        self.appr = self.appr / np.linalg.norm(self.appr)
        self.home = {n: float(d.qpos[self.qa[n]]) for n in JOINTS}
        self.cmd = dict(self.home)
        for n in JOINTS:
            d.ctrl[self.ac[n]] = self.cmd[n]
        self._survey()
        self._calibrate()

    # ---------- where can the arm reach ----------
    def _survey(self):
        m = self.s.m
        d = mujoco.MjData(m)
        d.qpos[:] = self.s.d.qpos
        pan = self.jid["shoulder_pan"]
        self.base = self.s.d.xanchor[pan].copy()
        self.pan_sign = 1.0 if self.s.d.xaxis[pan][2] > 0 else -1.0
        d.qpos[self.qa["shoulder_pan"]] = 0.0
        d.qpos[self.qa["wrist_roll"]] = 0.0
        grid = {n: np.linspace(*self.rg[n], 14) for n in IK_JOINTS[1:]}
        pts = []
        for tol, az in [(0.02, -0.8), (0.04, -0.5)]:
            for a in grid["shoulder_lift"]:
                for b in grid["elbow_flex"]:
                    for c in grid["wrist_flex"]:
                        d.qpos[self.qa["shoulder_lift"]] = a
                        d.qpos[self.qa["elbow_flex"]] = b
                        d.qpos[self.qa["wrist_flex"]] = c
                        mujoco.mj_kinematics(m, d)
                        p = d.site_xpos[self.site]
                        ap = d.site_xmat[self.site].reshape(3, 3) @ self.appr
                        if abs(p[2] - 0.02) < tol and ap[2] < az:
                            pts.append(p[:2] - self.base[:2])
            if pts:
                break
        if not pts:
            raise SystemExit("Survey found no way to touch the table. Run workspace.py and send me its output.")
        pts = np.array(pts)
        r = np.hypot(pts[:, 0], pts[:, 1])
        u = (pts / r[:, None]).mean(0)
        self.phi0 = float(np.arctan2(u[1], u[0]))
        self.rmin, self.rmax = float(np.percentile(r, 5)), float(np.percentile(r, 95))
        print(f"arm reach on table: {self.rmin:.3f} - {self.rmax:.3f} m, forward angle {np.degrees(self.phi0):.0f} deg")

    def zone_centre(self):
        r = self.rmin + 0.5 * (self.rmax - self.rmin)
        return self.base[:2] + r * np.array([np.cos(self.phi0), np.sin(self.phi0)])

    # ---------- measure the gripper jaws ----------
    def _split_geoms(self):
        m = self.s.m
        bm = int(m.jnt_bodyid[self.jid["gripper"]])

        def under(b):
            while True:
                if b == bm:
                    return True
                if b == 0:
                    return False
                b = int(m.body_parentid[b])

        sb = int(m.site_bodyid[self.site])
        if under(sb):
            sb = int(m.body_parentid[bm])
        moving = [g for g in range(m.ngeom) if under(int(m.geom_bodyid[g]))]
        static = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == sb and not under(sb)]
        return static, moving

    def _verts(self, geoms, d):
        m = self.s.m
        out = []
        for g in geoms:
            xp, xm = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
            if m.geom_type[g] == int(mujoco.mjtGeom.mjGEOM_MESH):
                did = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[did]: m.mesh_vertadr[did] + m.mesh_vertnum[did]]
                out.append(xp + v @ xm.T)
            else:
                sz = m.geom_size[g].copy()
                sz[sz == 0] = sz[0]
                c = np.array([[a, b, c_] for a in (-1, 1) for b in (-1, 1) for c_ in (-1, 1)]) * sz
                out.append(xp + c @ xm.T)
        return np.vstack(out) if out else np.zeros((0, 3))

    def _jaw_profile(self, gval):
        m = self.s.m
        d = mujoco.MjData(m)
        d.qpos[:] = self.s.d.qpos
        d.qpos[self.qa["gripper"]] = gval
        mujoco.mj_kinematics(m, d)
        R = d.site_xmat[self.site].reshape(3, 3)
        s = d.site_xpos[self.site]
        A = self.appr
        gs, gm = self._split_geoms()
        Ps = (self._verts(gs, d) - s) @ R
        Pm = (self._verts(gm, d) - s) @ R
        if len(Ps) < 8 or len(Pm) < 8:
            return None
        ts = Ps[Ps @ A > (Ps @ A).max() - 0.03]
        tm = Pm[Pm @ A > (Pm @ A).max() - 0.03]
        dv = tm.mean(0) - ts.mean(0)
        U = dv - (dv @ A) * A
        if np.linalg.norm(U) < 1e-4:
            return None
        U = U / np.linalg.norm(U)
        V = np.cross(A, U)
        us = float(np.percentile(ts @ U, 95))
        um = float(np.percentile(tm @ U, 5))
        cv = float(np.clip(0.5 * (np.mean(ts @ V) + np.mean(tm @ V)), -0.01, 0.01))
        protr = float(max((Ps @ A).max(), (Pm @ A).max()))
        return dict(U=U, V=V, us=us, um=um, gap=um - us, cv=cv, protr=protr)

    def _calibrate(self):
        m = self.s.m
        lo, hi = self.rg["gripper"]
        pa, pb = self._jaw_profile(lo), self._jaw_profile(hi)
        gs, gm = self._split_geoms()
        print(f"gripper geoms: {len(gs)} fixed (body {m.body(int(m.geom_bodyid[gs[0]])).name if gs else '-'}), "
              f"{len(gm)} moving")
        if pa is not None and pb is not None:
            if pb["gap"] >= pa["gap"]:
                self.g_closed, open_end = lo, hi
            else:
                self.g_closed, open_end = hi, lo
            self.g_open = self.g_closed + 0.95 * (open_end - self.g_closed)
            self.prof = self._jaw_profile(self.g_open)
        if pa is None or pb is None or self.prof is None or self.prof["gap"] < 0.01:
            print("   WARNING: could not measure the jaws, using safe guesses.")
            bm = int(m.jnt_bodyid[self.jid["gripper"]])
            best = None
            for gv in (lo, hi):
                d = mujoco.MjData(m)
                d.qpos[:] = self.s.d.qpos
                d.qpos[self.qa["gripper"]] = gv
                mujoco.mj_kinematics(m, d)
                R = d.site_xmat[self.site].reshape(3, 3)
                v = R.T @ (d.xpos[bm] - d.site_xpos[self.site])
                v = v - (v @ self.appr) * self.appr
                if best is None or np.linalg.norm(v) < best[0]:
                    best = (np.linalg.norm(v), gv)
                if gv == hi:
                    Ud = v
            self.g_closed = best[1]
            open_end = hi if self.g_closed == lo else lo
            self.g_open = self.g_closed + 0.95 * (open_end - self.g_closed)
            d = mujoco.MjData(m)
            d.qpos[:] = self.s.d.qpos
            d.qpos[self.qa["gripper"]] = open_end
            mujoco.mj_kinematics(m, d)
            R = d.site_xmat[self.site].reshape(3, 3)
            v = R.T @ (d.xpos[bm] - d.site_xpos[self.site])
            v = v - (v @ self.appr) * self.appr
            U = v / (np.linalg.norm(v) + 1e-9)
            self.prof = dict(U=U, V=np.cross(self.appr, U), us=-0.025, um=0.025, gap=0.05, cv=0.0, protr=0.02)
        p = self.prof
        print(f"gripper: open gap {p['gap']*100:.1f} cm (block is {BW*100:.1f} cm), "
              f"gap centre {0.5*(p['us']+p['um'])*100:.1f} cm off the tip point, "
              f"jaws stick out {p['protr']*100:.1f} cm past the tip point")
        if p["gap"] < BW + 0.004:
            print("   WARNING: block is too wide for the open gripper. Make SIZE_MM smaller in make_cube_stl.py.")

    # ---------- inverse kinematics ----------
    def ik(self, target, wo=0.4, rand=None, roll=0.0):
        m = self.s.m
        d = mujoco.MjData(m)
        d.qpos[:] = self.s.d.qpos
        for n in JOINTS:
            d.qpos[self.qa[n]] = self.cmd[n]
        d.qpos[self.qa["wrist_roll"]] = roll
        rel = target[:2] - self.base[:2]
        pan = self.pan_sign * (np.arctan2(rel[1], rel[0]) - self.phi0)
        pan = (pan + np.pi) % (2 * np.pi) - np.pi
        d.qpos[self.qa["shoulder_pan"]] = np.clip(pan, *self.rg["shoulder_pan"])
        if rand is not None:
            for n in IK_JOINTS[1:]:
                lo, hi = self.rg[n]
                d.qpos[self.qa[n]] = rand.uniform(lo, hi)
        dofs = [self.da[n] for n in IK_JOINTS]
        down = np.array([0.0, 0.0, -1.0])
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(300):
            mujoco.mj_forward(m, d)
            a = d.site_xmat[self.site].reshape(3, 3) @ self.appr
            ep = target - d.site_xpos[self.site]
            eo = down - a
            if np.linalg.norm(ep) < 5e-4 and (wo == 0 or np.linalg.norm(eo) < 0.05):
                break
            mujoco.mj_jacSite(m, d, jp, jr, self.site)
            J = np.vstack([jp[:, dofs], wo * (-skew(a) @ jr[:, dofs])])
            e = np.hstack([ep, wo * eo])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            nrm = np.linalg.norm(dq)
            if nrm > 0.15:
                dq *= 0.15 / nrm
            for n, x in zip(IK_JOINTS, dq):
                d.qpos[self.qa[n]] = np.clip(d.qpos[self.qa[n]] + x, *self.rg[n])
        return {n: float(d.qpos[self.qa[n]]) for n in IK_JOINTS}, float(np.linalg.norm(ep))

    def _pose(self, q, roll):
        m = self.s.m
        d = mujoco.MjData(m)
        d.qpos[:] = self.s.d.qpos
        for n, v in q.items():
            d.qpos[self.qa[n]] = v
        d.qpos[self.qa["wrist_roll"]] = roll
        mujoco.mj_kinematics(m, d)
        return d.site_xmat[self.site].reshape(3, 3).copy(), d.site_xpos[self.site].copy()

    def _tilt(self, q, roll):
        R, _ = self._pose(q, roll)
        a = R @ self.appr
        return float(np.degrees(np.arccos(np.clip(-a[2], -1, 1))))

    def _solve(self, tgt, roll, max_tilt=None):
        max_tilt = MAX_TILT if max_tilt is None else max_tilt
        rand = np.random.default_rng(1)
        tries = [(0.4, None), (0.8, None), (0.15, None)]
        tries += [(wo, rand) for wo in (0.8, 0.4) for _ in range(15)]
        best = None
        for wo, r in tries:
            q, err = self.ik(tgt, wo, r, roll)
            tl = self._tilt(q, roll)
            if err < 0.01 and tl < max_tilt:
                return q, err, tl, True
            if best is None or err < best[1]:
                best = (q, err, tl, False)
        return best

    def _pick_roll(self, q, yaw):
        """Wrist roll that lines the jaw opening up with the block's sides."""
        m = self.s.m
        d = mujoco.MjData(m)
        d.qpos[:] = self.s.d.qpos
        for n, v in q.items():
            d.qpos[self.qa[n]] = v
        lo, hi = self.rg["wrist_roll"]
        best = None
        for r in np.linspace(lo, hi, 73):
            d.qpos[self.qa["wrist_roll"]] = r
            mujoco.mj_kinematics(m, d)
            Uw = d.site_xmat[self.site].reshape(3, 3) @ self.prof["U"]
            psi = np.arctan2(Uw[1], Uw[0])
            cost = abs(wrap90(psi - yaw)) + 0.02 * abs(r)
            if best is None or cost < best[0]:
                best = (cost, float(r))
        return best[1]

    # ---------- motion ----------
    def goto(self, q, T=1.2):
        start = dict(self.cmd)
        n = max(1, int(T / self.s.m.opt.timestep))
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)
            for k, v in q.items():
                self.cmd[k] = start[k] + s * (v - start[k])
                self.s.d.ctrl[self.ac[k]] = self.cmd[k]
            self.s.step()

    def move_to(self, xyz, yaw, T=1.2, hold=False):
        """Move so the block (or gap) centre is at xyz[:2] and the tip point is at height xyz[2]."""
        xyz = np.asarray(xyz, float)
        p = self.prof
        off_u = (p["us"] + BW / 2) if hold else 0.5 * (p["us"] + p["um"])
        roll, tgt = 0.0, xyz.copy()
        for _ in range(3):
            q, err, tl, ok = self._solve(tgt, roll)
            if not ok:
                break
            roll = self._pick_roll(q, yaw)
            R, _ = self._pose(q, roll)
            off = R @ (off_u * p["U"] + p["cv"] * p["V"])
            tgt = xyz - np.array([off[0], off[1], 0.0])
        q, err, tl, ok = self._solve(tgt, roll)
        if not ok:
            print(f"   cannot reach {np.round(xyz, 3)} upright: {err*100:.1f} cm off, tilt {tl:.0f} deg")
            return False
        q["wrist_roll"] = float(np.clip(roll, *self.rg["wrist_roll"]))
        self.goto(q, T)
        return True

    def grip(self, open_, dwell=0.3):
        self.goto({"gripper": self.g_open if open_ else self.g_closed}, 0.6)
        self.s.seconds(dwell)

    def park(self):
        """Swing the arm to the side so it is out of the camera's view of the blocks."""
        q = dict(self.home)
        lo, hi = self.rg["shoulder_pan"]
        q["shoulder_pan"] = float(0.85 * (hi if abs(hi) > abs(lo) else lo))
        self.goto(q, 1.5)

    def move_tip(self, tgt, yaw, T=1.2, max_tilt=None):
        tgt = np.asarray(tgt, float)
        roll = 0.0
        q, err, tl, ok = self._solve(tgt, roll, max_tilt)
        if ok:
            roll = self._pick_roll(q, yaw)
            q, err, tl, ok = self._solve(tgt, roll, max_tilt)
        if not ok:
            print(f"   cannot reach {np.round(tgt, 3)}: {err*100:.1f} cm off, tilt {tl:.0f} deg")
            return False
        q["wrist_roll"] = float(np.clip(roll, *self.rg["wrist_roll"]))
        self.goto(q, T)
        self.s.seconds(0.2)
        return True

    def pick_place(self, pick_xy, pick_yaw, place_xy, place_yaw, level, pad_h=0.0, attempt=0):
        """Pick one block and drop it on the white box.
        Returns True only if the block was picked AND released on the stack.
        self.picked tells you whether the grasp itself was verified."""
        m, d = self.s.m, self.s.d
        p = self.prof
        self.picked = False
        pick_xy, place_xy = np.asarray(pick_xy, float), np.asarray(place_xy, float)
        zc = float(np.clip(max(BH / 2, p["protr"] + 0.004), BH / 2, 1.2 * BH))
        zc = max(zc - 0.003 * attempt, 0.35 * BH)       # each retry grabs a bit lower
        bk = [(i, d.xpos[m.body(f"bk{i}").id].copy()) for i in range(N_BLOCKS)]
        i, _ = min(bk, key=lambda t: np.linalg.norm(t[1][:2] - pick_xy))
        bid = m.body(f"bk{i}").id
        gapc = lambda R: R @ (0.5 * (p["us"] + p["um"]) * p["U"] + p["cv"] * p["V"])

        def putback():
            print("   cannot place it, putting the block back down")
            self.move_to([pick_xy[0], pick_xy[1], zc + 0.01], pick_yaw, 1.5, hold=True)
            self.grip(True)
            self.move_to([pick_xy[0], pick_xy[1], zc + HOVER], pick_yaw, 1.0)
            return False

        def abort_pick(why):
            print(f"   pick failed ({why}), reopening gripper")
            self.picked = False
            self.grip(True)
            s = d.site_xpos[self.site].copy()
            self.move_tip([s[0], s[1], zc + HOVER], pick_yaw, 1.0)
            return False

        # ---- approach and align the jaw gap with the block ----
        self.grip(True)
        if not self.move_to([pick_xy[0], pick_xy[1], zc + HOVER], pick_yaw, 1.5):
            return False
        for _ in range(3):
            self.s.seconds(0.2)
            R = d.site_xmat[self.site].reshape(3, 3)
            s0 = d.site_xpos[self.site].copy()
            e = d.xpos[bid][:2] - (s0 + gapc(R))[:2]
            if np.linalg.norm(e) < 0.002:
                break
            if not self.move_tip([s0[0] + e[0], s0[1] + e[1], zc + HOVER], pick_yaw, 0.8):
                return False
        s0 = d.site_xpos[self.site].copy()
        R = d.site_xmat[self.site].reshape(3, 3)
        e = d.xpos[bid][:2] - (s0 + gapc(R))[:2]
        print(f"   above block bk{i}: off by {np.linalg.norm(e)*100:.1f} cm "
              f"(attempt {attempt + 1}, grasp z={zc:.3f})")

        # ---- descend slowly, settle ----
        if not self.move_tip([s0[0] + e[0], s0[1] + e[1], zc], pick_yaw, 1.5):
            return False
        self.s.seconds(0.3)

        # ---- close, then verify with a small test lift ----
        self.grip(False, dwell=0.5 + 0.3 * attempt)
        s1 = d.site_xpos[self.site].copy()
        if not self.move_tip([s1[0], s1[1], s1[2] + 0.02], pick_yaw, 1.2):
            return abort_pick("cannot lift")
        self.s.seconds(0.4)
        b = d.xpos[bid]
        sx = d.site_xpos[self.site]
        lifted = b[2] > BH / 2 + 0.01
        near = np.linalg.norm(b[:2] - sx[:2]) < 0.04
        if not (lifted and near):
            return abort_pick(f"block z={b[2]:.3f}, lifted={lifted}, near={near}")

        # ---- full lift to hover and re-check ----
        if not self.move_tip([s1[0], s1[1], zc + HOVER], pick_yaw, 1.0):
            return abort_pick("cannot reach hover")
        s2 = d.site_xpos[self.site].copy()
        bpos = d.xpos[bid].copy()
        self.picked = bool(bpos[2] > BH / 2 + 0.02 and np.linalg.norm(bpos[:2] - s2[:2]) < 0.05)
        print(f"   after lift: block z={bpos[2]:.3f} -> {'HELD' if self.picked else 'MISSED'}")
        if not self.picked:
            self.grip(True)
            return False

        # ---- carry it over the white box; correct the aim until the BLOCK is centred ----
        dzh = s2[2] - bpos[2]                       # tip height above block centre while held
        z_tip = pad_h + BH * (level + 0.5) + 0.004 + dzh
        tip = np.array([place_xy[0], place_xy[1], z_tip + HOVER])
        tip[:2] -= (bpos[:2] - s2[:2])
        for it in range(4):
            if not self.move_tip(tip, place_yaw, 1.5 if it == 0 else 0.8, max_tilt=55):
                return putback()
            if d.xpos[bid][2] < BH / 2 + 0.01:      # dropped on the way
                print("   dropped the block during transport")
                self.picked = False
                return False
            e = place_xy - d.xpos[bid][:2]
            if np.linalg.norm(e) < 0.003:
                break
            tip[:2] += e
        print(f"   block is {np.linalg.norm(place_xy - d.xpos[bid][:2])*100:.1f} cm from the target centre")
        if not self.move_tip([tip[0], tip[1], z_tip], place_yaw, 1.0, max_tilt=55):
            return putback()
        self.grip(True)
        self.move_tip([tip[0], tip[1], z_tip + HOVER], place_yaw, 1.0, max_tilt=55)
        return True


def find_target(m, d, arm):
    """Find the white box in the scene. Returns (xy, height_of_its_top)."""
    if TARGET_XY is not None:
        return np.array(TARGET_XY, float), 0.0

    def root(b):
        while b != 0 and m.body_parentid[b] != 0:
            b = int(m.body_parentid[b])
        return b

    arm_root = root(int(m.jnt_bodyid[arm.jid["shoulder_pan"]]))
    found = []
    for g in range(m.ngeom):
        t = int(m.geom_type[g])
        if t == 0 and (m.geom_size[g][0] == 0 or m.geom_size[g][0] > 0.3):
            continue                                   # the big floor
        b = int(m.geom_bodyid[g])
        if b != 0 and (root(b) == arm_root or m.body(b).name.startswith("bk")):
            continue
        rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
        if rgba[:3].min() > 0.85 and rgba[3] > 0.5:
            h = {5: m.geom_size[g][1], 6: m.geom_size[g][2]}.get(t, 0.0)
            found.append((m.geom(g).name or f"geom{g}", d.geom_xpos[g].copy(), float(d.geom_xpos[g][2] + h)))
    for s in range(m.nsite):
        b = int(m.site_bodyid[s])
        if b != 0 and root(b) == arm_root:
            continue
        if m.site_rgba[s][:3].min() > 0.85 and m.site_rgba[s][3] > 0.5:
            found.append((m.site(s).name or f"site{s}", d.site_xpos[s].copy(), float(d.site_xpos[s][2])))
    for name, pos, top in found:
        print(f"   white thing: {name} at {np.round(pos, 3)}")
    if not found:
        return None, 0.0
    name, pos, top = found[0]
    return pos[:2].copy(), float(max(0.0, min(top, 0.02)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="no window")
    ap.add_argument("--video", default=None, help="save mp4, e.g. run.mp4")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sim = Sim(args.headless, args.video)
    arm = Arm(sim)
    centre = arm.zone_centre()
    target_xy, pad_h = find_target(sim.m, sim.d, arm)
    if target_xy is None:
        print("Could not find the white box. Stacking at the middle of the work zone instead. "
              "(Set TARGET_XY at the top of this file to choose a spot.)")
        target_xy, pad_h = centre.copy(), 0.0
    print(f"stack target (white box): x={target_xy[0]:.3f} y={target_xy[1]:.3f}")
    print(f"   distance from arm base: {np.linalg.norm(target_xy - arm.base[:2]):.3f} m (arm reaches {arm.rmin:.3f} - {arm.rmax:.3f} m on the table)")

    view = 0.5 * (centre + target_xy)
    if sim.vcam is not None:
        sim.vcam.lookat[:] = [view[0], view[1], 0.05]
    cam = DepthPerception(sim.m)
    cam.aim(view)

    # scatter blocks in the arm's reachable zone, away from the target
    rng = np.random.default_rng(args.seed)
    rlo = max(0.10, 0.45 * arm.rmax)
    rhi = max(rlo + 0.04, 0.70 * arm.rmax)
    pts = []
    for i in range(N_BLOCKS):
        for _ in range(500):
            r, a = rng.uniform(rlo, rhi), arm.phi0 + rng.uniform(-0.55, 0.55)
            p = arm.base[:2] + r * np.array([np.cos(a), np.sin(a)])
            if all(np.linalg.norm(p - q) > 0.07 for q in pts) and np.linalg.norm(p - target_xy) > 0.09:
                break
        pts.append(p)
        yaw = rng.uniform(-np.pi / 4, np.pi / 4)
        adr = int(sim.m.jnt_qposadr[sim.m.joint(f"bk{i}_j").id])
        sim.d.qpos[adr:adr + 7] = [p[0], p[1], BH / 2 + 0.002, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(sim.m, sim.d)
    sim.seconds(1.0)

    def perceive():
        arm.park()
        sim.seconds(0.5)
        dets = cam.detect(sim.d, save=os.path.join(HERE, "perception.png"))
        keep = [b for b in dets if 0.7 * arm.rmin < np.linalg.norm(b["xy"] - arm.base[:2]) < 1.1 * arm.rmax]
        for b in keep:
            print(f"   sees block x={b['xy'][0]:.3f} y={b['xy'][1]:.3f} yaw={np.degrees(b['yaw']):.0f}deg level={b['level']}")
        return keep

    # failed-grasp bookkeeping: [[xy, count], ...], blocks matched by position (within 4 cm)
    fails = []

    def fail_count(xy):
        for pos, c in fails:
            if np.linalg.norm(pos - xy) < 0.04:
                return c
        return 0

    def add_fail(xy):
        for f in fails:
            if np.linalg.norm(f[0] - xy) < 0.04:
                f[1] += 1
                f[0] = xy.copy()
                return
        fails.append([xy.copy(), 1])

    try:
        print("Perceiving...")
        for step in range(6 * N_BLOCKS):
            dets = perceive()
            on_stack = [b for b in dets if np.linalg.norm(b["xy"] - target_xy) < 0.045]
            level = max([b["level"] for b in on_stack], default=0)
            loose = [b for b in dets if b["level"] == 1 and np.linalg.norm(b["xy"] - target_xy) > 0.06]
            if level >= N_BLOCKS or not loose:
                break
            cand = [b for b in loose if fail_count(b["xy"]) < MAX_TRIES]
            if not cand:
                print("Giving up: every remaining block failed too many times.")
                break
            b = min(cand, key=lambda b: np.linalg.norm(b["xy"] - target_xy))
            att = fail_count(b["xy"])
            print(f"Picking block at {np.round(b['xy'], 3)}, tower height now {level}, "
                  f"attempt {att + 1}/{MAX_TRIES}")
            ok = arm.pick_place(b["xy"], b["yaw"], target_xy, 0.0, level, pad_h, attempt=att)
            print(f"   picked={arm.picked}  placed={ok}")
            if not ok:
                add_fail(b["xy"])
        dets = perceive()
        h = max([b["level"] for b in dets if np.linalg.norm(b["xy"] - target_xy) < 0.045], default=0)
        print(f"DONE: tower on the white box is {h} blocks high (goal {N_BLOCKS}).")
        if sim.viewer is not None:
            print("Close the viewer window to exit.")
            while True:
                sim.step(5)
    finally:
        if sim.frames:
            import imageio
            imageio.mimsave(args.video, sim.frames, fps=30)
            print("saved", args.video)


if __name__ == "__main__":
    main()