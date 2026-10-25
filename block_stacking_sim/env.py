import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "SO-ARM100", "Simulation", "SO101", "row_scene.xml")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CONTROL_HZ = 30


class RowEnv:
    def __init__(self, height=480, width=640, render=True):
        self.model = mujoco.MjModel.from_xml_path(SCENE)
        self.data = mujoco.MjData(self.model)
        m = self.model

        # stickier contacts so a gripped block does not slip
        m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        m.opt.impratio = 10

        # give the gripper jaws real friction (they came with torsion/rolling = 0)
        for g in range(m.ngeom):
            body = m.body(int(m.geom_bodyid[g])).name.lower()
            if "gripper" in body or "jaw" in body:
                m.geom_friction[g] = [1.5, 0.02, 0.002]
                m.geom_condim[g] = 4

        self.qadr = [int(m.jnt_qposadr[m.joint(n).id]) for n in JOINTS]
        self.act = []
        for n in JOINTS:
            jid = m.joint(n).id
            self.act.append([a for a in range(m.nu) if int(m.actuator_trnid[a, 0]) == jid][0])
        self.blocks = [m.body(f"bk{i}").id for i in range(100)
                       if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"bk{i}") >= 0]
        self.sq = m.geom("white_square").id
        self.substeps = max(1, round(1 / (CONTROL_HZ * m.opt.timestep)))
        self.renderer = mujoco.Renderer(m, height, width) if render else None
        self.reset()

    # ---------- what the ENGINE is allowed to use ----------
    def reset(self, seed=None, jitter=0.0):
        """Put everything back. jitter = random sideways nudge of each block in metres."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if jitter > 0:
            rng = np.random.default_rng(seed)
            for b in self.blocks:
                adr = int(m.jnt_qposadr[m.body_jntadr[b]])
                d.qpos[adr:adr + 2] += rng.uniform(-jitter, jitter, 2)
        for q, a in zip(self.qadr, self.act):
            d.ctrl[a] = d.qpos[q]
        mujoco.mj_forward(m, d)
        for _ in range(int(0.5 / m.opt.timestep)):     # let blocks settle
            mujoco.mj_step(m, d)
        return self.get_obs()

    def get_obs(self, camera="topdown"):
        qpos = np.array([self.data.qpos[q] for q in self.qadr])
        image = None
        if self.renderer is not None:
            self.renderer.update_scene(self.data, camera=camera)
            image = self.renderer.render().copy()
        return {"image": image, "qpos": qpos}

    def step(self, action):
        """action = 6 joint targets (radians) in JOINTS order."""
        for a, v in zip(self.act, action):
            self.data.ctrl[a] = v
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        return self.get_obs()

    # ---------- scoring only: the engine must NOT call these ----------
    def block_positions(self):
        return np.array([self.data.xpos[b] for b in self.blocks])

    def blocks_on_square(self):
        c = self.data.geom_xpos[self.sq]
        half = self.model.geom_size[self.sq][0]
        p = self.block_positions()
        return int(np.sum((np.abs(p[:, 0] - c[0]) < half) & (np.abs(p[:, 1] - c[1]) < half)))