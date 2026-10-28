import os
import numpy as np
import mujoco
from make_scene import OUT,build_scene

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = str(OUT)
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CONTROL_HZ = 30

# ---- the fix from the article (ggando.com/blog/so101-rl-lift) ----
ADD_FINGER_PADS = True                  # tiny box pads at the fingertips
DISABLE_FINGER_MESH_COLLISION = False   # leave False unless check_pads.py shows the pads are well placed
PAD_HALF = 0.00125                      # 2.5 mm boxes
PAD_FRICTION = [1.0, 0.05, 0.001]
PADS = [  # (body, name, position in that body's frame, colour)
    ("gripper", "static_finger_pad", [-0.008875, 0.0, -0.100], [1, 0.5, 0.5, 0.8]),
    ("moving_jaw_so101_v1", "moving_finger_pad", [-0.01136, -0.076, 0.019], [0.5, 0.5, 1, 0.8]),
]


def load_model():
    build_scene()  # Refresh editable settings and resolve assets independently of cwd.
    spec = mujoco.MjSpec.from_file(SCENE)
    if ADD_FINGER_PADS:
        for body, name, pos, rgba in PADS:
            b = spec.body(body)
            if b is None:
                raise SystemExit(f"Body '{body}' not found in the model.")
            b.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[PAD_HALF] * 3,
                       pos=pos, rgba=rgba, friction=PAD_FRICTION, condim=4)
    return spec.compile()


class RowEnv:
    def __init__(self, height=480, width=640, render=True):
        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        m = self.model

        # stickier contacts so a gripped block does not slip
        m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        m.opt.impratio = 10
        m.opt.noslip_iterations = 3

        for g in range(m.ngeom):
            body = m.body(int(m.geom_bodyid[g])).name.lower()
            if "gripper" in body or "jaw" in body:
                if m.geom(g).name.endswith("_finger_pad"):
                    continue
                m.geom_friction[g] = [1.0, 0.05, 0.001]
                m.geom_condim[g] = 4
                if DISABLE_FINGER_MESH_COLLISION and ADD_FINGER_PADS:
                    m.geom_contype[g] = 0
                    m.geom_conaffinity[g] = 0

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
        self.on_step = None  # Optional live viewer callback used by the launcher.
        self.on_plan = None  # Pump UI/cancellation during an IK search, without stepping physics.
        if self.renderer is not None:
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
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

    def get_image(self, camera="topdown"):
        if self.renderer is None:raise ValueError("Camera images require RowEnv(render=True)")
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def get_obs(self, camera="topdown"):
        qpos = np.array([self.data.qpos[q] for q in self.qadr])
        image = self.get_image(camera) if self.renderer is not None else None
        return {"image": image, "qpos": qpos}

    def step(self, action, render=True):
        """action = 6 joint targets (radians) in JOINTS order."""
        action = np.asarray(action,dtype=float)
        if action.shape!=(6,) or not np.isfinite(action).all():
            raise ValueError("Action must contain six finite joint targets")
        for a, v in zip(self.act, action):
            self.data.ctrl[a] = v
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        if self.on_step is not None:self.on_step()
        if render:
            return self.get_obs()
        return {"image": None, "qpos": np.array([self.data.qpos[q] for q in self.qadr])}

    def close(self):
        if self.renderer is not None:self.renderer.close()

    # ---------- scoring only: the engine must NOT call these ----------
    def block_positions(self):
        return np.array([self.data.xpos[b] for b in self.blocks])

    def blocks_on_square(self):
        c = self.data.geom_xpos[self.sq]
        half = self.model.geom_size[self.sq][0]
        p = self.block_positions()
        return int(np.sum((np.abs(p[:, 0] - c[0]) < half) & (np.abs(p[:, 1] - c[1]) < half)))
