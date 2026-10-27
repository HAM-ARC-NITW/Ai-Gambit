import time
import numpy as np
import mujoco
import mujoco.viewer
from env import load_model

m = load_model()
d = mujoco.MjData(m)
frames = np.load("stack_run.npy")
print(f"opening the MuJoCo window: {len(frames)} frames ({len(frames) / 30:.0f} s). Close the window to exit.")

viewer = mujoco.viewer.launch_passive(m, d)
viewer.cam.lookat[:] = [0.2, 0.08, 0.03]
viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.85, 150, -35
while viewer.is_running():
    for fq in frames:
        if not viewer.is_running():
            break
        with viewer.lock():
            d.qpos[:] = fq
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
        viewer.sync()
        time.sleep(1 / 30)
    time.sleep(1.5)
