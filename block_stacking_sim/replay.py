import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from env import load_model

m = load_model()
d = mujoco.MjData(m)
here = Path(__file__).resolve().parent
recording = here/"results/stack_run.npy"
if not recording.is_file():recording = here/"stack_run.npy"  # Historical recording bundled with the project.
frames = np.load(recording)
if frames.ndim!=2 or frames.shape[1]!=m.nq:
    raise SystemExit("Recording does not match this scene; create one with run_simulation.py --stack first")
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
