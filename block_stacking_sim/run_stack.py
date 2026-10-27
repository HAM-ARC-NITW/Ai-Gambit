import subprocess
import sys
import numpy as np
from env import RowEnv
from engine import StackEngine

JITTER = 0.0       # set to 0.01 to nudge the blocks around randomly (metres)

env = RowEnv()
env.reset(seed=1, jitter=JITTER)
eng = StackEngine(env)
try:
    eng.run()
finally:
    sq = env.data.geom_xpos[env.sq]
    pos = env.block_positions()
    on = [p for p in pos if np.linalg.norm(p[:2] - sq[:2]) < 0.03]
    heights = sorted(round(float(p[2]), 3) for p in on)
    print(f"\nRESULT (true positions, for scoring only): {len(on)} block(s) stacked on the square, "
          f"centre heights {heights}")
    if on:
        print(f"top block is {np.linalg.norm(max(on, key=lambda p: p[2])[:2] - sq[:2]) * 1000:.0f} mm off the square centre")
    if len(eng.frames) > 0:
        np.save("stack_run.npy", np.array(eng.frames))
        print("opening the live MuJoCo window...")
        subprocess.run([sys.executable, "replay.py"])
