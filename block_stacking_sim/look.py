import numpy as np
from PIL import Image
from env import RowEnv

env = RowEnv()
obs = env.reset()

Image.fromarray(obs["image"]).save("view_topdown.png")
Image.fromarray(env.get_obs("overview")["image"]).save("view_overview.png")

base = env.data.xanchor[env.model.joint("shoulder_pan").id]
print("joint angles at start:", np.round(obs["qpos"], 3))
print("\nblock positions after settling (x, y, z):")
for i, p in enumerate(env.block_positions()):
    dist = np.hypot(p[0] - base[0], p[1] - base[1])
    print(f"   bk{i}: {np.round(p, 3)}   {dist:.3f} m from the arm base")
print("\nblocks on the white square:", env.blocks_on_square())
print("saved view_topdown.png and view_overview.png")