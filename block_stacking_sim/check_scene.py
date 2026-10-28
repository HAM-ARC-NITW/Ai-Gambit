import mujoco, numpy as np
from env import load_model
m = load_model()
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
np.set_printoptions(precision=3, suppress=True)
for s in ["baseframe", "gripperframe"]:
    i = m.site(s).id
    print(s, "pos:", d.site_xpos[i])
print("gripperframe axes (columns = local x, y, z in world):")
print(d.site_xmat[m.site("gripperframe").id].reshape(3, 3))
