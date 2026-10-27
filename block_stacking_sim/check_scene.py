import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path("SO-ARM100/Simulation/SO101/stack_scene.xml")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
np.set_printoptions(precision=3, suppress=True)
for s in ["baseframe", "gripperframe"]:
    i = m.site(s).id
    print(s, "pos:", d.site_xpos[i])
print("gripperframe axes (columns = local x, y, z in world):")
print(d.site_xmat[m.site("gripperframe").id].reshape(3, 3))
