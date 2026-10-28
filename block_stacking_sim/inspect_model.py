import mujoco, sys
from env import load_model
m = mujoco.MjModel.from_xml_path(sys.argv[1]) if len(sys.argv)>1 else load_model()
print("JOINTS")
for i in range(m.njnt):
    print(" ", i, m.joint(i).name, m.jnt_range[i])
print("ACTUATORS")
for i in range(m.nu):
    print(" ", i, m.actuator(i).name, m.actuator_ctrlrange[i])
print("SITES")
for i in range(m.nsite):
    print(" ", i, m.site(i).name)
