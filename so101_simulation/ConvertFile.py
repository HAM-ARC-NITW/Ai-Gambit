import mujoco

# Load your URDF model
model = mujoco.MjModel.from_xml_path("so101_leader_new_calib.urdf")

# Save it out as a native MuJoCo XML file
mujoco.mj_saveLastXML("assets/so101_leader_new_calib.xml", model)
print("Successfully converted URDF to MuJoCo XML!")
