import mujoco

urdf_path = "assets/so101_leader_new_calib.urdf"
xml_path = "assets/so101_leader_new_calib.xml"

# Read URDF and fix path prefixes if needed
with open(urdf_path, "r") as f:
    content = f.read()

# Fix paths like "assets/" to match the local structure if needed
# This replaces any double 'assets/assets/' or fixes relative references
content = content.replace("filename=\"assets/", "filename=\"")

temp_urdf = "assets/temp_fixed.urdf"
with open(temp_urdf, "w") as f:
    f.write(content)

# Load and save as MuJoCo XML
model = mujoco.MjModel.from_xml_path(temp_urdf)
mujoco.mj_saveLastXML(xml_path, model)
print("Successfully converted URDF to MuJoCo XML!")

