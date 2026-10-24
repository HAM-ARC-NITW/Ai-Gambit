import time
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("chess_world.xml")
data = mujoco.MjData(model)

targets = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.3,
    "elbow_flex": 0.8,
    "wrist_flex": -0.5,
    "wrist_roll": 0.0,
    "gripper": 0.5,
}

# Map each actuator to the joint it controls.
actuator_ids = {}

for aid in range(model.nu):
    joint_id = model.actuator_trnid[aid, 0]
    joint_name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id)
    )
    if joint_name:
        actuator_ids[joint_name] = aid

print("Actuators mapped to joints:", actuator_ids)

for joint_name, angle in targets.items():
    if joint_name not in actuator_ids:
        raise ValueError(f"No actuator found for joint: {joint_name}")
    data.ctrl[actuator_ids[joint_name]] = angle

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
