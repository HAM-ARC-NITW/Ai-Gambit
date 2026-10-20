import time

import mujoco
import mujoco.viewer

MODEL_PATH = "chess_world.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# The actuators are ordered as declared in chess_world.xml.
joint_names = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

if model.nu != len(joint_names):
    raise ValueError(
        f"Expected {len(joint_names)} actuators, found {model.nu}"
    )

# Map joint names to joint IDs and qpos addresses.
joint_info = {}

for i, name in enumerate(joint_names):
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if joint_id == -1:
        raise ValueError(f"Joint not found: {name}")

    qpos_address = model.jnt_qposadr[joint_id]

    joint_info[name] = {
        "actuator_id": i,
        "joint_id": joint_id,
        "qpos_address": qpos_address,
    }

    print(f"{name}: actuator ID={i}, joint ID={joint_id}")

# Initialize the simulation.
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

# Hold all joints at their initial positions.
for name, info in joint_info.items():
    data.ctrl[info["actuator_id"]] = data.qpos[
        info["qpos_address"]
    ]

shoulder_info = joint_info["shoulder_pan"]
initial_position = data.qpos[shoulder_info["qpos_address"]]
target_position = initial_position + 0.15

duration = 2.0

print("\nStarting simulation...")
print("Shoulder target:", target_position, "radians")

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.monotonic()

    while viewer.is_running():
        elapsed = time.monotonic() - start_time

        # Smoothly interpolate toward the target.
        alpha = min(elapsed / duration, 1.0)

        data.ctrl[shoulder_info["actuator_id"]] = (
            initial_position
            + alpha * (target_position - initial_position)
        )

        mujoco.mj_step(model, data)
        if int(elapsed * 10) != int((elapsed - model.opt.timestep) * 10):
           joint_id = joint_info["shoulder_pan"]["joint_id"]
           qpos_address = joint_info["shoulder_pan"]["qpos_address"]

           print(
           f"Target: {data.ctrl[shoulder_info['actuator_id']]:.3f}, "
           f"Actual: {data.qpos[qpos_address]:.3f}, "
           f"Velocity: {data.qvel[model.jnt_dofadr[joint_id]]:.3f}"
           )       
        viewer.sync()

        time.sleep(model.opt.timestep)

print("Simulation closed.")
