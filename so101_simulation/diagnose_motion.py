import mujoco

model = mujoco.MjModel.from_xml_path("chess_world.xml")
data = mujoco.MjData(model)

joint_name = "shoulder_pan"

joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    joint_name,
)

if joint_id < 0:
    raise ValueError(f"Joint not found: {joint_name}")

actuator_id = next(
    (
        i for i in range(model.nu)
        if model.actuator_trnid[i, 0] == joint_id
    ),
    -1,
)

if actuator_id < 0:
    raise ValueError("No actuator found for shoulder_pan")

qpos_id = model.jnt_qposadr[joint_id]
dof_id = model.jnt_dofadr[joint_id]

print("Joint:", joint_name)
print("Joint ID:", joint_id)
print("Actuator ID:", actuator_id)
print("Joint range:", model.jnt_range[joint_id])
print("Timestep:", model.opt.timestep)
print("Position gain:", model.actuator_gainprm[actuator_id, 0])
print("Velocity gain:", model.actuator_biasprm[actuator_id, 2])

initial_position = 0.0
target_position = 0.15
duration = 2.0

data.qpos[qpos_id] = initial_position
data.qvel[dof_id] = 0.0

# Hold all joints at their initial positions.
for i in range(model.nu):
    data.ctrl[i] = data.qpos[model.jnt_qposadr[
        model.actuator_trnid[i, 0]
    ]]

mujoco.mj_forward(model, data)

steps = int(duration / model.opt.timestep)
report_every = max(1, steps // 10)

print("\nRunning simulation...")

for step in range(steps):
    elapsed = step * model.opt.timestep
    alpha = min(elapsed / duration, 1.0)

    data.ctrl[actuator_id] = (
        initial_position
        + alpha * (target_position - initial_position)
    )

    mujoco.mj_step(model, data)

    if step % report_every == 0:
        print(
            f"t={data.time:.3f}s | "
            f"target={data.ctrl[actuator_id]:.3f} | "
            f"actual={data.qpos[qpos_id]:.3f} | "
            f"velocity={data.qvel[dof_id]:.3f} | "
            f"force={data.actuator_force[actuator_id]:.3f}"
        )

print("\nFinal position:", data.qpos[qpos_id])
print("Final velocity:", data.qvel[dof_id])
print("Final target:", data.ctrl[actuator_id])
