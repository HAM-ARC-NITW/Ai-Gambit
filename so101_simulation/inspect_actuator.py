import mujoco

model = mujoco.MjModel.from_xml_path("chess_world.xml")
data = mujoco.MjData(model)

for aid in range(model.nu):
    name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid
    )
    print(f"\nActuator {aid}: {name}")
    print("  Type:", model.actuator_biastype[aid])
    print("  Gain parameters:", model.actuator_gainprm[aid])
    print("  Bias parameters:", model.actuator_biasprm[aid])
    print("  Dynamics parameters:", model.actuator_dynprm[aid])

jid = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_pan"
)
print("\nShoulder joint ID:", jid)
print("Shoulder DOF address:", model.jnt_dofadr[jid])
print("Gravity:", model.opt.gravity)
print("Solver iterations:", model.opt.iterations)
