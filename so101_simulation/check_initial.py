import mujoco

m = mujoco.MjModel.from_xml_path("chess_world.xml")
d = mujoco.MjData(m)

mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)

print("Initial joint positions:")
for name in [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]:
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    qadr = m.jnt_qposadr[jid]
    print(f"{name}: {d.qpos[qadr]:.4f} rad")

print(f"\nInitial contacts: {d.ncon}")
print("\nInitial floor contacts:")

for i in range(d.ncon):
    c = d.contact[i]
    g1 = m.geom(c.geom1)
    g2 = m.geom(c.geom2)

    if g1.name == "floor" or g2.name == "floor":
        other = g2 if g1.name == "floor" else g1
        body = m.body(other.bodyid)

        print(
            f"floor <-> {body.name or 'unnamed'} "
            f"(geom ID {other.id}), "
            f"distance={c.dist:.5f} m"
        )
