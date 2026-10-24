import mujoco

m = mujoco.MjModel.from_xml_path("chess_world.xml")
d = mujoco.MjData(m)

mujoco.mj_resetData(m,d)
# Initialize all actuators to their current joint positions
for a in range(m.nu):
    j = m.actuator_trnid[a, 0]
    if j >= 0:
        d.ctrl[a] = d.qpos[m.jnt_qposadr[j]]

# Move shoulder_pan to +0.15 radians
pan_act = mujoco.mj_name2id(
    m, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_pan"
)

for step in range(1000):
    fraction = min(1.0, (step + 1) / 300)
    d.ctrl[pan_act] = 0.15 * fraction

    mujoco.mj_step(m, d)

    if d.ncon > 0:
        print(f"First contact at t={d.time:.4f}s")
        print(f"Contacts: {d.ncon}")

        for i in range(d.ncon):
            c = d.contact[i]
            g1 = m.geom(c.geom1)
            g2 = m.geom(c.geom2)

            b1 = m.body(g1.bodyid).name or "world"
            b2 = m.body(g2.bodyid).name or "world"

            print(
                f"{b1} [{g1.name or c.geom1}] <-> "
                f"{b2} [{g2.name or c.geom2}] "
                f"distance={c.dist:.5f} m"
            )

        break
else:
    print("No contacts detected during movement.")
