import mujoco

m = mujoco.MjModel.from_xml_path("chess_world.xml")

for i in range(m.ngeom):
    g = m.geom(i)
    body_name = m.body(g.bodyid).name

    print(
        f"Geom ID: {i:2d} | "
        f"Name: {g.name or '(unnamed)':35s} | "
        f"Body: {body_name:20s} | "
        f"Type: {g.type} | "
        f"contype: {g.contype} | "
        f"conaffinity: {g.conaffinity}"
    )
