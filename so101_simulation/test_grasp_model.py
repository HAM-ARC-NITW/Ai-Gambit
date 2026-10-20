import mujoco, numpy as np

xml = open('test_grasp_model.xml').read()
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)

site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'pinch_site')
pawn_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target_pawn')

# Settle pawn first
for _ in range(200):
    mujoco.mj_step(m, d)
pawn_init_pos = d.xpos[pawn_id].copy()
print('Pawn settled pos:', pawn_init_pos)

# IK solver to reach a target pinch position
def solve_ik(target_pos, q_cur):
    q = q_cur.copy()
    for it in range(80):
        d.qpos[:6] = q
        mujoco.mj_forward(m, d)
        cur_pos = d.site_xpos[site_id]
        err = target_pos - cur_pos
        if np.linalg.norm(err) < 1e-4:
            break
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, site_id)
        J = jacp[:, :5]
        dq = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(3)) @ err
        q[:5] += 0.5 * dq
        for j in range(5):
            q[j] = np.clip(q[j], m.jnt_range[j, 0], m.jnt_range[j, 1])
    return q

# Approach pose: 6cm above pawn, gripper open (0.8 rad)
q_cur = np.zeros(6)
q_cur[5] = 0.8
q_above = solve_ik(pawn_init_pos + np.array([0, 0, 0.06]), q_cur)
q_above[5] = 0.8

print('Moving above pawn...')
for step in range(500):
    alpha = min(1.0, step / 300)
    d.ctrl[:5] = alpha * q_above[:5]
    d.ctrl[5] = 0.8
    mujoco.mj_step(m, d)

# Descend to pawn neck height (~1.8 cm above base)
target_grasp = pawn_init_pos.copy()
target_grasp[2] += 0.012
q_grasp = solve_ik(target_grasp, d.qpos[:6])
q_grasp[5] = 0.8

print('Descending to grasp height...')
for step in range(500):
    alpha = min(1.0, step / 300)
    d.ctrl[:5] = q_above[:5] + alpha * (q_grasp[:5] - q_above[:5])
    d.ctrl[5] = 0.8
    mujoco.mj_step(m, d)

print('Pinch site at grasp:', d.site_xpos[site_id])
print('Pawn pos at grasp:', d.xpos[pawn_id])

# Close gripper
print('Closing gripper...')
for step in range(500):
    alpha = min(1.0, step / 300)
    d.ctrl[:5] = q_grasp[:5]
    d.ctrl[5] = 0.8 - alpha * 0.9 # close to -0.1
    mujoco.mj_step(m, d)

print('Contacts while gripping:')
grip_contacts = 0
for i in range(d.ncon):
    c = d.contact[i]
    b1 = m.body(m.geom(c.geom1).bodyid).name
    b2 = m.body(m.geom(c.geom2).bodyid).name
    if 'pawn' in (b1, b2):
        print(f'   {b1} <-> {b2}, dist={c.dist:.5f}')
        grip_contacts += 1

# Lift up by 7 cm
print('Lifting up...')
q_lift = solve_ik(target_grasp + np.array([0, 0, 0.07]), d.qpos[:6])
q_lift[5] = -0.1
for step in range(600):
    alpha = min(1.0, step / 400)
    d.ctrl[:5] = q_grasp[:5] + alpha * (q_lift[:5] - q_grasp[:5])
    d.ctrl[5] = -0.1
    mujoco.mj_step(m, d)

final_pawn_z = d.xpos[pawn_id][2]
print(f'Initial pawn Z: {pawn_init_pos[2]:.4f} m')
print(f'Final pawn Z:   {final_pawn_z:.4f} m')
print(f'Height lifted:  {(final_pawn_z - pawn_init_pos[2])*1000:.1f} mm')
if final_pawn_z > pawn_init_pos[2] + 0.03:
    print('SUCCESS: PAWN SUCCESSFULLY PICKED UP AND LIFTED INTO THE AIR!')
else:
    print('Pawn not lifted high enough.')
