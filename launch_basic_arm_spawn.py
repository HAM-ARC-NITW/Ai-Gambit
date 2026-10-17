import mujoco, mujoco.viewer
import cv2
import numpy as np

model = mujoco.MjModel.from_xml_path("./simulation/scene.xml")
data = mujoco.MjData(model)

#this is required to get the default position to spawn properly
mujoco.mj_resetDataKeyframe(model, data, 0)

#spawn camera
renderer = mujoco.Renderer(model, height=480, width=640)

lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]

#code for live video feed
with mujoco.viewer.launch_passive(model, data) as v:
    
    step = 0

    while v.is_running():
        t = data.time

        # order: pan, lift, elbow, wrist_flex, wrist_roll, gripper
        data.ctrl[0] = 0.8 * np.sin(2 * np.pi * 0.25 * t)
        #that is a simple sinusoidal signal to pan the arm
        data.ctrl[:] = np.clip(data.ctrl, lo, hi)
        #clipping values to be in range of the arms limits

        mujoco.mj_step(model, data)
        v.sync()

        if step % 10 == 0:
            renderer.update_scene(data, camera="top_camera")
            cv2.imshow("top", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        step += 1

    renderer.close()
    cv2.destroyAllWindows()

#print final positions of all joints once the simulation is done
print(data.qpos)