# Basic setup

make sure you have mujoco and opencv installed

`pip install mujoco`  
`pip install opencv-python`

---
### Spawning the arm


the necessary stl and config files are stored in [./simulation](./simulation)  
the main file that the simulator loads is called [scene.xml](./simulation/scene.xml)  
default spawn positions can be changed in the keyframe section of the scene.xml file  

| Joint | Range (rad) |
| ----- | ----------- |
| shoulder_pan | ±1.92 |
| shoulder_lift | ±1.75 |
| elbow_flex | ±1.69 |
| wrist_flex | ±1.66 |
| wrist_roll | -2.74 to 2.84 |
| gripper | -0.17 to 1.75 |

the actual robotic arm joints positions and stuff is defined in the [so101_new_calib.xml](./simulation/so101_new_calib.xml)  

---
### Camera

a camera has also been spawned, defined in the wordlbody section of the [scene.xml](./simulation/scene.xml) file.  
pos is the cameras position in world coordinates (25cm in front of the base plate of the arm, and 60cm up)  
fovy is the vertical field of view in degrees (change this according to the fov of our actual camera)  

the single python file to run a panning simulation is [launch_basic_arm_spawn.py](./launch_basic_arm_spawn.py)  
run it as follows:  
`python3 ./launch_basic_arm_spawn.py`

apart from that the code has relevant comments, you can read and understand stuff there

