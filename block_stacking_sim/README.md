# Block Stacking Sim (SO-101 arm in MuJoCo)

A MuJoCo simulation of the SO-101 arm and a row of blocks. It is a test bench for a
camera-driven block-stacking engine: the engine will look at a camera image and tell the arm what
to do. The simulation checks that the engine works before it touches a real arm or LeRobot.

## Status

- Scene: 4 blocks in a row perpendicular to the arm, a white square at the end of the row, and a
  top-down camera.
- env.py: the test bench. It gives the engine a camera image and joint angles and takes 6 joint targets.
- watch.py: tries several grasps on one block, carries it to the white square, tries to place it,
  then replays the motion in the live MuJoCo window.
- Work in progress: reliable grasping, camera-based block detection, stacking.

## Setup (Ubuntu or WSL)

Clone the team repo on the simulation branch, then clone the robot model inside this folder:

    git clone -b feature/so101-simulation https://github.com/HAM-ARC-NITW/Ai-Gambit.git
    cd Ai-Gambit/block_stacking_sim
    git clone https://github.com/TheRobotStudio/SO-ARM100.git

Create a virtual environment and install the libraries:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

For the live window on WSL (Windows 11, or Windows 10 21H2 and newer):

    sudo apt update
    sudo apt install -y libgl1 libglfw3 libglu1-mesa mesa-utils x11-apps

Run xclock to test. If a clock window appears, windows work. If MuJoCo shows a graphics error,
run "export LIBGL_ALWAYS_SOFTWARE=1" first.

## Run

Activate the environment in every new terminal with "source .venv/bin/activate", then:

    python make_scene.py   # builds SO-ARM100/Simulation/SO101/row_scene.xml
    python look.py         # optional: saves two pictures and prints block positions
    python stack_engine.py # runs block stacking through the ROS perception node
    python watch.py        # optional legacy grasp test

The stack engine requires the ROS 2 environment containing `ai_gambit_perception`
to be sourced first. It feeds the simulated `topdown` camera directly through
the node's `process_frame` method; no ground-truth block positions are supplied
to the perception path. From the `Ai-Gambit` directory, source ROS and the
workspace before running it:

    source /opt/ros/humble/setup.bash
    source perception_ws/install/setup.bash
    python block_stacking_sim/stack_engine.py --headless

watch.py prints HELD or MISSED for each grasp attempt, then the final block position. Close the
window to exit.

## Settings

- make_scene.py: N_BLOCKS, ROW_X (distance of the row from the arm), GAP, white square position.
- watch.py: BLOCK (which block, 0 to 3), SPEED (replay speed), ATTEMPTS (grasp heights and shifts).
- block_dims.json: block size in metres as {"size": [x, y, z]}. Without it blocks are 25 mm cubes.

## Files

- make_scene.py: writes the MuJoCo scene file
- env.py: test bench (camera image + joint angles in, joint targets out)
- watch.py: pick-and-place attempt with the live viewer
- look.py: quick check of the scene
- check_setup.py: prints the robot's joints, actuators and reach
- stack_engine.py: runs the simulation camera through the ROS `AiGambitPerceptionNode`
  and converts its `PoseArray` pixel detections into simulation coordinates
- perception.py: legacy MuJoCo depth-camera detector (kept for standalone experiments)

## Planned

1. Reliable grasping in simulation.
2. Engine that finds blocks from the camera image only.
3. Stack the blocks on the white square.
4. Match observations and actions to the real SO-101 so it plugs into LeRobot.
