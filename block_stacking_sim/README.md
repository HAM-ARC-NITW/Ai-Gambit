# SO-101 block stacking in MuJoCo

The scene and existing stacking experiments now run using the robot and meshes
already in `../simulation`. **No separate SO-ARM100 clone is required.** The
generated scene has four 30 mm blocks, a white target and an overhead camera.

Setup is working; reliable four-block stacking is still unfinished. The former
IK solver falsely rejected upper tower poses because it favoured an upright wrist
over accurate positioning. Its corrected position/orientation weighting now finds
all four target levels while enforcing joint and tilt limits. Placement/grasp
stability and tower verification still need work. Nothing here controls a real
robot or uses MoveIt.

After the IK/clearance correction, one native end-to-end run formed a four-block
tower with centre heights approximately 17, 46, 76 and 106 mm. Its report and
recording are in `results/reach_fix_result.json` and `results/reach_fix_run.npy`.
This demonstrates the corrected motion; it is not a repeated-trial reliability
claim. The ROS runner's separate grasp calibration limitations below still apply.

The ROS runner also reports an approximately 25 mm calibrated open gap against
the supplied 30 mm blocks, and its bounded test missed the attempted grasps.
Gripper calibration/contact geometry or the intended block size must be resolved
before calling that runner a working stacking game. The supplied block dimensions
have been retained rather than silently shrinking them.

## Windows: easiest way to run

From the repository root in PowerShell, install dependencies into the existing
project environment, then open the scene:

```powershell
.\.venv\Scripts\python.exe -m pip install -r block_stacking_sim/requirements.txt
.\block_stacking_sim\run_simulation.cmd
```

Or double-click `run_simulation.cmd`. If `.venv` does not exist yet, first run
`py -3.13 -m venv .venv`. The launcher resolves models beside the scripts, so it
also works from another working directory. Drag/scroll in MuJoCo to inspect the
scene; close the window or press Ctrl+C in its console to stop.

```powershell
# Run the existing RGB stacking engine, with a live 3D viewer.
.\block_stacking_sim\run_simulation.cmd --stack

# Run the same attempts without windows; bound the motion phase by wall time.
.\block_stacking_sim\run_simulation.cmd --stack --headless --seconds 60

# Check actual rendered detections against independent simulator positions.
.\block_stacking_sim\run_simulation.cmd --check

# Physics-only smoke check; no camera or viewer is allocated.
.\block_stacking_sim\run_simulation.cmd --headless --seconds 2
```

`run_stack.py` is a compatibility entry point for `run_simulation.py --stack`.
The viewer updates at about 30 Hz and camera rendering has shadows/MSAA disabled
to reduce Windows overhead. During IK the arm pauses while planning, but the
viewer and cancellation remain responsive. `--seconds` is checked during planning
as well as motion. The initial leftward move parks the arm for the camera;
reach checking and pickup follow. Closing the viewer stops the live run.

The native tower reach check now completes in under a second on this machine.
IK uses kinematic transforms/Jacobians rather than contact simulation for each
trial pose. Carry clearance is 30 mm above the landing pose to avoid unnecessarily
high, tilted approaches. Five regression checks cover upper poses, independent
forward-kinematics validation, unreachable points, tilt limits and planning callbacks:
`python -m pytest block_stacking_sim/test_motion.py -q`.

## What was missing and what changed

- Original scripts referenced a missing `SO-ARM100/Simulation/SO101` checkout and
  inconsistent `row_scene.xml` / `stack_scene.xml` files. `make_scene.py` now builds
  `generated/row_scene.xml` from the shared model and uses absolute mesh paths.
- SciPy, Pillow, imageio and the MP4 backend were missing from the Windows venv;
  dependencies have been installed. WSL has its own compatible Linux dependencies.
- The resting arm hides blocks. The camera check parks it to the side before
  detection, as the RGB engine does. Adjusted lighting avoids washed-out colours.
- The generated arm uses neutral grey materials because the ROS colour detector
  assumes an unsaturated arm. This changes appearance only, not the shared model,
  dimensions or dynamics. Finger pads remain the existing simplified contact model.
- Added a Windows launcher, bounded/headless checks, resource cleanup and a WSL
  launcher preserving the newer ROS adapter in `stack_engine.py`.
- Fixed the ROS block message's integer `Point.z` assignment, which Humble rejects;
  it is now explicitly a float.

The camera setup check detected all four blocks within about 1 mm of independent
simulator XY positions and the target within 0.1 mm in this synthetic scene.
This does not establish real-camera accuracy or grasp success.

## Outputs and settings

Generated XML lives in `generated/`; run results live in `results/`. Both are
gitignored. `--check` writes `setup_check.json` and `top_camera.png`. A stacking
attempt writes `last_run.json`, `camera_last.png` and `stack_run.npy`.
`blocks_over_target` is an XY footprint score, **not proof of a stable tower**;
a block in the gripper can project over the target too. Interrupted runs save
their current state, which can include a held or moving block.

To replay the latest native recording:

```powershell
.\.venv\Scripts\python.exe block_stacking_sim/replay.py
```

If no new recording exists, replay falls back to the historical bundled
`stack_run.npy`. Recordings must match the current model's joint count and scene
settings. Replay displays stored positions; it does not re-run perception.

Edit `make_scene.py` for the row and target position, and `block_dims.json` for
block dimensions. The main scene uses analytical box geometry matching those
dimensions; `meshes/block1.stl` is for the optional mesh experiment. Changes are
rebuilt at launch. Use `reach_check.py` to explore reachable target locations.

## ROS 2 Humble in WSL

The newer `stack_engine.py` uses the repository's `AiGambitPerceptionNode`.
Native Windows Python cannot import Ubuntu's `rclpy`; use WSL Ubuntu 22.04 and a
Linux Python 3.10 venv with system packages. This machine already has Humble and
the Linux venv at `~/.local/share/ai-gambit/venv`. From WSL at the repository root:

```bash
bash ros_tools/run_block_stacking.sh --headless --check
bash ros_tools/run_block_stacking.sh
# A bounded run without the viewer:
bash ros_tools/run_block_stacking.sh --headless --seconds 30
# Optional video (captures RGB frames in memory; start with a short run):
bash ros_tools/run_block_stacking.sh --headless --seconds 5 --video /tmp/stack.mp4
```

The script sources `/opt/ros/humble`, activates the Linux venv, and uses ROS domain
53 by default. Override `AI_GAMBIT_ROS_VENV` or `ROS_DOMAIN_ID` if needed. WSLg is
required for the live viewer; headless camera rendering still needs working Mesa/
OpenGL libraries. A fresh Linux venv can be prepared with:

```bash
python3 -m venv --system-site-packages ~/.local/share/ai-gambit/venv
source ~/.local/share/ai-gambit/venv/bin/activate
pip install 'numpy>=1.26,<2' 'opencv-python>=4.8,<4.12' 'mujoco>=3.2,<4' 'scipy>=1.13,<1.16' pillow imageio imageio-ffmpeg
```

This assumes apt Humble, cv_bridge and standard ROS message packages are already
installed. Keep NumPy below 2 for Humble's cv_bridge. Do not activate the Windows
`.venv` in WSL. No custom-message build is required for this particular adapter.

The adapter calls the ROS node's `process_frame` method directly with rendered
BGR frames. This is not an Image-topic/ROS-action motion pipeline: MuJoCo movement
still uses direct calls. The current node reports image centres, fixed yaw zero
and level one; its block outputs do not measure tower height. The engine also
uses simulator body positions internally for grasp/transport checks, so this path
is not fully perception-only autonomous control. Target discovery still inspects
model geometry. These are existing prototype limitations, not completed features.

## Existing experiments

The existing scripts remain available rather than deleting your experiments:

| Files | Purpose |
| --- | --- |
| `env.py`, `engine.py`, `vision.py` | Native RGB runner and camera-based block/target detection. |
| `stack_engine.py` | Separate ROS perception adapter and experimental controller. |
| `watch.py`, `grasp_test.py`, `check_pads.py` | Individual grasp/contact investigations. |
| `check_setup.py`, `check_scene.py`, `inspect_model.py`, `workspace.py`, `reach_check.py` | Model and workspace diagnostics using the shared model. |
| `stack.py` | Older standalone cube experiment; writes its own generated scene. |
| `perception.py` | Older depth/segmentation detector; simulation-specific, not the current ROS adapter. |
| `make_cube_stl.py`, `gen_blocks.py` | Optional mesh/dimension generation; not needed for normal launch. |

Older diagnostic scripts may write pictures into their working directory. Prefer
the new launcher and `--check` for repeatable setup validation. The old
`run_log.txt` is historical and is not evidence of the current run.
