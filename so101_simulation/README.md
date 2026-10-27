# SO-101 chess simulation

This folder contains the MuJoCo chess scene, movement code, RGB perception adapter,
playable game and visual experiments. The [main project README](../README.md)
explains the complete architecture, what has been implemented, all 56 perception
scenarios, validation results, limitations and pending work.

**The chess opponent uses Stockfish.** python-chess supplies the rules and
confirmed position. The [engine guide](../engine_stuff/README.md) explains setup,
`--think-time`, `--skill`, optional `--depth` and executable selection. The former
negamax placeholder has been removed; camera verification still gates every move.

**Full ROS 2 integration is pending.** These Windows runners call Python/MuJoCo
directly. Prepared ROS perception interfaces exist under `../perception_ws/src`,
but ROS runtime validation, movement actions/nodes, MoveIt 2 and ros2_control
integration remain to be done.

## Setup and entry points

From the repository root in PowerShell, create `.venv` if needed and install:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\so101_simulation\requirements.txt
powershell -File .\engine_stuff\install_stockfish.ps1

# Play White against the arm: 3D viewer, overhead camera and occupancy grid.
.\so101_simulation\play_chess.cmd

# Inspect camera experiments: Space/N advances; Escape exits.
.\so101_simulation\run_perception.cmd

# Run the original motion-only demonstration.
.\.venv\Scripts\python.exe .\so101_simulation\run_simulation.py

# Generate the full visual report.
.\.venv\Scripts\python.exe .\so101_simulation\run_perception.py --headless
Start-Process .\so101_simulation\perception_results\index.html
```

In playing mode, click source and destination in the occupancy window and press
Enter, or type a UCI move. Append `q`, `r`, `b` or `n` for promotion. The virtual
human action is placed instantly; robot replies are animated by default. Camera
verification of the board and inventories must succeed before a turn is committed.

`run_simulation.py` accepts `--moves`, `--speed`, `--no-viewer`, `--fps 30|60` and
`--low-graphics`. It does not offer interactive move input or a `--demo` option,
and it does not run the camera-perception game loop. `play_chess.py --demo` is the
visual game replay mode.

Physics uses a 2 ms timestep (500 Hz). Viewer synchronization is batched at an
approximately 60 Hz target and accounts for computation time when pacing motion.
Use `--fps 30 --low-graphics` with the motion runner for lighter viewer settings.
Those flags do not change physics or offscreen camera rendering and are not
accepted by the other runners.

## How the pieces fit together

| File | Responsibility |
| --- | --- |
| `chess_world.xml` | Base robot, board, pieces, actuators and cameras |
| `chess_coords.py` | Squares, grasp/rest heights, HOME/PARK poses and storage coordinates |
| `ik_solver.py` | Damped position IK and a downward gripper preference |
| `motion_planner.py` | Cosine joint interpolation, simulation stepping and viewer pacing |
| `chess_robot.py` | Approach, grasp/transfer, placement, capture and collision-mask handling |
| `perception_sim.py` | Generate camera fixtures, render RGB, arrange virtual actions and execute special-move transfers |
| `chess_game.py` | Rules adapter, legal occupancy matching, versioned coordinator and temporary search |
| `play_chess.py` | Human input, opponent turns and final camera verification |
| `run_perception.py` | Controlled rendered cases, scoring and HTML/JSON evidence |
| `test_perception.py` | Observation, rules, inventory, geometry and commit contracts |
| `test_system.py` | Original stability, IK and scripted-move checks |
| `publish_camera.py` | Prepared optional ROS Image/CameraInfo source |

The shared RGB-only detector lives in
[`perception.py`](../perception_ws/src/ai_gambit_perception/ai_gambit_perception/perception.py).
Its [workspace configuration](../perception_ws/src/ai_gambit_perception/ai_gambit_perception/workspace.json)
defines camera calibration settings, tag locations, detection thresholds and spare
type assignments. The [perception guide](PERCEPTION.md) explains the image stages,
coordinates, controls, ROS messages/services and real-camera setup in more detail.

The generated perception scene extends the base XML with 12 AprilTags, 32 promotion
slots and 48 capture/retirement slots. Spare pieces use cylinder proxies. The
perception camera renders 1280 x 960 RGB and the rectified board is 800 x 800.
This is a different camera setup from the original motion scene.

## Validation and limits

The implementation's Windows validation passed all 56 expected rendered outcomes,
34 perception contract/geometry tests, the original motion checks and GUI/scripted
game checks. A passing obstruction case means the system blocked the uncertain
observation. The [main README's validation section](../README.md#recorded-implementation-validation)
explains the scope and recorded measurements.

Run contracts from the repository root:

```powershell
.\.venv\Scripts\python.exe -B -m pytest .\so101_simulation\test_perception.py -q
```

Run the older motion checks from this folder, because they load relative XML paths:

```powershell
..\.venv\Scripts\python.exe -B test_system.py
```

Selected-case runs replace the gallery summary with that subset. Run without
`--case` to regenerate the full 56-case report. Generated assets and results are
ignored by Git and are recreated locally.

Pieces are carried kinematically and placement is imposed in the simulation;
these results do not prove realistic frictional grasping or full-arm collision
avoidance. Binary occupancy cannot prove piece identity or detect every swap.
Real-camera calibration, hardware limits, physical grasping, the custom engine
and full ROS movement integration remain separate validation/integration work.
