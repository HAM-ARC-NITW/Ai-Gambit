# Connect 4 perception in MuJoCo

The supplied models now have a fixed front RGB camera, independent board/supply
localization, colour detection, settling checks and verification before the
confirmed game state changes. The detector never reads MuJoCo positions; tests
use them separately to score its output.

**The arm is parked.** Releases are scripted above a column; MuJoCo gravity and
contacts determine landing. Connect 4 arm trajectories, grasps, MoveIt and hardware
control remain pending. The optional opponent is a temporary win/block/centre
heuristic.

## Run on Windows

From the repository root, using the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r connect_4_perception/requirements.txt
.\connect_4_perception\run_perception.cmd
```

This opens the MuJoCo 3D viewer, front RGB and annotated perception windows.
Focus a camera window: **Space/N** advances, **Esc** closes. Drag/scroll the 3D
viewer to change its view. The console explains whether each move was committed.
For a shorter automatic demonstration:

```powershell
.\connect_4_perception\run_perception.cmd --demo --case drop_r0_c3 hand_on_board robot_wrong_column jammed_at_top rgb_depth_limit
```

Keyboard-controlled virtual play:

```powershell
.\connect_4_perception\run_perception.cmd --play
.\connect_4_perception\run_perception.cmd --play --human-colour yellow
```

Focus a camera window; **1–7** chooses a human column, **R** reloads the empty
fixture and simulated supply. Red starts. Perception verifies both players'
releases. This mode demonstrates the game/perception loop; it does not move the arm.

`run_simulation.cmd` still opens the original environment-only scene. The folder
is now `connect_4_perception`, so older `connect_4` commands need updating. Asset
paths resolve beside the scripts, independently of the working directory.

## Reports and tests

```powershell
.\connect_4_perception\run_perception.cmd --headless
Start-Process .\connect_4_perception\perception_results\index.html
.\.venv\Scripts\python.exe -m pytest connect_4_perception/test_perception.py -q
```

The gallery contains **98 scenarios: 97 checked outcomes and one RGB limitation**.
Each saves before/after RGB, annotated output and JSON with independent ground
truth, confidences, poses and commit decisions. Correctly blocking an uncertain
move counts as success. **LIMIT** reproduces an observability failure; it does
not mean that the fault was detected.

Coverage includes all 42 gravity landing cells; either human colour; free human
column choice; wrong columns/colours; removed/recoloured/multiple discs; failed
pickup/lost discs/supply mismatch; full columns and missing supply; board/supply
obstruction; missing/bad tags; shifted board/camera; mirrored/blurred/dark/noisy
images and exposure changes; falling/jammed/off-centre discs; stale, duplicate,
out-of-order and future frames; camera loss; execution inhibition; gravity/count
violations; win/draw/turn order; cancellation; fresh, incomplete and cached resets.
Unit tests cover atomic commit, version/replay rules and rendered camera checks.

Use `--case NAME ...` for a subset, `--output PATH` for a separate report, and
`--seconds 15` to bound a visual run. Reports overwrite matching previous case
names. Generated assets/results are gitignored.

## Perception → engine coordinates

The engine receives one event after a move passes verification, not on every
camera frame. Coordinates are **zero-based**: bottom-left is `row=0,column=0`;
top-right is `row=5,column=6`. Colour is red=1 or yellow=2. For example, a red
drop in the fourth column at the bottom produces:

```json
{"row":0,"column":3,"index":3,"colour":1,"operation_id":"unique-id","game_version":1,"observation_id":12,"timestamp":1793100000.0}
```

In a Python engine adapter, consume the event after the existing move operation
finishes (the camera/coordinator loop still performs `begin` and `finish`):

```python
move = game.take_move()  # game is the existing Coordinator instance
if move is not None:
    engine_board[move["row"]][move["column"]] = move["colour"]
    # Or call your engine's move-input method here. Most Connect 4 engines
    # only need move["column"], since their rules calculate the landing row.
```

Each queued move can be consumed once. Occlusions, falling discs, invalid changes,
failed verification and replay produce no event. Reset clears queued old-game
moves and must also explicitly reset the engine's board. The keyboard game now
demonstrates this connection: the placeholder engine maintains its own board
from these coordinate events. It prints them and writes
`perception_results/engine_moves.jsonl`; starting/resetting that game clears the log.
Each scenario's `result.json` also includes `engine_move`, or `null` if no move
was accepted. Two changed cells are treated as an error, not two engine moves.

For a ROS engine, subscribe to **`/connect4/verified_move`** (`std_msgs/String`)
and parse the same JSON object. It is published only after a successful `verify`
action against a matching baseline; raw observations alone do not publish moves.
The engine/coordinator must still arrange the baseline and verification requests.
Use `operation_id` to ignore duplicates and `game_version` to require consecutive
board versions. The topic is a live notification, not durable storage: on a missed
version or reconnect, resynchronize the board before accepting more moves. A reset
is a separate engine reset, not a coordinate event. There is no Connect 4 engine
API in this checkout; `engine_stuff/engine.py` belongs to block stacking. Replace
the small demonstrated consumer with the real engine's input method when available.

## Simple code map

| File | Responsibility |
| --- | --- |
| `perception_sim.py` | Tagged scene, RGB rendering, scripted releases and test-only ground truth. |
| `run_perception.py` | Scenarios, windows, gallery and virtual game. |
| `test_perception.py` | Rules, verification and rendered tests. |
| `../perception_ws/src/ai_gambit_perception/ai_gambit_perception/connect4.py` | Shared detector and separate coordinator; no MuJoCo imports. |
| `../perception_ws/src/ai_gambit_perception/ai_gambit_perception/connect4.json` | Geometry, calibration and thresholds. |
| `publish_camera.py`, `check_ros.py`, `connect4_node.py` | Optional ROS camera, integration test and perception node; the node is in the shared ROS package. |

OpenCV AprilTag 36h11 corners localize the board and feeder independently.
Each fixture requires three spatially distributed tags: one hidden tag is
tolerated; two hidden tags block verification. Homography rectifies the image;
PnP and camera TF provide fixture poses. The empty reference is captured explicitly
before loading discs. Visible hole regions are calibrated from it to account for
the thick, tapered mesh. HSV classifies red/yellow; reference comparison detects
empty. Ambiguous regions become UNKNOWN. Unexpected foreground material outside
allowed apertures invalidates the affected fixture. Confidence is pixel agreement,
not a calibrated probability.

The front camera is 1280×800. Defaults require three stable fresh frames and
0.15 seconds of settling, 0.82 confidence, reprojection error below 2 pixels and
fixture movement within 4 mm/3°. Motion, occlusion, poor sharpness, stale frames
and execution inhibit acceptance. These are simulation settings, not measured
hardware tolerances. Real cameras need measured K, distortion and camera/base TF.

Rows run **bottom to top**, columns **left to right**, index=`row*7+column`.
Values: empty=0, red=1, yellow=2, unknown=255. Board origin is the bottom-left
cell centre; X right, Y up, Z outward. The independently tagged feeder has 21
red and 21 yellow slots.

The coordinator records a fresh matching board/supply baseline, operation ID and
game version. After completion it requires a newer settled observation, compares
every board cell and the exact supply change, checks gravity/turn order, then
commits. Humans may choose any legal column; robot moves must match their intended
column. Failures/cancellation pause recovery and leave confirmed state unchanged.
Operation IDs cannot be replayed even after reset. Reset requires a fresh empty
board; it does not prove a physical reset latch is locked or supply replenished.

## Simulation assumptions and remaining limits

Original STL files are unchanged. The original token diameter is about 15 mm;
board row pitch is about 19.4 mm. Generated discs use 19.4 mm diameter and retain
11 mm thickness so stacks align with openings. The board is raised clear of the
table. Since convex mesh collision seals the board holes, analytical guides and
a bottom stop provide collision geometry. Discs have three translation joints
and fixed upright orientation: they fall and contact neighbours but cannot tip.
The feeder is a synthetic support fixture.

A matte backing makes empty holes consistent. `rgb_depth_limit` places red
material between board and backing: an empty hole then resembles a seated red
disc. The gallery intentionally exposes this false occupancy. Front RGB cannot
establish depth, hidden engagement, a locked reset slider or every jam. Additional
views/depth or independent sensors are needed where those facts matter. A box
representing a hand tests occlusion; it is not a human safety system. Upright
guides do not validate real wobble, grasp reliability or gripper contact.

Deployment needs real calibration, an operating-condition empty reference,
measured colour thresholds, hardware execution/recovery and physical testing.
These are broad synthetic cases, not all possible physical edge cases.

## Optional ROS 2 Humble in WSL Ubuntu 22.04

Windows mode needs no ROS. The ROS path uses apt Humble at `/opt/ros/humble`
and a **Linux Python 3.10** environment. Reuse your existing compatible Linux
environment; do not activate the Windows `.venv` in WSL. After installing Humble,
a suitable environment can be prepared with:

```bash
sudo apt install ros-humble-cv-bridge ros-humble-tf2-ros python3-colcon-common-extensions python3-venv
python3 -m venv --system-site-packages ~/.local/share/ai-gambit/venv
source ~/.local/share/ai-gambit/venv/bin/activate
pip install 'numpy>=1.26,<2' 'opencv-python>=4.8,<4.12' 'mujoco>=3.2,<4'
cd /mnt/c/Users/DEVELOPMENT/Desktop/coding_projects/Ai-Gambit
bash ros_tools/run_connect4_ros.sh build
bash ros_tools/run_connect4_ros.sh run
```

In another WSL terminal at the repository root:

```bash
bash ros_tools/run_connect4_ros.sh test
```

The actual ROS integration test checks images, TF, baseline service, verify/reset
actions, version/replay rejection, wrong column, cancellation, execution inhibition,
camera watchdog and recovery. Screenshots/results go in `perception_results/ros`.
Ctrl+C stops the stack's own two nodes, leaving other ROS applications alone.

Linux build output lives in `~/.local/share/ai-gambit-connect4`. Override the
venv using `AI_GAMBIT_ROS_VENV`. ROS domain defaults to **52**; both terminals must
use the same domain. WSL allows 2-second frame age for software rendering; native
mode uses 1 second. Use wall-clock ROS time (`use_sim_time=false`) and synchronized
camera exposure timestamps.

| Interface | Purpose |
| --- | --- |
| `/connect4/camera/image_raw`, `/connect4/camera/camera_info` | RGB and matching exposure-stamped calibration. |
| `/connect4/board_observation` | Typed board/supply cells, confidence and validity. |
| `/connect4/verified_move` | JSON coordinate event for the engine after successful move verification. |
| `/connect4/perception_status`, `/connect4/debug_image` | Watchdog/status and annotated output. |
| `/connect4/execution_status` | Inhibit acceptance during execution, pause or recovery. |
| `/connect4/set_baseline` | Record expected board/supply, operation ID and version. |
| `/connect4/observe` | Cancellable `snapshot`, `verify`, `reset` action with freshness boundary and timeout. |
| TF | Camera extrinsics and measured board/feeder poses in `base_link`. |

`/connect4/simulation/command` and `/connect4/simulation/status` are JSON channels
for test-fixture control. Acknowledgement means release was applied, not that
a grasp or settling succeeded. A production coordinator must set a baseline,
execute motion, request fresh verification after completion, then advance its
logical board only on success. This adds perception transport, not a ROS game
orchestrator or Connect 4 MoveIt controller.
