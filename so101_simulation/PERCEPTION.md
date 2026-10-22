# Camera perception and playing chess

This implementation replaces the dummy-image ROS prototype with an actual RGB
pipeline. The same detector is called directly on Windows and by the ROS node.
Windows simulation does not need ROS installed.

**Current integration status:** the working game and movement pipeline use direct
Python/MuJoCo calls. Full ROS 2 movement/game integration is still pending. The
prepared ROS perception bridge has not been build/runtime-validated in Humble.
MoveIt 2 and ros2_control have not been connected.

**The current opponent is a temporary placeholder.** python-chess owns the rules,
and a small local search chooses moves until the team's custom chess engine is
integrated. See the [main README](../README.md) for the complete implementation
overview, exact scenario list, recorded validation and remaining work.

## Start on Windows

From the repository root, using PowerShell:

```powershell
# First-time setup only. The existing project environment already has these.
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\so101_simulation\requirements.txt

# Play White against the arm.
.\.venv\Scripts\python.exe .\so101_simulation\play_chess.py

# Step through the before/after camera experiments.
.\.venv\Scripts\python.exe .\so101_simulation\run_perception.py

# Run all experiments automatically and generate the HTML gallery.
.\.venv\Scripts\python.exe .\so101_simulation\run_perception.py --headless
Start-Process .\so101_simulation\perception_results\index.html
```

You can also double-click `play_chess.cmd` or `run_perception.cmd`. Both use the
project's `.venv`; they do not depend on your editor's working directory.

### Playing

Three windows open: the MuJoCo 3D scene, the overhead camera, and the measured
occupancy grid. You play White; the arm plays Black. In the grid window:

1. Click the source square, then the destination square.
2. Press Enter. You can instead type a move such as `e2e4` and press Enter.
3. Your virtual hand moves the pieces, including capture removal and rook movement
   during castling. Promotion requires an explicit `q`, `r`, `b` or `n` suffix.
4. The camera observes several fresh frames. The rules component compares their
   occupancy to **all** legal outcomes. Your typed/clicked move is not itself the
   perception result. If several captures or promotions look identical in binary
   occupancy, the explicit destination/promotion selection disambiguates them.
5. The arm chooses and animates a reply. Camera verification must confirm the
   final board, promotion rack and capture tray before the turn is committed.

Backspace edits input. Escape exits. When input is empty, Q exits and R reloads
the confirmed FEN with fresh inventory after a mismatch. Reloading is an explicit
simulation reset; it does not attempt to reconstruct an uncertain physical game.
An illegal input leaves the game unchanged. A failed observation pauses play.

The human action uses instant piece placement as a virtual hand. Robot replies
use the existing arm controller by default. `--no-animation` uses instant robot
placement too, which is useful for quick perception experiments.

```powershell
# Watch a short game without entering moves.
.\.venv\Scripts\python.exe .\so101_simulation\play_chess.py --demo

# Scripted end-to-end camera/arm verification without windows.
.\.venv\Scripts\python.exe .\so101_simulation\play_chess.py --headless --moves e2e4 g1f3

# Start from another valid position with White to move.
.\.venv\Scripts\python.exe .\so101_simulation\play_chess.py --fen "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
```

The temporary opponent uses negamax with alpha-beta pruning, material values and
a central-square bonus. python-chess supplies legal moves and the game position.
`--depth 1`, `2` or `3` controls its search; the default is two plies (an arm move
and a possible human reply). Replace `choose_move()` in `chess_game.py` to connect
the team's custom engine while preserving camera verification before commits.
No internet connection or external engine executable is needed during play.

## Visually inspect the experiments

Space or N advances from BEFORE to AFTER, then to the next experiment. Escape
stops. To inspect one case, pass `--case`, for example:

```powershell
.\.venv\Scripts\python.exe .\so101_simulation\run_perception.py --case capture partial_castle hand_on_board
```

Each case saves `before.png`, `after.png`, `occupancy.png` and `result.json`.
`perception_results/index.html` puts the images and scores side by side.
`summary.json` contains machine-readable outcomes. Green outlines mean occupied,
grey empty, and orange unknown. Confidence numbers are heuristic detector quality
scores, not statistically calibrated probabilities.

The top-level gallery/summary reflects only the selected run. Running with
`--case` replaces it with that subset; omit `--case` to regenerate all 56 cases.
Generated images, XML and results are ignored by Git and may be absent in a clone.

A passing obstruction case means the system **refused** the observation as
specified. It does not mean an occluded square was correctly guessed. The report
therefore shows known-square coverage and correctness separately. A READY image
must classify all 64 squares correctly against independent simulator ground truth.
Simulator state is used by the test scorer and scene arranger, never by the RGB
detector. The report also measures fixture-corner error and approximate visual
piece-centre error where the observation is READY.

The suite covers:

| Area | Rendered experiments |
| --- | --- |
| Legal outcomes | Pawn and knight moves; ambiguous captures; White and Black castling; White and Black en passant; Q/R/B/N promotion; capture promotion |
| Incomplete actions | Partial castling/en passant; failed pickup; wrong square; extra accidental move; missing capture storage; incorrect promotion inventory |
| Physical placement | Small/large offsets; piece between squares; toppled piece; hand, arm, foreign object and object in the workspace gap |
| Calibration | One hidden tag; one inconsistent tag; all board/rack/tray tags hidden; shifted board; shifted camera |
| Sensor/temporal failures | Moderate exposure change; noise; severe blur; disconnected camera; stale/out-of-order frames; robot still executing |
| State/inventory | Missing rack piece; touch-and-return; explicit identity-swap observability limit; an actual animated arm move |

Automated contract and geometry tests supplement those images:

```powershell
.\.venv\Scripts\python.exe -B -m pytest .\so101_simulation\test_perception.py -q
cd so101_simulation
..\.venv\Scripts\python.exe -B test_system.py
```

Do not run unqualified `pytest` over all the older diagnostic scripts; several
are interactive utilities rather than isolated tests.

## Small code map

| File | Responsibility |
| --- | --- |
| `perception_ws/src/ai_gambit_perception/ai_gambit_perception/perception.py` | RGB-only calibration/localization, rectification, occupancy, confidence, temporal stability, baselines and verification |
| `.../workspace.json` | Camera, marker, fixture and threshold settings; lengths are metres |
| `.../perception_node.py` | ROS transport, synchronized Image/CameraInfo, watchdog, services and TF2 |
| `perception_sim.py` | Extend the original robot XML with tags/rack/trays; render camera frames; arrange physical actions for tests and virtual-human input |
| `chess_game.py` | Legal candidate matching, simple opponent and versioned game-state commits |
| `play_chess.py` | Click/type UI and observe → interpret → execute → verify turn loop |
| `run_perception.py` | Named, data-driven edge cases, independent scoring and visual gallery |
| `test_perception.py` | Rules, timing, atomic commits and rendered camera/geometry contracts |

There is one core detector, not a different algorithm for each test or transport.
Comments explain the geometry and the reasons for the conservative failure paths.

## How the detector works

1. Undistort the RGB image using camera intrinsics and distortion coefficients.
2. Detect AprilTag 36h11 markers with OpenCV. Marker IDs and physical dimensions
   come from configuration, not simulator body coordinates.
3. Fit each fixture independently using at least two visible tags, RANSAC and
   solvePnP. Reject excessive reprojection error and unexpected fixture motion.
4. Compute `T_base_fixture = T_base_camera_optical * T_camera_optical_fixture`.
5. Rectify the board to 800×800 pixels and inspect the rack, trays and workspace
   apron against a deliberately acquired empty reference. Robust background
   normalization handles moderate global exposure changes.
6. Classify compact foreground near square/slot centres. Large or displaced
   components, unexpected objects and uncertain measurements block acceptance.
7. Require multiple consistent fresh frames. Moving/occluded/robot-intermediate
   observations cannot be committed. Camera failures clear temporal stability.
8. Compare against an engine-supplied expected occupancy and inventory. A fresh
   observation ID, timestamp, operation boundary and baseline version are required.

Board arrays are always `a1,b1,...h1,a2,...h8`: `index = rank*8 + file`.
The rectified image shows rank 8 at the top. Board-frame X goes a→h, Y goes 1→8,
and Z points upward. In this simulation the board origin is at base coordinates
`[0.13, 0.12, 0.01]`; board X points along base -Y and board Y along base +X.

The rack contains 32 labelled slots, ordered White row then Black row. Within
each row, configured types repeat Q,R,B,N four times. Spare pieces are compact
cylinder **proxies**; their types are inventory assignments, not visually inferred
identities. Edit `rack_types` in workspace.json to change which types are loaded
(for example, to reserve more queens). A requested type with no available spare
pauses the action. The tray has 48 observed slots: 24 per colour, three rows of eight.
This leaves room for captured pieces and retired promotion pawns. Black storage
uses the positive base-Y side; White uses negative base-Y.

The generated XML, actual marker PNGs and empty reference photograph are in
`perception_assets/`. They are regenerated from the original `chess_world.xml`
and configuration; the original robot model is not duplicated by hand.

## ROS 2 Humble interface

**Prepared perception transport; full ROS integration pending.** The commands
below are for building/testing that optional transport. They do not launch a
ROS-controlled chess game. ROS movement actions/nodes, joint-state publishing,
coordinator/engine connections, MoveIt 2 and ros2_control integration still need
implementation and end-to-end testing.

Native Windows simulation uses direct Python calls. The optional ROS path needs
a working Ubuntu 22.04 / ROS 2 Humble environment. It has not been runtime-validated
by the native Windows tests. The old `package.xaml` typo and missing ament resource
have been repaired, and a separate interface package supplies typed messages.

In Ubuntu, after installing ROS Humble, colcon, cv_bridge, TF2 and the standard
message packages, use an environment compatible with Humble's NumPy-based bridge:

```bash
source /opt/ros/humble/setup.bash
python3 -m venv --system-site-packages .ros-venv
source .ros-venv/bin/activate
pip install 'numpy<2' 'opencv-python>=4.8,<4.12' 'mujoco>=3.2,<4' python-chess
colcon build --base-paths perception_ws/src --symlink-install \
  --build-base perception_ws/build --install-base perception_ws/install
source perception_ws/install/setup.bash

# Terminal 1: creates the fixture/reference files and publishes camera data.
python so101_simulation/publish_camera.py

# Terminal 2: source the same ROS/environment/install setup first.
ros2 run ai_gambit_perception perception_node --ros-args \
  -p empty_reference:="$(pwd)/so101_simulation/perception_assets/empty_workspace.png"
```

Use OpenCV ≥4.8: Ubuntu's older default OpenCV may not provide ArucoDetector.
Use a matching OpenCV/NumPy/cv_bridge build; the example pins avoid Humble's
NumPy-1 binary bridge being mixed with NumPy 2.

| Interface | Type and behaviour |
| --- | --- |
| `/camera/image_raw` | `sensor_msgs/Image`, RGB/BGR converted through cv_bridge; sensor-data QoS |
| `/camera/camera_info` | `sensor_msgs/CameraInfo`, dimensions, optical frame and timestamp checked against the image |
| `/coordinator/state` | `std_msgs/String`; ROBOT_EXECUTING, PAUSED or RECOVERING inhibits acceptance |
| `/perception/board_observation` | `ai_gambit_interfaces/BoardObservation`: timestamp, sequence, pose validity, stable flag, status, 64 occupancy/confidence/offset entries |
| `/perception/rack_observation` | `RackObservation`: 32 promotion and 48 capture occupancy/confidence entries, validity |
| `/perception/status` | `std_msgs/String`: INITIALIZING, MOVING, READY, OCCLUDED, CALIBRATION_INVALID, CAMERA_FAILURE, ROBOT_EXECUTING or BOARD_MISMATCH |
| `/perception/snapshot` | `Snapshot`: operation ID, version, after-time and bounded timeout; waits for a new stable frame, not a cached result |
| `/perception/set_baseline` | `SetBaseline`: version and expected board/rack/tray; refuses an inconsistent initial scene |
| `/perception/verify` | `VerifyState`: fresh snapshot plus expected board/rack/tray, operation ID, version and after-time |
| TF2 | This node publishes base_link→board_frame, promotion_rack_frame and capture_tray_frame. The camera publisher owns base_link→camera_link→camera_optical_frame. |

AprilTags are detected inside the core with OpenCV; a second apriltag_ros detector
is not required. Do not publish the same authoritative fixture frames from two
nodes. The ROS simulated camera publishes at 5 Hz; processing drops overlapping
callbacks rather than queuing stale frames. Services use a multi-thread executor
so new camera frames can arrive while a request waits.

### Real-camera calibration

Before hardware use, photograph a calibration target at several positions and
angles and obtain K/distortion with OpenCV or ROS camera_calibration. Publish the
measured CameraInfo. Measure the static camera-to-base transform and supply its
4×4 `T_base_camera_optical` matrix in `base_camera_optical` in a copied workspace
JSON. This matrix maps camera-optical coordinates into `base_link` coordinates.
Set the real fixture origins, marker IDs/sizes and dimensions in that file; pass
it with `-p workspace_config:=/absolute/path/workspace.json`. Acquire an independent
empty-workspace photograph from that camera and pass it as `empty_reference`.
The supplied camera defaults describe the simulator only.

Validate physical board corners and centre/edge square targets using measured
robot/tool reference points. Choose thresholds from actual base diameters,
gripper clearance and repeatability. A shifted board/camera is rejected in this
fixed-fixture implementation; validate and deliberately recalibrate it before
resuming. It does not silently move the planning scene during an active action.

## Practical limits and extension points

Binary occupancy cannot prove piece identity, distinguish a wrong type loaded in
a labelled rack, or detect an identity swap that preserves occupancy. The gallery
includes this limitation explicitly. Captures use logical identities, control
results and storage occupancy together. Promotion type comes from an explicit
selection and the known rack inventory. No twelve-class recognizer is claimed.

Offsets are approximate foreground centroids from overhead RGB. They are not
precision 3D base estimates. Excessively displaced/uncertain pieces are rejected;
measured transforms and offsets feed the simulated arm's approach targets.
The heuristic obstruction detector is validated on these controlled scenarios,
not every possible real hand, material, lighting condition or fallen-piece pose.
It supplies observations and inhibits new actions, not a certified protective stop.

The existing controller still isolates contacts, carries a selected piece
kinematically and snaps placement to its target. Passing visual verification
demonstrates the camera/coordinator flow, not a validated frictional grasp.
Unreachable targets are rejected before transfer. ROS control, a MoveIt planning
scene and hardware-level stopping remain Control/Integration responsibilities;
they are not silently emulated by the perception node.

Reference APIs: [python-chess rules](https://python-chess.readthedocs.io/en/latest/core.html)
and [OpenCV marker detection](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html).
