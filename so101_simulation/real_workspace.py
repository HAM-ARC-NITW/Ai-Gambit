"""The real board through the real camera; MuJoCo only plans the arm.

RealWorkspace keeps the Simulation interface the game uses (frame, snapshot,
physical_move, sync_mapping), so the Coordinator, play_turn and every camera
check run unchanged. What differs from the simulation:
- frame() comes from the LeRobot camera, not the renderer;
- the detector uses the measured calibration and empty-fixture photo saved in
  real_setup.json by real_setup.py;
- the twin's arm starts wherever the real arm is, and every planned pose is
  streamed to the real arm (see lerobot_arm.py).
"""
import json
from pathlib import Path
import cv2
import numpy as np
import mujoco
from perception_sim import Simulation, HERE
from chess_coords import CLEARANCE_HEIGHT_Z
from ai_gambit_perception.perception import Perception
from lerobot_arm import ArmError, LeRobotArm, make_so101

SETUP = HERE / "real_setup.json"


def load_setup(path=SETUP):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} is missing. Create it with: python real_setup.py init --port <PORT>")
    return json.loads(path.read_text())


def save_setup(setup, path=SETUP):
    Path(path).write_text(json.dumps(setup, indent=2))


def calibration_from_setup(setup):
    """Measured camera calibration, as RealWorkspace expects it."""
    missing = [k for k in ("camera_matrix", "distortion", "base_camera_optical", "empty_reference")
               if setup.get(k) is None]
    if missing:
        raise ValueError("real_setup.json lacks " + ", ".join(missing) +
                         ". Run the real_setup.py steps intrinsics, extrinsics and empty first.")
    image = cv2.imread(str(HERE / setup["empty_reference"]))
    if image is None:
        raise FileNotFoundError("Empty-fixture photo not found: " + setup["empty_reference"])
    return dict(camera_matrix=setup["camera_matrix"], distortion=setup["distortion"],
                base_camera_optical=setup["base_camera_optical"],
                empty_reference=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def connect_arm(setup):
    """The real SO-101 and overhead camera, connected through LeRobot."""
    camera = setup.get("camera", {})
    robot = make_so101(setup["port"], setup.get("robot_id", "chess_arm"),
                       camera.get("index", 0), camera.get("width", 1280),
                       camera.get("height", 720), camera.get("fps", 30),
                       release_torque=setup.get("release_torque_on_exit", False))
    arm = LeRobotArm(robot, setup.get("arm"))
    arm.connect()
    return arm


class RealWorkspace(Simulation):
    def __init__(self, arm, calibration=None):
        """Without a calibration (during setup) the arm can move, but the
        camera's board readings are meaningless."""
        self.live = False  # The base class renders its own scene while building.
        super().__init__()
        self.arm = arm
        if calibration is not None:
            self.detector = Perception(self.config, calibration["camera_matrix"], calibration["distortion"],
                                       calibration["base_camera_optical"])
            self.detector.set_empty_reference(np.asarray(calibration["empty_reference"]))
        arm.attach(self.model)
        self.robot.planner.arm = arm
        self.live = True
        self.match_arm()

    def frame(self):
        return self.arm.frame() if self.live else super().frame()

    def match_arm(self):
        """Put the twin's arm where the real arm is; every plan starts from there."""
        q = self.arm.read_q()
        self.data.qpos[:6], self.data.ctrl[:6], self.data.qvel[:6] = q, q, 0
        mujoco.mj_forward(self.model, self.data)
        self.arm.start()

    def reset(self):
        super().reset()
        if self.live:
            self.match_arm()

    def physical_move(self, board, move, animate=False, viewer=None, speed=1, observation=None):
        if not animate:
            raise ArmError("The real arm cannot teleport pieces: arm moves must be animated")
        self.match_arm()
        return super().physical_move(board, move, animate, viewer, speed, observation)

    def sync_mapping(self, board):
        """After a human move the twin's pieces follow the confirmed board. Grasp
        targets come from the camera; twin pieces only keep collision rules."""
        rack, tray = list(self.rack_inventory), list(self.tray_inventory)
        self.load_position(board)
        self.rack_inventory, self.tray_inventory = rack, tray

    def park(self, viewer=None, instant=False):
        if instant and self.live:
            raise ArmError("The real arm cannot be parked instantly")
        super().park(viewer, instant)

    def retreat(self):
        """After a stopped move the jaws may be down among the pieces: lift
        straight up to clearance height first, then swing clear to park."""
        self.match_arm()
        planner = self.robot.planner
        q = self.data.qpos[:6].copy()
        tip = self.data.site_xpos[self.robot.site_id].copy()
        if tip[2] < CLEARANCE_HEIGHT_Z - .01:
            for waypoint in self.robot._vertical_waypoints(tip[:2], tip[2], CLEARANCE_HEIGHT_Z, 5, q):
                waypoint[5] = q[5]  # Keep the jaws as they are.
                planner.execute_segment(q, waypoint, steps=60)
                q = waypoint
        planner.carrying_piece_id = None
        self.park()

    def close(self):
        """Park clear of the board before torque is released, then disconnect."""
        try:
            if self.live:
                self.retreat()
        finally:
            self.live = False
            self.arm.disconnect()
            super().close()
