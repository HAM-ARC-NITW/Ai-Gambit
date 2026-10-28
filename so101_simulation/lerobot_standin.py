"""A hardware-free stand-in for LeRobot's SO-101 follower and its camera.

A second MuJoCo world plays the physical table. It accepts LeRobot actions
(degrees and gripper percent), poses its arm, and its camera renders the
"real" board. A piece moves only if the jaws close around it, so an arm whose
units or calibration are wrong really does miss, and the game must notice.

This tests the LeRobot interface and everything above it. It is not a model
of real grasp physics, motor lag or a real camera's image quality.
"""
import cv2
import numpy as np
import mujoco
from perception_sim import Simulation, HERE
from ai_gambit_perception.perception import camera_calibration
from lerobot_arm import CAMERA, LeRobotArm

CLOSED_BELOW = 31.  # Gripper percent below which the jaws are closing on a piece.
STALL_PCT = 34.     # Where the jaws stop when a piece is between them.


class StandInSO101:
    """Implements the parts of lerobot.robots.Robot the chess game uses."""

    def __init__(self, world=None, true_settings=None, stuck=False):
        self.world = world or Simulation(button=True)
        # How the "physical" arm really maps LeRobot units; tests make it disagree
        # with the assumed calibration to check that the game notices.
        self.units = LeRobotArm(None, true_settings)
        self.stuck = stuck
        self.is_connected = False
        self.site = self.world.robot.site_id
        self.held, self.offset, self.rest_z, self.closed = None, None, None, False
        self.command = self.units.to_action(self.world.data.qpos[:6])
        self.image, self.scene = None, None

    def connect(self, calibrate=True):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def _check(self):
        if not self.is_connected:
            raise ConnectionError("StandInSO101 is not connected")

    def get_observation(self):
        self._check()
        measured = self.units.to_action(self.world.data.qpos[:6])
        measured["gripper.pos"] = self.command["gripper.pos"]
        if self.held is not None:
            measured["gripper.pos"] = max(self.command["gripper.pos"], STALL_PCT)
        # Render only when the table changed: by the arm, or by a person's hand.
        scene = self.world.data.qpos.tobytes()
        if scene != self.scene:
            self.image, self.scene = self.world.frame(), scene
        return {**measured, CAMERA: self.image}

    def send_action(self, action):
        self._check()
        self.command = {k: float(v) for k, v in action.items()}
        if self.stuck:
            return self.command
        q = self.units.from_observation(self.command)
        self.world.data.qpos[:6] = q
        mujoco.mj_forward(self.world.model, self.world.data)
        closing = self.command["gripper.pos"] < CLOSED_BELOW
        if closing and not self.closed and self.held is None:
            self._grab()
        elif not closing and self.held is not None:
            self._release()
        self.closed = closing
        if self.held is not None:
            self._carry()
        return self.command

    def _grab(self):
        tip = self.world.data.site_xpos[self.site]
        if not .01 < tip[2] < .08:
            return
        near = [(np.hypot(*(self.world.data.xpos[b][:2] - tip[:2])), b) for b in self.world.piece_ids
                if self.world.data.xpos[b][2] > 0]
        distance, body = min(near)
        if distance < .006:
            self.held, self.rest_z = body, self.world.data.xpos[body][2]
            self.offset = self.world.data.xpos[body] - tip

    def _carry(self):
        tip = self.world.data.site_xpos[self.site]
        self.world.put(self.held, tip + self.offset)
        mujoco.mj_forward(self.world.model, self.world.data)

    def _release(self):
        position = self.world.data.xpos[self.held].copy()
        position[2] = self.rest_z
        self.world.put(self.held, position)
        mujoco.mj_forward(self.world.model, self.world.data)
        self.held = None


def standin_calibration(world):
    """The stand-in camera is the ideal simulated one, with its empty photo."""
    K, distortion, base_camera = camera_calibration(world.config)
    empty = cv2.cvtColor(cv2.imread(str(HERE / "perception_assets/empty_workspace.png")), cv2.COLOR_BGR2RGB)
    return dict(camera_matrix=K, distortion=distortion, base_camera_optical=base_camera, empty_reference=empty)
