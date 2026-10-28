"""Drive the real SO-101 through LeRobot, mirroring the simulated arm.

The MuJoCo arm stays the planner (a "digital twin"): IK, waypoints and
collision rules are unchanged. Every pose the twin is commanded is converted to
LeRobot units and streamed to the real arm at a real-time pace. Each segment
must physically arrive before the next starts, and a grasp must be felt by the
gripper motor, or the move stops and the game pauses.

This file does not import LeRobot; make_so101() does. Anything with LeRobot's
Robot interface (connect, get_observation, send_action, disconnect) works,
which is how the tests use a stand-in robot without hardware.
"""
import time
import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
CAMERA = "top"

# Defaults, overridden by the "arm" section of real_setup.json. The MuJoCo model
# uses LeRobot's "new calibration" (zero = middle of each joint's range), so
# only joint directions and small offsets need checking on a real arm.
DEFAULTS = {
    "sign": [1, 1, 1, 1, 1],          # Per arm joint: -1 if the real joint turns the other way.
    "offset_deg": [0, 0, 0, 0, 0],    # Real degrees when the twin joint is at zero.
    "gripper_rad": [-0.2, 1.8],       # Twin gripper angles that map to...
    "gripper_pct": [0, 100],          # ...LeRobot gripper percent (0 closed, 100 open).
    "slowdown": 4.0,                  # Real moves take this many times longer than simulated ones.
    "rate_hz": 30,                    # Commands per second sent to the motors.
    "realtime": True,                 # False only for tests: no sleeping between commands.
    "max_speed_deg_s": 60.0,          # Segments are stretched so no real joint moves faster.
    "max_step_deg": 8.0,              # A larger jump between commands is refused, never sent.
    "arrive_tol_deg": 4.0,            # Every segment must end this close to its target...
    "arrive_timeout": 4.0,            # ...within this many seconds.
    "grasp_margin_pct": 3.0,          # Jaws stopped this far above the command => a piece is held.
    "grasp_settle": 0.4,              # Seconds for the jaws to close before judging the grasp.
}


class ArmError(ValueError):
    """The real arm refused or failed a motion. The game pauses; nothing is committed."""


class GraspFailed(ArmError):
    pass


class LeRobotArm:
    def __init__(self, robot, settings=None, camera=CAMERA, clock=time.perf_counter, sleep=time.sleep):
        unknown = set(settings or {}) - set(DEFAULTS)
        if unknown:
            raise ValueError("Unknown arm settings: " + ", ".join(sorted(unknown)))
        self.robot, self.camera, self.clock, self.sleep = robot, camera, clock, sleep
        self.settings = {**DEFAULTS, **(settings or {})}
        s = self.settings
        self.sign = np.asarray(s["sign"], float)
        self.offset = np.asarray(s["offset_deg"], float)
        if self.sign.shape != (5,) or not np.all(np.abs(self.sign) == 1) or self.offset.shape != (5,):
            raise ValueError("Arm calibration needs five signs (+1/-1) and five offsets")
        if s["slowdown"] <= 0 or s["rate_hz"] <= 0:
            raise ValueError("slowdown and rate_hz must be positive")
        self.limits = None
        self.dt = .002
        self.last = None       # Degrees/percent last sent (or measured at start).
        self.pending = 0.      # Real seconds of motion not yet sent.
        self.deadline = None
        self.sent = 0

    # --- Units -----------------------------------------------------------------
    def attach(self, model):
        """Use the twin's joint limits and time step."""
        self.limits = np.array(model.jnt_range[:6], float)
        self.dt = float(model.opt.timestep)

    def to_action(self, q):
        q = np.asarray(q, float)
        degrees = np.degrees(q[:5]) * self.sign + self.offset
        s = self.settings
        gripper = float(np.interp(q[5], s["gripper_rad"], s["gripper_pct"]))
        return {**{f"{n}.pos": float(d) for n, d in zip(JOINTS, degrees)}, "gripper.pos": gripper}

    def from_observation(self, obs):
        degrees = np.array([obs[f"{n}.pos"] for n in JOINTS[:5]], float)
        q = np.radians((degrees - self.offset) * self.sign)
        s = self.settings
        gripper = float(np.interp(obs["gripper.pos"], s["gripper_pct"], s["gripper_rad"]))
        return np.append(q, gripper)

    @staticmethod
    def _vector(action):
        return np.array([action[f"{n}.pos"] for n in JOINTS], float)

    # --- Connection --------------------------------------------------------------
    def connect(self):
        self.robot.connect()

    def disconnect(self):
        if getattr(self.robot, "is_connected", True):
            self.robot.disconnect()

    def read_q(self):
        return self.from_observation(self.robot.get_observation())

    def frame(self):
        """RGB frame from the LeRobot camera called CAMERA."""
        image = self.robot.get_observation().get(self.camera)
        if image is None:
            raise ArmError(f"LeRobot robot has no camera named '{self.camera}'")
        return np.ascontiguousarray(image)

    def start(self):
        """The next command must continue from where the real arm is now."""
        self.last = self._vector(self.robot.get_observation())
        self.pending, self.deadline = 0., None

    # --- Motion ------------------------------------------------------------------
    def send(self, q):
        q = np.asarray(q, float)
        if self.limits is not None and np.any((q < self.limits[:, 0]) | (q > self.limits[:, 1])):
            raise ArmError(f"Refused: pose outside joint limits {np.round(q, 3)}")
        action = self.to_action(q)
        target = self._vector(action)
        if self.last is None:
            self.start()
        jump = np.max(np.abs(target[:5] - self.last[:5]))
        if jump > self.settings["max_step_deg"]:
            raise ArmError(f"Refused: {jump:.1f} degree jump between commands (limit "
                           f"{self.settings['max_step_deg']}). The plan does not start where the arm is.")
        self.robot.send_action(action)
        self.last = target
        self.sent += 1

    def steps_for(self, q_start, q_end, steps):
        """Stretch a segment so no real joint exceeds max_speed_deg_s. The
        planner's short segments are fine in simulation, but between two IK
        waypoints a joint can swing tens of degrees in a few milliseconds."""
        distance = np.degrees(np.max(np.abs(np.asarray(q_end)[:5] - np.asarray(q_start)[:5])))
        # The cosine profile peaks at pi/2 times the average speed.
        seconds = np.pi / 2 * distance / self.settings["max_speed_deg_s"]
        return max(steps, int(np.ceil(seconds / (self.dt * self.settings["slowdown"]))))

    def follow(self, q, final=False):
        """Called by the planner for every simulated step of a segment."""
        s = self.settings
        self.pending += self.dt * s["slowdown"]
        if self.pending < 1 / s["rate_hz"] and not final:
            return
        if s["realtime"]:
            now = self.clock()
            self.deadline = (now if self.deadline is None else self.deadline) + self.pending
            if self.deadline > now:
                self.sleep(self.deadline - now)
        self.pending = 0.
        self.send(q)
        if final:
            self.wait_until_reached(q)
            self.deadline = None

    def wait_until_reached(self, q):
        target = self._vector(self.to_action(q))
        start = self.clock()
        while True:
            error = np.max(np.abs(self._vector(self.robot.get_observation())[:5] - target[:5]))
            if error <= self.settings["arrive_tol_deg"]:
                return
            if self.clock() - start > self.settings["arrive_timeout"]:
                raise ArmError(f"Arm did not reach its target: still {error:.1f} degrees away. "
                               "Check for an obstruction, then the joint calibration.")
            self.sleep(.02)

    def confirm_grasp(self):
        """A piece between the jaws stalls the gripper above its command."""
        if self.settings["realtime"]:
            self.sleep(self.settings["grasp_settle"])
        measured = self.robot.get_observation()["gripper.pos"]
        command = self.last[5]
        if measured < command + self.settings["grasp_margin_pct"]:
            raise GraspFailed(f"Gripper closed on nothing (measured {measured:.0f}%, commanded "
                              f"{command:.0f}%). The piece was missed; check board and joint calibration.")


def camera_config(index=0, width=1280, height=720, fps=30):
    from lerobot.cameras.opencv import OpenCVCameraConfig
    return OpenCVCameraConfig(index_or_path=index, fps=fps, width=width, height=height)


def make_so101(port, robot_id, camera_index=0, width=1280, height=720, fps=30,
               max_relative_target=10.0, release_torque=False):
    """A real LeRobot SO-101 follower with the overhead camera attached.

    By default the motors keep holding the parked pose after disconnecting:
    with torque released the arm would fall onto the board. Support the arm and
    switch off its power to release it.
    """
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    config = SO101FollowerConfig(port=port, id=robot_id, use_degrees=True,
                                 max_relative_target=max_relative_target,
                                 disable_torque_on_disconnect=release_torque,
                                 cameras={CAMERA: camera_config(camera_index, width, height, fps)})
    return SO101Follower(config)


def open_camera(index=0, width=1280, height=720, fps=30):
    """The overhead camera alone, through LeRobot, for steps that need no arm."""
    from lerobot.cameras.opencv import OpenCVCamera
    camera = OpenCVCamera(camera_config(index, width, height, fps))
    camera.connect()
    return camera
