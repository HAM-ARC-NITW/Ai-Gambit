"""The real-arm path through LeRobot, tested without hardware.

LeRobotArm is driven against stand-ins with LeRobot's Robot interface. The
stand-in table (lerobot_standin.py) only moves a piece when the jaws close
around it, so these games fail if units, pacing or aim are wrong.
Run with: python -m pytest test_lerobot.py
"""
import numpy as np
import pytest
import chess
import cv2
from perception_sim import Simulation
from chess_game import Coordinator, occupancy
from play_chess import play_turn
from play_game import start_human_turn, press_button
from virtual_hand import VirtualHand
from lerobot_arm import ArmError, GraspFailed, LeRobotArm, JOINTS
from lerobot_standin import StandInSO101, standin_calibration
from real_workspace import RealWorkspace
from real_setup import camera_pose
from ai_gambit_perception.perception import camera_calibration


class Recorder:
    """A LeRobot-shaped robot that holds whatever it was last told."""
    is_connected = True

    def __init__(self, degrees=None):
        self.state = {f"{n}.pos": 0. for n in JOINTS} if degrees is None else dict(degrees)
        self.actions = []

    def get_observation(self):
        return dict(self.state)

    def send_action(self, action):
        self.actions.append(action)
        self.state = dict(action)
        return action


@pytest.fixture(scope="module")
def table():
    world = Simulation(button=True)
    robot = StandInSO101(world)
    arm = LeRobotArm(robot, {"realtime": False})
    arm.connect()
    workspace = RealWorkspace(arm, standin_calibration(world))
    yield world, robot, workspace
    workspace.close()


def setup(table, fen=chess.STARTING_FEN, true_settings=None):
    world, robot, workspace = table
    board = chess.Board(fen)
    world.load_position(board)
    robot.units = LeRobotArm(None, true_settings)
    robot.held, robot.closed = None, False
    workspace.load_position(board)
    game = Coordinator(workspace.detector, board)
    game.rack = workspace.rack_inventory.copy()
    return world, workspace, game


def test_units_round_trip_with_signs_and_offsets():
    arm = LeRobotArm(None, {"sign": [1, -1, 1, 1, -1], "offset_deg": [5, 0, -3, 0, 10]})
    q = np.array([.3, -.6, 1.1, -.2, .4, .35])
    action = arm.to_action(q)
    assert action["shoulder_lift.pos"] == pytest.approx(-np.degrees(-.6))
    assert action["gripper.pos"] == pytest.approx(27.5)  # 0.35 rad on the -0.2..1.8 rad -> 0..100 % scale.
    assert np.allclose(arm.from_observation(action), q)


def test_unsafe_commands_are_refused_before_sending():
    model = Simulation().model
    robot = Recorder()
    arm = LeRobotArm(robot, {"realtime": False})
    arm.attach(model)
    arm.start()
    with pytest.raises(ArmError, match="jump"):
        arm.send([np.radians(20), 0, 0, 0, 0, 0])
    with pytest.raises(ArmError, match="limits"):
        arm.send([0, 3., 0, 0, 0, 0])
    assert robot.actions == []
    arm.send([np.radians(5), 0, 0, 0, 0, 0])
    assert len(robot.actions) == 1


def test_motion_is_paced_in_real_time():
    clock = [0.]
    robot = Recorder()
    arm = LeRobotArm(robot, {"slowdown": 4, "rate_hz": 30}, clock=lambda: clock[0],
                     sleep=lambda s: clock.__setitem__(0, clock[0] + s))
    arm.start()
    q = np.zeros(6)
    for s in range(1000):  # Two simulated seconds.
        q[0] = np.radians(30) * s / 999
        arm.follow(q.copy(), final=s == 999)
    assert clock[0] == pytest.approx(8., abs=.1)  # 2 s x slowdown 4.
    # 25-30 commands per second: sends fall on whole 2 ms simulation steps.
    assert 195 <= len(robot.actions) <= 245


def test_segments_are_stretched_to_a_safe_joint_speed():
    arm = LeRobotArm(None, {"max_speed_deg_s": 60, "slowdown": 4})
    steps = arm.steps_for(np.zeros(6), np.array([1., 0, 0, 0, 0, 0]), 25)
    seconds = steps * .002 * 4
    assert np.pi / 2 * np.degrees(1.) / seconds <= 60.01  # Peak speed of the cosine profile.
    assert arm.steps_for(np.zeros(6), np.full(6, 1e-4), 25) == 25


def test_arm_must_physically_arrive(table):
    world, _, _ = table
    robot = StandInSO101(world, stuck=True)
    robot.connect()
    arm = LeRobotArm(robot, {"realtime": False, "arrive_timeout": .2})
    arm.start()
    with pytest.raises(ArmError, match="did not reach"):
        # 6.9 degrees: beyond the 4 degree arrival tolerance, within the 8 degree step limit.
        arm.follow(arm.read_q() + [.12, 0, 0, 0, 0, 0], final=True)


def test_turns_through_lerobot_are_verified_by_the_camera(table):
    world, workspace, game = setup(table)
    play_turn(workspace, game, "e2e4", animate=True)
    hand = VirtualHand(world)  # The person at the real table.
    _, started = start_human_turn(workspace, game, hand)
    hand.perform(game.board, chess.Move.from_uci("d7d5"))
    press_button(workspace, game, hand, started)
    play_turn(workspace, game, "e4d5", animate=True)  # Capture: the arm clears d5 into the tray.
    assert game.version == 3
    assert world.snapshot().occupancy == occupancy(game.board)
    assert sum(game.tray) == 1


def test_miscalibrated_arm_misses_and_nothing_is_committed(table):
    # The real shoulder is 4 degrees off what real_setup.json assumes.
    world, workspace, game = setup(table, true_settings={"offset_deg": [4, 0, 0, 0, 0]})
    fen = game.board.fen()
    with pytest.raises(GraspFailed):
        play_turn(workspace, game, "e2e4", animate=True)
    assert game.version == 0 and game.board.fen() == fen
    workspace.retreat()  # Lifts clear and parks without touching the pieces.
    assert world.snapshot().occupancy == occupancy(game.board)


def test_real_arm_never_teleports_pieces(table):
    _, workspace, game = setup(table)
    with pytest.raises(ArmError, match="teleport"):
        workspace.physical_move(game.board, chess.Move.from_uci("e2e4"), animate=False)


def test_camera_position_is_recovered_from_the_board_tags(table):
    world, _, _ = table
    world.load_position(chess.Board())
    K, distortion, true_pose = camera_calibration(world.config)
    pose, error = camera_pose(world.frame(), K, distortion, world.config)
    assert error < 1.5
    assert np.linalg.norm(pose[:3, 3] - true_pose[:3, 3]) < .003
    angle = np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3].T @ true_pose[:3, :3]) - 1) / 2, -1, 1)))
    assert angle < .5


def test_real_lerobot_follower_is_configured_for_the_game():
    pytest.importorskip("lerobot")
    from lerobot.robots.so_follower import SO101Follower
    from lerobot_arm import CAMERA, make_so101
    robot = make_so101("COM_TEST", "chess_test", camera_index=2, width=640, height=480)
    assert isinstance(robot, SO101Follower) and not robot.is_connected
    assert robot.config.use_degrees and not robot.config.disable_torque_on_disconnect
    camera = robot.config.cameras[CAMERA]
    assert (camera.index_or_path, camera.width, camera.height) == (2, 640, 480)
    assert set(robot.action_features) == {f"{n}.pos" for n in JOINTS}
