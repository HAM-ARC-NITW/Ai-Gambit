"""One-time setup of the real SO-101 and overhead camera for play_real.py.

Run the steps in this order (README: "Real arm with LeRobot"). Each saves its
result in real_setup.json. Nothing here plays chess.

  init        create real_setup.json:  init --port COM5 [--camera 0]
  check       motors released: move the arm by hand, the twin must copy it; camera tags
  gripper     open and close the jaws, empty and on a pawn, to check grasp sensing
  intrinsics  photograph a printed checkerboard from many angles -> lens calibration
  extrinsics  with the board at its marked place -> where the camera is
  empty       photograph the empty board, rack and tray (the arm parks first)
  hover       the arm points at the four corner squares so you can check its aim
"""
import argparse
import time
import cv2
import numpy as np
import mujoco
# real_workspace (through perception_sim) puts the perception package on the path.
from real_workspace import SETUP, RealWorkspace, connect_arm, load_setup, save_setup
from perception_sim import Simulation
from ai_gambit_perception.perception import Perception, load_config, transform
from chess_coords import CLEARANCE_HEIGHT_Z, square_to_pos
from lerobot_arm import DEFAULTS, JOINTS, open_camera

EMPTY_PHOTO = "real_empty_reference.png"
WINDOW = "Setup camera"


def camera_of(setup):
    c = setup.get("camera", {})
    return open_camera(c.get("index", 0), c.get("width", 1280), c.get("height", 720), c.get("fps", 30))


def show(rgb, lines):
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for i, text in enumerate(lines):
        cv2.putText(image, text, (15, 32+i*30), cv2.FONT_HERSHEY_SIMPLEX, .7, (40, 220, 255), 2, cv2.LINE_AA)
    scale = 960 / image.shape[1]
    cv2.imshow(WINDOW, cv2.resize(image, None, fx=scale, fy=scale))
    return cv2.waitKey(30) & 255


def draw_tags(rgb):
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    corners, ids, _ = detector.detectMarkers(rgb)
    image = rgb.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
    return image, [] if ids is None else sorted(int(i) for i in ids.flat)


def init(args):
    if SETUP.exists() and not args.force:
        raise SystemExit(f"{SETUP.name} already exists; add --force to start over")
    arm = {k: v for k, v in DEFAULTS.items() if k != "realtime"}
    save_setup(dict(port=args.port, robot_id=args.robot_id,
                    camera=dict(index=args.camera, width=args.width, height=args.height, fps=30),
                    release_torque_on_exit=False, arm=arm,
                    camera_matrix=None, distortion=None, base_camera_optical=None, empty_reference=None))
    print(f"Wrote {SETUP}. Next: python real_setup.py check")


def check(args):
    """Motors released: pose the arm by hand and watch the twin copy it. A joint
    that turns the other way needs sign -1; a constant difference needs offset_deg."""
    setup = load_setup()
    arm = connect_arm(setup)
    twin = Simulation()
    renderer = mujoco.Renderer(twin.model, 480, 640)
    view = mujoco.MjvCamera()
    view.lookat[:], view.distance, view.azimuth, view.elevation = [.12, 0, .12], .8, 140, -20
    try:
        bus = getattr(arm.robot, "bus", None)
        if bus is not None:
            bus.disable_torque()
        expected = sorted(int(m) for g in ("board", "rack", "tray") for m in load_config()[f"{g}_markers"])
        print("Motors are released: support the arm and move each joint by hand.")
        print("The 'Twin' window must move the same way. Esc: finish (put the arm back at rest first).")
        while True:
            obs = arm.robot.get_observation()
            q = arm.from_observation(obs)
            twin.data.qpos[:6] = q
            mujoco.mj_forward(twin.model, twin.data)
            renderer.update_scene(twin.data, view)
            cv2.imshow("Twin - must match the real arm", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
            image, seen = draw_tags(arm.frame())
            missing = [m for m in expected if m not in seen]
            lines = ["  ".join(f"{n[:8]} {obs[n+'.pos']:.0f}" for n in JOINTS),
                     "twin rad: " + " ".join(f"{v:.2f}" for v in q),
                     f"tags seen {seen}" + (f"  MISSING {missing}" if missing else "  (all fixtures)")]
            if show(image, lines) == 27:
                print("\n".join(lines))
                break
    finally:
        renderer.close()
        twin.close()
        arm.disconnect()


def gripper(args):
    setup = load_setup()
    workspace = RealWorkspace(connect_arm(setup))
    arm = workspace.arm
    try:
        workspace.park()
        q = workspace.data.qpos[:6].copy()

        def jaws(angle, label):
            target = q.copy()
            target[5] = angle
            workspace.robot.planner.execute_segment(workspace.data.qpos[:6].copy(), target, steps=150)
            time.sleep(.8)
            measured = arm.robot.get_observation()["gripper.pos"]
            print(f"{label}: commanded {arm.last[5]:.0f}%, measured {measured:.0f}%")
            return measured, arm.last[5]

        jaws(.50, "open")
        _, command = jaws(.35, "closed on nothing")
        input("Hold a pawn upright between the open jaws, then press Enter...")
        jaws(.50, "open")
        measured, _ = jaws(.35, "closed on the pawn")
        stall = measured - command
        print(f"The pawn stopped the jaws {stall:.0f}% above the command "
              f"(grasp_margin_pct is {arm.settings['grasp_margin_pct']}).")
        if stall <= arm.settings["grasp_margin_pct"]:
            print("Too small to sense a grasp: lower the twin clamp angle or raise grasp_margin_pct.")
        jaws(.50, "open (take the pawn away)")
    finally:
        workspace.close()


def intrinsics(args):
    setup = load_setup()
    camera = camera_of(setup)
    pattern = tuple(int(n) for n in args.corners.lower().split("x"))
    grid = np.zeros((pattern[0]*pattern[1], 3), np.float32)
    grid[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * args.square
    points, pixels, size = [], [], None
    print("Hold the printed checkerboard in view at many angles and distances.")
    print("Space: keep this view (when the corners are found).  C: calibrate (after 12+).  Esc: cancel")
    try:
        while True:
            rgb = camera.read()
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            size = gray.shape[::-1]
            found, corners = cv2.findChessboardCorners(gray, pattern)
            image = rgb.copy()
            if found:
                cv2.drawChessboardCorners(image, pattern, corners, found)
            key = show(image, [f"views kept: {len(points)}", "corners found" if found else "no checkerboard"])
            if key == 32 and found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, .001))
                points.append(grid)
                pixels.append(corners)
            elif key == ord("c") and len(points) >= 12:
                break
            elif key == 27:
                return
    finally:
        camera.disconnect()
    rms, K, distortion, _, _ = cv2.calibrateCamera(points, pixels, size, None, None)
    print(f"Reprojection error {rms:.2f} px (under 1 px is good)")
    setup["camera_matrix"], setup["distortion"] = K.tolist(), distortion.ravel().tolist()
    save_setup(setup)
    print("Saved the lens calibration. Next: python real_setup.py extrinsics")


def camera_pose(rgb, K, distortion, config):
    """Camera pose in the arm-base frame, from the board tags, when the board
    stands exactly at its configured place. Returns (4x4 pose, tag fit error px)."""
    finder = Perception(config, K, distortion, base_camera=np.eye(4))  # Poses in camera coordinates.
    found, _ = finder._localize(cv2.undistort(rgb, K, distortion))
    if "board" not in found:
        raise SystemExit("Board tags not found well enough; check lighting and that the tags are flat")
    _, board_in_camera, error = found["board"]
    board_in_world = transform(config["board_rotation"], config["board_origin"])
    return board_in_world @ np.linalg.inv(board_in_camera), error


def extrinsics(args):
    setup = load_setup()
    if setup.get("camera_matrix") is None:
        raise SystemExit("Run intrinsics first")
    config = load_config()
    K, distortion = np.array(setup["camera_matrix"]), np.array(setup["distortion"])
    camera = camera_of(setup)
    print("Place the board exactly at its marked position (README). Space: measure.  Esc: cancel")
    try:
        while True:
            rgb = camera.read()
            image, seen = draw_tags(rgb)
            key = show(image, [f"tags seen {seen}", "Space: measure the camera position"])
            if key == 32:
                break
            if key == 27:
                return
    finally:
        camera.disconnect()
    base_camera, error = camera_pose(rgb, K, distortion, config)
    print(f"Camera is at {np.round(base_camera[:3, 3], 3)} m from the arm base "
          f"(simulated design {config['camera_position']}); tag fit error {error:.2f} px")
    setup["base_camera_optical"] = base_camera.tolist()
    save_setup(setup)
    print("Saved. Next: python real_setup.py empty")


def empty(args):
    setup = load_setup()
    if setup.get("base_camera_optical") is None:
        raise SystemExit("Run intrinsics and extrinsics first")
    workspace = RealWorkspace(connect_arm(setup))
    try:
        print("Parking the arm clear of the camera...")
        workspace.park()
        input("Remove every piece from the board, rack and tray, keep hands clear, then press Enter...")
        frames = [workspace.frame().astype(np.float32) for _ in range(9)]
        photo = np.median(frames, axis=0).astype(np.uint8)  # Less sensor noise than one frame.
        K, distortion = np.array(setup["camera_matrix"]), np.array(setup["distortion"])
        Perception(load_config(), K, distortion, np.array(setup["base_camera_optical"])).set_empty_reference(photo)
        cv2.imwrite(str(SETUP.parent / EMPTY_PHOTO), cv2.cvtColor(photo, cv2.COLOR_RGB2BGR))
        setup["empty_reference"] = EMPTY_PHOTO
        save_setup(setup)
        print(f"Saved {EMPTY_PHOTO}. Next: python real_setup.py hover")
    finally:
        workspace.close()


def hover(args):
    setup = load_setup()
    workspace = RealWorkspace(connect_arm(setup))
    planner, robot = workspace.robot.planner, workspace.robot
    try:
        workspace.park()
        q = workspace.data.qpos[:6].copy()
        for square in ("a1", "h1", "h8", "a8"):
            target = square_to_pos(square)
            above, low = target.copy(), target.copy()
            above[2], low[2] = CLEARANCE_HEIGHT_Z, target[2] + args.height
            for point in (above, low):
                goal = robot._reach(point, q)
                goal[5] = .5
                planner.execute_segment(q, goal, steps=250)
                q = goal
            input(f"The jaws should be centred {args.height*100:.0f} cm above {square}. Press Enter...")
            goal = robot._reach(above, q)
            goal[5] = .5
            planner.execute_segment(q, goal, steps=250)
            q = goal
        print("If a corner was off-centre, move the board (not the arm) and repeat extrinsics and hover.")
    finally:
        workspace.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    p = steps.add_parser("init", help="create real_setup.json")
    p.add_argument("--port", required=True, help="Arm USB port, from lerobot-find-port (e.g. COM5, /dev/ttyACM0)")
    p.add_argument("--robot-id", default="chess_arm", help="The id you used with lerobot-calibrate")
    p.add_argument("--camera", type=int, default=0, help="Overhead camera index")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--force", action="store_true")
    steps.add_parser("check", help="move the arm by hand; the twin must copy it")
    steps.add_parser("gripper", help="check that the gripper senses a held pawn")
    p = steps.add_parser("intrinsics", help="lens calibration from a printed checkerboard")
    p.add_argument("--corners", default="9x6", help="Inner corners per row x column (default 9x6)")
    p.add_argument("--square", type=float, default=.025, help="Checkerboard square size in metres")
    steps.add_parser("extrinsics", help="camera position from the board tags")
    steps.add_parser("empty", help="photograph the empty fixtures")
    p = steps.add_parser("hover", help="arm points at the corner squares")
    p.add_argument("--height", type=float, default=.06, help="Hover height above the board, metres")
    args = parser.parse_args()
    try:
        globals()[args.step](args)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error))
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
