"""
AI-Gambit SO-101 Robotic Chess Simulation Runner.
Simulation & Planning Team.

Launches the interactive 3D MuJoCo viewer and executes precision pick-and-place
chess moves with sub-millimeter accuracy and zero piece collisions.

Usage:
    python3 run_simulation.py
    python3 run_simulation.py --moves e2e4 d7d5 g1f3 e4d5
    python3 run_simulation.py --speed 1.5
    python3 run_simulation.py --no-viewer
"""

import sys
import time
import argparse
import mujoco
import mujoco.viewer
import numpy as np

from chess_robot import ChessRobot
from chess_coords import get_piece_at_square


def main():
    parser = argparse.ArgumentParser(description="SO-101 Robotic Chess Arm Simulation Runner")
    parser.add_argument("--moves", nargs="+", default=["e2e4", "d7d5", "g1f3", "e4d5"],
                        help="List of chess moves in algebraic notation (e.g. e2e4 d7d5)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Simulation speed factor (default 1.0, higher = faster)")
    parser.add_argument("--no-viewer", action="store_true",
                        help="Run in headless mode without 3D GUI window")
    parser.add_argument("--fps", type=int, choices=[30, 60], default=60,
                        help="Viewer updates per second (physics rate stays unchanged)")
    parser.add_argument("--low-graphics", action="store_true",
                        help="Disable shadows and reflections for slower graphics hardware")
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be a finite number greater than zero")

    print("=" * 60)
    print("  AI-GAMBIT SO-101 ROBOTIC CHESS SIMULATION")
    print("  Simulation & Planning Team")
    print("=" * 60)
    print(f"Scheduled Moves : {args.moves}")
    print(f"Speed Factor    : {args.speed}x")
    print(f"Viewer Mode     : {'Headless' if args.no_viewer else 'Interactive 3D Viewer'}")
    print("-" * 60)

    # Initialize Robot Controller
    robot = ChessRobot("chess_world.xml")
    robot.planner.viewer_fps = args.fps

    if args.no_viewer:
        print("\nStarting headless execution...")
        for i, m in enumerate(args.moves):
            print(f"\n[Move {i+1}/{len(args.moves)}] Playing {m.upper()}...")
            success = robot.play_move(m, speed=args.speed)
            if not success:
                print(f"Move {m} failed!")
                sys.exit(1)
        print("\nAll moves completed successfully!")
        return

    # Interactive 3D Viewer
    try:
        with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
            if args.low_graphics:
                # These are viewer-only options; camera images and physics keep
                # their original settings.
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            print("\n[Viewer] Interactive 3D window launched!")
            print("[Viewer] Adjust camera view with Mouse (Left: Rotate, Right: Pan, Scroll: Zoom)")

            # Set top-angled camera view
            viewer.cam.distance = 0.75
            viewer.cam.azimuth = 90.0
            viewer.cam.elevation = -45.0
            viewer.cam.lookat = np.array([0.25, 0.0, 0.05])
            viewer.sync()

            # Brief pause before starting moves
            time.sleep(1.0)

            # Play moves
            for i, m in enumerate(args.moves):
                if not viewer.is_running():
                    break
                print(f"\n[Move {i+1}/{len(args.moves)}] Playing {m.upper()}...")
                success = robot.play_move(m, viewer=viewer, speed=args.speed)
                if not success:
                    print(f"Execution halted on move {m}")
                    break
                time.sleep(0.5)

            print("\n" + "=" * 60)
            print("  CHESS SEQUENCE COMPLETE!")
            print("  Window is staying open for inspection. Close viewer to exit.")
            print("=" * 60)

            # Keep window open for user inspection
            frame_steps = max(1, round(args.speed / (args.fps * robot.model.opt.timestep)))
            while viewer.is_running():
                q_hold = robot.data.ctrl[:6].copy()
                robot.planner.execute_segment(q_hold, q_hold, steps=frame_steps,
                                              viewer=viewer, speed_factor=args.speed)

    except (InterruptedError, KeyboardInterrupt):
        print("\nSimulation stopped.")
    except Exception as e:
        # Replaying all moves here would execute them on a partly changed board.
        print(f"\nSimulation stopped with an error: {e}")
        raise


if __name__ == "__main__":
    main()
