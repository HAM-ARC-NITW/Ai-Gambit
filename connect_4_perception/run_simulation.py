"""Open the Connect 4 scene in MuJoCo.

Run this file directly, or double-click run_simulation.cmd on Windows.
This first launcher displays the arm, board and sample token. Game rules,
human input, perception and robot moves will be added in later steps.
"""
import argparse
import math
from pathlib import Path
import time

import mujoco
import mujoco.viewer


def main():
    parser = argparse.ArgumentParser(description="Launch the SO-101 Connect 4 simulation")
    parser.add_argument("--no-viewer", action="store_true", help="Step physics without a window")
    parser.add_argument("--seconds", type=float, help="Close automatically after this many simulated seconds")
    parser.add_argument("--fps", type=int, choices=(30,60), default=30, help="Viewer refresh rate (default: 30)")
    args = parser.parse_args()
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds<=0):
        parser.error("--seconds must be a finite number greater than zero")
    if args.no_viewer and args.seconds is None:
        parser.error("Use --seconds with --no-viewer, for example --no-viewer --seconds 2")

    # Resolve the scene beside this script, even when launched from another folder.
    scene = Path(__file__).resolve().with_name("connect4_new.xml")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    # The scene's keyframe supplies both starting joint angles and motor targets.
    # Keeping those targets holds the arm while we inspect the environment.
    mujoco.mj_resetDataKeyframe(model,data,model.key("test").id)
    mujoco.mj_forward(model,data)
    print("Connect 4 simulation loaded:",scene,flush=True)

    if args.no_viewer:
        while data.time<args.seconds:
            mujoco.mj_step(model,data)
        print(f"Physics check completed: {data.time:.3f} simulated seconds.")
        return

    # Physics keeps its original 2 ms timestep. Only redraw the viewer 30 times
    # per second by default, instead of redrawing after every physics step.
    steps_per_frame = max(1,round(1/(args.fps*model.opt.timestep)))
    with mujoco.viewer.launch_passive(model,data) as viewer:
        viewer.cam.lookat[:] = [.18,0,.12]
        viewer.cam.distance = .85
        viewer.cam.azimuth = 140
        viewer.cam.elevation = -25
        # Lighter rendering helps the Windows viewer on slower graphics hardware.
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        print("Drag to rotate/pan, scroll to zoom. Close the viewer or press Ctrl+C to exit.",flush=True)
        while viewer.is_running():
            if args.seconds is not None and data.time>=args.seconds:
                break
            frame_start = time.perf_counter()
            mujoco.mj_step(model,data,nstep=steps_per_frame)
            viewer.sync()
            # Subtract computation time so it is not added to every frame's delay.
            remaining = steps_per_frame*model.opt.timestep-(time.perf_counter()-frame_start)
            if remaining>0:
                time.sleep(remaining)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nConnect 4 simulation stopped.")
