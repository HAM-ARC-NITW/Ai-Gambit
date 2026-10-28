"""Windows/Linux entry point: inspect the scene, check the camera, or try stacking."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import mujoco
from env import RowEnv,CONTROL_HZ

HERE = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack",action="store_true",help="Run the existing RGB stacking engine live")
    parser.add_argument("--check",action="store_true",help="Check rendered detections against ground truth and exit")
    parser.add_argument("--headless",action="store_true",help="Do not open the 3D viewer")
    parser.add_argument("--seconds",type=float,help="Stop after this many wall-clock seconds")
    args = parser.parse_args(argv)
    if args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds<=0):
        parser.error("--seconds must be finite and positive")
    if args.check and args.stack:parser.error("Use --check or --stack separately")
    if args.headless and not args.stack and args.seconds is None:args.seconds=2
    output = HERE/"results"
    output.mkdir(exist_ok=True)
    env = RowEnv(render=args.check or args.stack)
    viewer,engine = None,None
    deadline = time.monotonic()+args.seconds if args.seconds is not None else float("inf")
    try:
        print("Scene loaded: SO-101, blocks, white target and top camera.",flush=True)
        if args.check:
            from vision import TopCamera,detect
            from PIL import Image
            # The resting arm crosses the row. Just as the engine does before
            # observing, park it to the side so all four block tops are visible.
            env.data.qpos[env.qadr[0]] = -1.5
            env.data.ctrl[env.act[0]] = -1.5
            mujoco.mj_forward(env.model,env.data)
            image = env.get_image()
            scene = detect(image,TopCamera(env.model))
            truth = env.block_positions()[:,:2]
            errors = [float(np.min(np.linalg.norm(truth-b["xy"],axis=1))) for b in scene["blocks"]]
            square = scene["square"]
            square_error = float(np.linalg.norm(square["xy"]-env.data.geom_xpos[env.sq][:2])) if square else None
            # Ground truth is used only here for checking, never by the RGB engine.
            matched = {int(np.argmin(np.linalg.norm(truth-b["xy"],axis=1))) for b in scene["blocks"]}
            ok = len(matched)==len(errors)==len(truth) and max(errors,default=1)<.01 and square_error is not None and square_error<.005
            report = dict(passed=bool(ok),blocks_detected=len(errors),block_errors_mm=[e*1000 for e in errors],
                          square_error_mm=None if square_error is None else square_error*1000)
            (output/"setup_check.json").write_text(json.dumps(report,indent=2))
            Image.fromarray(image).save(output/"top_camera.png")
            print(json.dumps(report,indent=2),flush=True)
            if not ok:raise SystemExit("Camera setup check failed; inspect results/top_camera.png")
            return
        if not args.headless:
            import mujoco.viewer as mjviewer
            viewer = mjviewer.launch_passive(env.model,env.data)
            viewer.cam.lookat[:] = [.16,.035,.05]
            viewer.cam.distance,viewer.cam.azimuth,viewer.cam.elevation = .7,140,-35
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False

        def refresh(pace=True):
            if time.monotonic()>deadline or (viewer is not None and not viewer.is_running()):raise KeyboardInterrupt
            if viewer is not None:
                viewer.sync()
                if pace:time.sleep(1/CONTROL_HZ)

        env.on_step = refresh
        env.on_plan = lambda:refresh(False)
        if args.stack:
            from engine import StackEngine
            engine = StackEngine(env)
            engine.run()
        else:
            targets = env.get_obs()["qpos"]
            while True:env.step(targets,render=False)
    except KeyboardInterrupt:
        print("Simulation stopped.",flush=True)
    finally:
        if engine is not None:
            positions = env.block_positions()
            count = env.blocks_on_square()
            report = dict(blocks_over_target=count,total_blocks=len(positions),positions=positions.tolist(),
                          all_blocks_over_target=count==len(positions))
            (output/"last_run.json").write_text(json.dumps(report,indent=2))
            if engine.frames:np.save(output/"stack_run.npy",np.asarray(engine.frames))
            print("Ground-truth XY scoring (not proof of a stable tower):",count,"of",len(positions),flush=True)
        if viewer is not None:viewer.close()
        env.close()


if __name__=="__main__":main()
