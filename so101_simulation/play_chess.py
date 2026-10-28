"""Play White against the arm: click two squares, then Enter to move.

Also accepts typed UCI moves in the camera/grid window. Your virtual hand places
pieces; RGB perception independently recognizes the legal outcome. The arm then
executes its reply, and the same camera must verify that reply before continuing.
"""
import argparse
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import chess
import chess.engine
import mujoco.viewer
from perception_sim import Simulation, show_observation, overlay, HERE
from chess_game import Coordinator, occupancy
# Also work when launched directly from inside so101_simulation.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine_stuff.stockfish_engine import StockfishEngine


def checked_snapshot(sim, check, observer=None, attempts=3):
    """Snapshot, then check it with check(obs) -> (ok, reason).

    A busy computer can spend longer than max_frame_age processing one frame.
    That frame is correctly refused as expired; the remedy is a fresh snapshot,
    not a looser limit. Only the observation is retried, never an action.
    """
    for attempt in range(attempts):
        obs = sim.snapshot(on_observation=observer)
        ok, reason = check(obs)
        if ok or "expired" not in reason or attempt == attempts-1:
            return obs, ok, reason


def play_turn(sim, game, text, animate=False, viewer=None, speed=1, observer=None):
    """Shared by the GUI and automated end-to-end game tests."""
    before, ok, reason = checked_snapshot(sim, game.baseline, observer)
    if not ok:
        raise ValueError("Cannot start move: " + reason)
    requested = chess.Move.from_uci(text)
    if requested not in game.board.legal_moves:
        raise ValueError("Illegal move: " + text)
    started = time.time()
    mapping, rack, tray = sim.physical_move(game.board, requested, animate, viewer, speed,observation=before)
    options = []

    def verify(obs):
        recognized, options[:] = game.interpret(obs, selection=text)
        if recognized != requested:
            raise ValueError("Observed move differs from the requested physical action")
        return game.accept(recognized, obs, started, rack, tray)

    observed, ok, reason = checked_snapshot(sim, verify, observer)
    if not ok:
        raise ValueError("Verification failed; game paused: " + reason)
    sim.mapping, sim.rack_inventory, sim.tray_inventory = mapping, rack, tray
    return observed, options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run a scripted game without windows")
    parser.add_argument("--moves", nargs="+", default=["e2e4", "g1f3"], help="Scripted White moves for --headless")
    parser.add_argument("--engine",help="Path to the Stockfish executable (or set STOCKFISH_PATH)")
    parser.add_argument("--think-time",type=float,default=.5,help="Engine seconds per reply (default .5)")
    parser.add_argument("--skill",type=int,choices=range(21),default=10,help="Stockfish skill 0..20")
    parser.add_argument("--depth",type=int,help="Optional search depth cap, in addition to thinking time")
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--no-animation", action="store_true", help="Instant robot transfers for quick perception testing")
    parser.add_argument("--demo", action="store_true", help="Visually replay the scripted White moves and arm replies")
    parser.add_argument("--exit-after-demo", action="store_true", help="Close after --demo finishes (useful for GUI smoke tests)")
    parser.add_argument("--fen", default=chess.STARTING_FEN, help="Optional legal starting position")
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    if not np.isfinite(args.think_time) or args.think_time<=0 or (args.depth is not None and args.depth<1):
        parser.error("Thinking time and optional depth must be finite/positive")
    board = chess.Board(args.fen)
    if not board.is_valid() or board.turn != chess.WHITE:
        parser.error("Starting FEN must be a valid position with White to move")
    # Fail clearly before loading the large scene if the engine is unavailable.
    try:engine = StockfishEngine(args.engine,args.think_time,args.skill)
    except (OSError,ValueError,TimeoutError,chess.engine.EngineError) as error:parser.error(str(error))
    print(f"Chess engine: {engine.name}; skill {args.skill}; think time {args.think_time}s",flush=True)
    sim = None
    try:
        sim = Simulation()
        game = Coordinator(sim.detector, board)
        sim.load_position(board)
        game.rack = sim.rack_inventory.copy()
        if args.headless:
            for text in args.moves:
                obs, _ = play_turn(sim, game, text)
                print(f"Human {text}: camera verified, version {game.version}")
                if game.board.is_game_over(claim_draw=True):
                    break
                reply = engine.choose_move(game.board,args.depth)
                obs, _ = play_turn(sim, game, reply.uci(), animate=not args.no_animation, speed=args.speed)
                print(f"Arm {reply}: camera verified, version {game.version}")
            print("Confirmed FEN:", game.board.fen())
            return
        text, selected, message, paused = "", [], "You play White. Click source and destination, then Enter.", False
        demo_index = 0
        latest = sim.snapshot()

        def click(event, x, y, flags, param):
            nonlocal text, selected
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(selected) == 2:
                    selected = []
                if 0 <= x < 800 and 0 <= y < 800:
                    selected.append(chess.square_name(chess.square(x//100,7-y//100)))
                    text = "".join(selected)

        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            viewer.cam.lookat = [.25,0,.02]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = .9, 90, -55
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            viewer.sync()
            title = "Chess"
            show_observation(sim, latest, title, message)
            cv2.setMouseCallback(title+" - measured occupancy", click)
            last_camera = 0.

            def during_motion():
                nonlocal last_camera
                if time.perf_counter()-last_camera > .3:
                    rgb = sim.frame()
                    cv2.putText(rgb,"ROBOT EXECUTING - waiting for final verification",(15,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,220,60),2)
                    cv2.imshow(title+" - overhead camera", cv2.cvtColor(cv2.resize(rgb,(960,720)),cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 255 == 27:
                        raise InterruptedError("Stopped by player")
                    last_camera = time.perf_counter()
            sim.robot.planner.viewer_callback = during_motion

            def observing(obs):
                # Pump the camera windows between observations so they remain
                # responsive while several stability frames are being collected.
                show_observation(sim,obs,title,"Observing physical board; waiting for a stable result...")
                viewer.sync()
                if cv2.waitKey(1) & 255 == 27:
                    raise InterruptedError("Stopped by player")

            while viewer.is_running():
                show_observation(sim, latest, title, message+"   Move: "+text)
                key = cv2.waitKey(30) & 255
                if key == 27 or (key == ord('q') and not text):
                    break
                if args.demo and not paused and game.board.turn == chess.WHITE:
                    if demo_index < len(args.moves):
                        text, key = args.moves[demo_index], 13
                        demo_index += 1
                    elif args.exit_after_demo:
                        break
                if key == ord('r') and not text:
                    sim.load_position(game.board)
                    # Reset is an explicit restart, since captured identities cannot
                    # be reconstructed from binary occupancy alone.
                    game = Coordinator(sim.detector, game.board)
                    game.rack = sim.rack_inventory.copy()
                    engine.new_game()
                    latest, paused = sim.snapshot(), False
                    message = "Restarted this FEN with a freshly loaded rack/tray."
                elif key in (8,127):
                    text, selected = text[:-1], []
                elif key in (10,13) and text:
                    if paused:
                        message = "Paused after mismatch. Clear input, then R to reload confirmed position."
                        text, selected = "", []
                        continue
                    if len(text) == 4 and text[1] in '27' and text[3] in '18':
                        p = game.board.piece_at(chess.parse_square(text[:2]))
                        if p and p.piece_type == chess.PAWN:
                            message = "Promotion: append q, r, b or n, then Enter."
                            continue
                    requested = text
                    text, selected = "", []
                    try:
                        latest, options = play_turn(sim, game, requested,observer=observing)
                        message = "Camera accepted human "+requested
                        if game.board.is_game_over(claim_draw=True):
                            message = "Game over: "+game.board.result(claim_draw=True)
                            continue
                        show_observation(sim,latest,title,"Thinking...")
                        cv2.waitKey(1)
                        reply = engine.choose_move(game.board,args.depth)
                        message = "Arm playing "+reply.uci()
                        show_observation(sim,latest,title,message)
                        cv2.waitKey(1)
                        latest, _ = play_turn(sim,game,reply.uci(),animate=not args.no_animation,viewer=viewer,speed=args.speed,observer=observing)
                        message = f"Verified arm {reply}. Your turn. Version {game.version}."
                        if game.board.is_game_over(claim_draw=True):
                            message = "Game over: "+game.board.result(claim_draw=True)
                    except (ValueError,TimeoutError,chess.engine.EngineError) as error:
                        message = str(error)
                        latest = sim.snapshot()
                        paused = (isinstance(error,(TimeoutError,chess.engine.EngineError)) or occupancy(game.board) != latest.occupancy or latest.status != "READY" or game.rack != latest.rack or game.tray != latest.tray)
                        print(message)
                elif key != 255 and chr(key) in "abcdefgh12345678qrbn":
                    text += chr(key)
                # Keep the 3D viewer responsive without advancing unrequested turns.
                viewer.sync()
    except (KeyboardInterrupt, InterruptedError):
        print("Game stopped; no unverified turn committed.")
    finally:
        if sim is not None:sim.close()
        engine.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
