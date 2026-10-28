"""End-to-end game against the arm, driven by the camera like a chess clock.

1. The overhead camera reads the board; it must match the confirmed position.
2. Stockfish chooses the arm's move; the arm plays it; the camera verifies it.
3. You move pieces yourself (virtual hand or 3D-viewer drag) and press the big
   red button. Nothing tells the game your move: the camera works it out.
4. Repeat from 2.

The arm plays White and moves first unless --human white is given.
"""
import argparse
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import chess
import chess.engine
import mujoco
import mujoco.viewer
from perception_sim import Simulation, overlay
from chess_game import Coordinator, AmbiguousMove, occupancy
from play_chess import play_turn, checked_snapshot
from virtual_hand import VirtualHand
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine_stuff.stockfish_engine import StockfishEngine

BOARD, PANEL = 800, 170
BUTTON_CENTER, BUTTON_RADIUS = (720, BOARD + 85), 62
TITLE = "Chess"
IDLE_REFRESH = .8  # Seconds without clicks before the camera view is refreshed.


def read_board(sim, game, observer=None):
    """The camera reads the board, which must agree with the confirmed position."""
    obs, ok, reason = checked_snapshot(sim, game.baseline, observer)
    if not ok:
        raise ValueError("Board read failed: " + reason)
    return obs


def arm_turn(sim, game, engine, depth=None, animate=True, viewer=None, speed=1, observer=None):
    move = engine.choose_move(game.board, depth)
    obs, _ = play_turn(sim, game, move.uci(), animate=animate, viewer=viewer, speed=speed, observer=observer)
    return move, obs


def start_human_turn(sim, game, hand, observer=None):
    """Baseline the confirmed position before the human touches anything."""
    obs = read_board(sim, game, observer)
    hand.begin(game.board, game.rack, game.tray)
    return obs, time.time()


def press_button(sim, game, hand, started, selection=None, observer=None):
    """The big red button: the camera alone decides what the human played."""
    if hand.held is not None:
        raise ValueError("You are still holding a piece; put it down first")
    read = {}

    def verify(obs):
        read["move"], read["rack"], read["tray"] = game.read_human_move(obs, selection)
        return game.accept(read["move"], obs, started, read["rack"], read["tray"])

    obs, ok, reason = checked_snapshot(sim, verify, observer)
    if not ok:
        raise ValueError("Verification failed: " + reason)
    move, rack, tray = read["move"], read["rack"], read["tray"]
    sim.rack_inventory, sim.tray_inventory = list(rack), list(tray)
    sim.sync_mapping(game.board)
    return move, obs


def board_window(obs, hand, lines, button_lit, pressed):
    image = cv2.cvtColor(overlay(obs), cv2.COLOR_RGB2BGR)

    def corner(square):
        return chess.square_file(square)*100, (7-chess.square_rank(square))*100

    if hand.held_from is not None:
        x, y = corner(hand.held_from)
        cv2.rectangle(image, (x+4, y+4), (x+96, y+96), (0, 230, 255), 5)
        for move in hand.targets:
            x, y = corner(move.to_square)
            cv2.circle(image, (x+50, y+50), 16, (20, 20, 20), -1)
            cv2.circle(image, (x+50, y+50), 13, (60, 230, 60), -1)
    if hand.move is not None:
        for square in (hand.move.from_square, hand.move.to_square):
            x, y = corner(square)
            cv2.rectangle(image, (x+4, y+4), (x+96, y+96), (255, 160, 40), 5)
    panel = np.full((PANEL, BOARD, 3), 30, np.uint8)
    for i, (text, color) in enumerate(lines):
        cv2.putText(panel, text, (12, 28+i*30), cv2.FONT_HERSHEY_SIMPLEX, .52, color, 1, cv2.LINE_AA)
    image = np.vstack([image, panel])
    red = (20, 20, 120) if pressed else ((30, 30, 230) if button_lit else (60, 60, 90))
    cv2.circle(image, BUTTON_CENTER, BUTTON_RADIUS, (15, 15, 15), -1)
    cv2.circle(image, BUTTON_CENTER, BUTTON_RADIUS-(10 if pressed else 6), red, -1)
    cv2.putText(image, "DONE", (BUTTON_CENTER[0]-30, BUTTON_CENTER[1]+8), cv2.FONT_HERSHEY_SIMPLEX, .75,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(TITLE+" - board", image)


def camera_window(sim, text):
    rgb = sim.frame()
    cv2.putText(rgb, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 220, 60), 2)
    cv2.imshow(TITLE+" - overhead camera", cv2.cvtColor(cv2.resize(rgb, (960, 720)), cv2.COLOR_RGB2BGR))


def headless(sim, game, hand, engine, args, arm_color):
    human_moves = list(args.human_moves or [])
    auto = args.auto_human if not human_moves else 0
    read_board(sim, game)
    print("Camera read the board: matches the confirmed position", flush=True)
    while not game.board.is_game_over(claim_draw=True):
        if game.board.turn == arm_color:
            move, _ = arm_turn(sim, game, engine, args.depth, not args.no_animation, speed=args.speed)
            print(f"Arm {move}: camera verified, version {game.version}", flush=True)
            continue
        if human_moves:
            move = chess.Move.from_uci(human_moves.pop(0))
        elif auto > 0:
            auto -= 1
            move = engine.choose_move(game.board, args.depth)
        else:
            break
        if move not in game.board.legal_moves:
            raise ValueError("Scripted human move is illegal: " + move.uci())
        _, started = start_human_turn(sim, game, hand)
        hand.perform(game.board, move)
        try:
            played, _ = press_button(sim, game, hand, started)
        except AmbiguousMove as error:
            # A person would answer the prompt; the script knows its own move.
            print(f"  camera: {error}; answering {move}", flush=True)
            played, _ = press_button(sim, game, hand, started, selection=move.uci())
        if played != move:
            raise ValueError(f"Camera read {played}, but the hand played {move}")
        print(f"Human {played}: read by camera after button, version {game.version}", flush=True)
    print("Confirmed FEN:", game.board.fen())
    if game.board.is_game_over(claim_draw=True):
        print("Game over:", game.board.result(claim_draw=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--human", choices=("white", "black"), default="black",
                        help="Your colour (default black: the arm plays White and moves first)")
    parser.add_argument("--headless", action="store_true", help="Scripted game without windows")
    parser.add_argument("--demo", action="store_true", help="Windows open; a scripted hand makes your moves")
    parser.add_argument("--exit-after-demo", action="store_true", help="Close when --demo runs out of moves")
    parser.add_argument("--human-moves", nargs="+", help="Scripted human moves for --headless/--demo")
    parser.add_argument("--auto-human", type=int, default=3,
                        help="Without --human-moves, Stockfish makes this many scripted human moves")
    parser.add_argument("--engine", help="Path to the Stockfish executable (or set STOCKFISH_PATH)")
    parser.add_argument("--think-time", type=float, default=.5, help="Engine seconds per move (default .5)")
    parser.add_argument("--skill", type=int, choices=range(21), default=10, help="Stockfish skill 0..20")
    parser.add_argument("--depth", type=int, help="Optional search depth cap, in addition to thinking time")
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--no-animation", action="store_true", help="Instant arm transfers")
    parser.add_argument("--fen", default=chess.STARTING_FEN, help="Optional legal starting position")
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    if not np.isfinite(args.think_time) or args.think_time <= 0 or (args.depth is not None and args.depth < 1):
        parser.error("Thinking time and optional depth must be finite/positive")
    board = chess.Board(args.fen)
    if not board.is_valid():
        parser.error("Starting FEN must be a valid position")
    arm_color = chess.BLACK if args.human == "white" else chess.WHITE
    try:
        engine = StockfishEngine(args.engine, args.think_time, args.skill)
    except (OSError, ValueError, TimeoutError, chess.engine.EngineError) as error:
        parser.error(str(error))
    print(f"Chess engine: {engine.name}; skill {args.skill}; think time {args.think_time}s", flush=True)
    sim = None
    try:
        sim = Simulation(button=True)
        sim.load_position(board)
        game = Coordinator(sim.detector, board)
        game.rack = sim.rack_inventory.copy()
        hand = VirtualHand(sim)
        if args.headless:
            headless(sim, game, hand, engine, args, arm_color)
            return
        play(sim, game, hand, engine, args, arm_color)
    except (KeyboardInterrupt, InterruptedError):
        print("Game stopped; no unverified turn committed.")
    finally:
        if sim is not None:
            sim.close()
        engine.close()
        cv2.destroyAllWindows()


def play(sim, game, hand, engine, args, arm_color):
    human_name = "White" if arm_color == chess.BLACK else "Black"
    scripted = list(args.human_moves or [])
    auto = [args.auto_human if not scripted else 0]
    state = dict(phase="READ", message="Reading the board...", options=[], latest=sim.snapshot(),
                 started=None, press=False, pressed_until=0., demo_at=None, demo_move=None)
    viewer_keys = []

    def key_callback(keycode):
        # Runs on the viewer thread: only record the key.
        viewer_keys.append(keycode)

    def lines():
        color = (255, 255, 255)
        phase = state["phase"]
        result = [(state["message"], (60, 220, 255))]
        if phase == "HUMAN":
            if hand.move is not None:
                status = f"you played {hand.move.uci()} - locked until DONE"
            elif hand.held is not None:
                status = f"holding piece from {chess.square_name(hand.held_from)}"
            else:
                status = "click one of your pieces"
            result += [(f"YOUR MOVE ({human_name}) - {status}", (80, 255, 120)),
                       ("Click your piece, then a green dot.  Click it again to put it back.  U: undo", color),
                       (f"Q/R/B/N: promotion piece (now {hand.promotion}).  Esc: quit", color),
                       ("Then press the red button: here, Space/Enter, or double-click it in 3D", color)]
            if state["options"]:
                result[0] = ("Ambiguous - press a number: " +
                             "  ".join(f"{i+1}) {o}" for i, o in enumerate(state["options"])), (60, 180, 255))
        elif phase == "PAUSED":
            result += [("Paused: the board disagrees with the confirmed game.", (90, 90, 255)),
                       ("L: reload the confirmed position.  Esc: quit", color)]
        elif phase == "OVER":
            result += [("Game over.  Esc: quit", color)]
        else:
            result += [("Arm's turn - please keep your hands off the board", color)]
        return result

    def redraw(obs=None):
        if obs is not None:
            state["latest"] = obs
        pressed = time.perf_counter() < state["pressed_until"]
        shown = (state["phase"], state["message"])
        if shown != state.get("shown"):
            # Mirror the panel in the console, which helps when windows overlap.
            print(f"[{shown[0]}] {shown[1]}", flush=True)
            state["shown"] = shown
        board_window(state["latest"], hand, lines(), state["phase"] == "HUMAN", pressed)
        camera_window(sim, f"{state['latest'].status}: {state['latest'].reason}")

    def click(event, x, y, flags, param):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            print(f"click at ({x}, {y}) during {state['phase']}", flush=True)
        if state["phase"] != "HUMAN":
            return
        if event == cv2.EVENT_LBUTTONDOWN and np.hypot(x-BUTTON_CENTER[0], y-BUTTON_CENTER[1]) < BUTTON_RADIUS:
            state["press"] = True
        elif event == cv2.EVENT_LBUTTONDOWN and 0 <= x < BOARD and 0 <= y < BOARD:
            try:
                state["message"] = hand.click(chess.square(x//100, 7-y//100))
            except ValueError as error:
                state["message"] = str(error)
            state["options"] = []
            state["live_at"] = time.perf_counter()+IDLE_REFRESH

    def observing(obs):
        redraw(obs)
        viewer.sync()
        if cv2.waitKey(1) & 255 == 27:
            raise InterruptedError("Stopped by player")

    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=key_callback) as viewer:
        viewer.cam.lookat = [.25, -.13, .02]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.1, 90, -55
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        viewer.sync()
        redraw()
        cv2.setMouseCallback(TITLE+" - board", click)
        last_camera = [0.]

        def during_motion():
            if time.perf_counter()-last_camera[0] > .3:
                camera_window(sim, "ROBOT EXECUTING - waiting for final verification")
                if cv2.waitKey(1) & 255 == 27:
                    raise InterruptedError("Stopped by player")
                last_camera[0] = time.perf_counter()
        sim.robot.planner.viewer_callback = during_motion
        steps = max(1, round(.03/sim.model.opt.timestep))
        state["live_at"] = float("inf")

        while viewer.is_running():
            phase = state["phase"]
            if phase == "READ":
                try:
                    read_board(sim, game, observing)
                    state["message"] = "Camera read the board: it matches the confirmed position."
                    print(state["message"], flush=True)
                    state["phase"] = "ARM" if game.board.turn == arm_color else "START_HUMAN"
                except ValueError as error:
                    state["message"], state["phase"] = str(error), "PAUSED"
                continue
            if phase == "ARM":
                try:
                    state["message"] = "Stockfish is thinking..."
                    redraw()
                    cv2.waitKey(1)
                    move = engine.choose_move(game.board, args.depth)
                    state["message"] = f"Arm is playing {move}..."
                    redraw()
                    play_turn(sim, game, move.uci(), animate=not args.no_animation, viewer=viewer,
                              speed=args.speed, observer=observing)
                    state["message"] = f"Arm played {move} (camera verified)."
                    print(f"Arm {move}: camera verified, version {game.version}", flush=True)
                    state["phase"] = "OVER" if game.board.is_game_over(claim_draw=True) else "START_HUMAN"
                except (ValueError, TimeoutError, chess.engine.EngineError) as error:
                    state["message"], state["phase"] = str(error), "PAUSED"
                    print(error)
                if state["phase"] == "OVER":
                    state["message"] = "Game over: " + game.board.result(claim_draw=True)
                continue
            if phase == "START_HUMAN":
                try:
                    state["latest"], state["started"] = start_human_turn(sim, game, hand, observing)
                    state["options"], state["phase"] = [], "HUMAN"
                    state["demo_at"] = time.perf_counter()+1. if args.demo else None
                except ValueError as error:
                    state["message"], state["phase"] = str(error), "PAUSED"
                continue

            key = cv2.waitKey(30) & 255
            selection = None
            while viewer_keys:
                code = viewer_keys.pop(0)
                if code in (257, 335):  # GLFW Enter / keypad Enter in the 3D viewer.
                    key = 13
            if key == 27:
                break
            selected = viewer.perturb.select
            if selected > 0:
                # Pieces move only through the rule-checked hand, never by 3D drags;
                # selecting the red button is how the 3D view presses it.
                with viewer.lock():
                    viewer.perturb.select, viewer.perturb.active = 0, 0
            if phase == "HUMAN":
                # Step physics so the button springs back and pieces settle.
                for _ in range(steps):
                    mujoco.mj_step(sim.model, sim.data)
                if selected in sim.button_ids or sim.button_down():
                    state["press"] = True
                if key in (13, 10, 32):
                    state["press"] = True
                elif key == ord("u"):
                    state["message"], state["options"] = hand.undo(), []
                    state["live_at"] = time.perf_counter()+IDLE_REFRESH
                elif key != 255 and chr(key) in "qrbn":
                    hand.promotion = {"q": "queen", "r": "rook", "b": "bishop", "n": "knight"}[chr(key)]
                    state["message"] = f"Promotion piece set to {hand.promotion}"
                elif key != 255 and chr(key).isdigit() and 0 < int(chr(key)) <= len(state["options"]):
                    selection = state["options"][int(chr(key))-1]
                    state["press"] = True
                if args.demo and state["demo_at"] is not None and time.perf_counter() > state["demo_at"]:
                    if state["demo_move"] is None:
                        if not scripted and auto[0] <= 0:
                            # Demo finished: the real player takes over the hand.
                            state["demo_at"] = None
                            if args.exit_after_demo:
                                break
                            continue
                        if scripted:
                            move = chess.Move.from_uci(scripted.pop(0))
                        else:
                            auto[0] -= 1
                            move = engine.choose_move(game.board, args.depth)
                        hand.perform(game.board, move)
                        state["demo_move"], state["demo_at"] = move, time.perf_counter()+1.
                        state["message"] = f"Scripted hand moved pieces ({move})"
                        state["live_at"] = time.perf_counter()
                    else:
                        # The scripted player answers any ambiguity prompt itself.
                        state["press"], state["demo_at"] = True, None
                        selection = state["demo_move"].uci()
                if state["press"] and hand.move is None:
                    state["press"] = False
                    state["message"] = ("Put the piece down on a green dot first" if hand.held is not None
                                        else "Make your move first, then press DONE")
                elif state["press"]:
                    state["press"] = False
                    state["pressed_until"] = time.perf_counter()+.4
                    # Two captures from one square look identical to the camera;
                    # the player answers that prompt with the move they made.
                    selection = selection or hand.move.uci()
                    if sim.button_ids:
                        sim.data.qpos[sim.button_qpos] = -.012
                    try:
                        state["message"] = "Button pressed - camera is reading your move..."
                        redraw()
                        cv2.waitKey(1)
                        move, obs = press_button(sim, game, hand, state["started"], selection, observing)
                        state["message"], state["options"], state["demo_move"] = (
                            f"Camera read your move: {move}.", [], None)
                        print(f"Human {move}: read by camera after button, version {game.version}", flush=True)
                        state["phase"] = "OVER" if game.board.is_game_over(claim_draw=True) else "ARM"
                        if state["phase"] == "OVER":
                            state["message"] = "Game over: " + game.board.result(claim_draw=True)
                    except AmbiguousMove as error:
                        state["options"] = error.options
                    except ValueError as error:
                        state["message"], state["options"] = str(error) + " - press U, redo your move, press DONE.", []
                        print(error)
                    continue
                if time.perf_counter() > state["live_at"]:
                    # Perception takes most of a second, so refresh the camera's
                    # board view once the player pauses, not while clicking.
                    # Acceptance always uses a fresh button snapshot.
                    state["latest"] = sim.detector.process(sim.frame())
                    state["live_at"] = float("inf")
            elif phase == "PAUSED" and key == ord("l"):
                sim.load_position(game.board)
                # Reset is an explicit restart: identities cannot be
                # reconstructed from binary occupancy alone.
                game = Coordinator(sim.detector, game.board)
                game.rack = sim.rack_inventory.copy()
                engine.new_game()
                state["message"], state["phase"] = "Reloaded the confirmed position.", "READ"
            redraw()
            viewer.sync()


if __name__ == "__main__":
    main()
