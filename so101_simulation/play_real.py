"""Play chess against the real SO-101 through LeRobot.

The arm plays White and moves first unless --human white is given. You move the
physical pieces yourself, then click DONE (or press Space): the overhead camera
alone works out your move, exactly as in the simulated game. Run the
real_setup.py steps once before the first game (README: "Real arm with LeRobot").

--rehearsal runs this same program without hardware: a stand-in robot with
LeRobot's SO-101 interface drives a simulated table, whose pieces you move by
clicking. --rehearsal --headless also scripts your moves.
"""
import argparse
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import chess
import chess.engine
from chess_game import Coordinator, AmbiguousMove
from lerobot_arm import ArmError, LeRobotArm
from play_chess import play_turn
from play_game import (BOARD, BUTTON_CENTER, BUTTON_RADIUS, TITLE, board_window, camera_window,
                       headless, press_button, read_board, start_human_turn)
from real_workspace import SETUP, RealWorkspace, calibration_from_setup, connect_arm, load_setup
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine_stuff.stockfish_engine import StockfishEngine

LIVE_REFRESH = 1.  # Seconds between camera previews while you move pieces.


class PhysicalPlayer:
    """The person moves real pieces; nothing is clicked on the board image."""
    held = held_from = move = None
    targets = []
    promotion = "queen"

    def begin(self, board, rack, tray):
        pass

    def click(self, square):
        return "Move the real pieces on the table, then click DONE"

    def undo(self):
        return "Put the pieces back by hand if you changed your mind"


def open_workspace(args, board):
    """The real arm and camera, or the hardware-free stand-in for --rehearsal."""
    if args.rehearsal:
        from lerobot_standin import StandInSO101, standin_calibration
        from perception_sim import Simulation
        from virtual_hand import VirtualHand
        world = Simulation(button=True)
        world.load_position(board)
        settings = {"realtime": not args.headless}
        if args.slowdown:
            settings["slowdown"] = args.slowdown
        arm = LeRobotArm(StandInSO101(world), settings)
        arm.connect()
        return RealWorkspace(arm, standin_calibration(world)), VirtualHand(world)
    setup = load_setup(args.setup)
    calibration = calibration_from_setup(setup)  # Fail before touching the hardware.
    if args.slowdown:
        setup.setdefault("arm", {})["slowdown"] = args.slowdown
    arm = connect_arm(setup)
    try:
        return RealWorkspace(arm, calibration), PhysicalPlayer()
    except Exception:
        arm.disconnect()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--human", choices=("white", "black"), default="black",
                        help="Your colour (default black: the arm plays White and moves first)")
    parser.add_argument("--setup", default=str(SETUP), help="Calibration file written by real_setup.py")
    parser.add_argument("--rehearsal", action="store_true", help="No hardware: LeRobot stand-in on a simulated table")
    parser.add_argument("--headless", action="store_true", help="With --rehearsal: no windows, scripted moves")
    parser.add_argument("--auto-human", type=int, default=2, help="--headless: Stockfish plays this many of your moves")
    parser.add_argument("--slowdown", type=float, help="Real moves take this many times the simulated time")
    parser.add_argument("--engine", help="Path to the Stockfish executable (or set STOCKFISH_PATH)")
    parser.add_argument("--think-time", type=float, default=.5, help="Engine seconds per move (default .5)")
    parser.add_argument("--skill", type=int, choices=range(21), default=10, help="Stockfish skill 0..20")
    parser.add_argument("--depth", type=int, help="Optional search depth cap")
    args = parser.parse_args()
    if args.headless and not args.rehearsal:
        parser.error("--headless needs --rehearsal: a real game needs a person at the board")
    if args.slowdown is not None and (not np.isfinite(args.slowdown) or args.slowdown <= 0):
        parser.error("--slowdown must be finite and positive")
    arm_color = chess.BLACK if args.human == "white" else chess.WHITE
    try:
        engine = StockfishEngine(args.engine, args.think_time, args.skill)
    except (OSError, ValueError, TimeoutError, chess.engine.EngineError) as error:
        parser.error(str(error))
    print(f"Chess engine: {engine.name}; skill {args.skill}; think time {args.think_time}s", flush=True)
    board, workspace = chess.Board(), None
    try:
        workspace, hand = open_workspace(args, board)
        workspace.load_position(board)  # The twin's pieces; the camera checks the real ones.
        print("Arm connected through LeRobot; moving it clear of the camera...", flush=True)
        workspace.park()
        game = Coordinator(workspace.detector, board)
        game.rack = workspace.rack_inventory.copy()
        if not args.rehearsal:
            # Whatever spares you loaded into the rack are the ones the arm may
            # promote with; an unreadable slot stays UNKNOWN and stops the game.
            first = workspace.snapshot()
            if first.status == "READY":
                game.rack = workspace.rack_inventory = [int(v) for v in first.rack]
        if args.headless:
            headless(workspace, game, hand, engine, argparse.Namespace(
                human_moves=None, auto_human=args.auto_human, depth=args.depth, no_animation=False, speed=1),
                arm_color)
            return
        play(workspace, game, hand, engine, args, arm_color)
    except (KeyboardInterrupt, InterruptedError):
        print("Game stopped; no unverified turn committed.")
    except (ArmError, OSError, ValueError) as error:
        print("Stopped:", error)
        raise SystemExit(1)
    finally:
        try:
            if workspace is not None:
                print("Parking the arm and disconnecting...", flush=True)
                workspace.close()
        finally:
            engine.close()
            cv2.destroyAllWindows()


def play(ws, game, hand, engine, args, arm_color):
    human_name = "White" if arm_color == chess.BLACK else "Black"
    physical = isinstance(hand, PhysicalPlayer)
    state = dict(phase="READ", message="Reading the board...", options=[], press=False,
                 pressed_until=0., started=None, latest=ws.detector.process(ws.frame()), live_at=0.)

    def lines():
        white, phase = (255, 255, 255), state["phase"]
        result = [(state["message"], (60, 220, 255))]
        if phase == "HUMAN":
            how = ("Move the real pieces on the table" if physical else
                   "Click your piece, then a green dot.  U: undo.  Q/R/B/N: promotion piece")
            result += [(f"YOUR MOVE ({human_name})", (80, 255, 120)), (how, white),
                       ("Then click DONE, or press Space.  Esc: quit (the arm parks first)", white)]
            if state["options"]:
                result[0] = ("Ambiguous - press a number: " +
                             "  ".join(f"{i+1}) {o}" for i, o in enumerate(state["options"])), (60, 180, 255))
        elif phase == "PAUSED":
            result += [("Paused: the board disagrees with the confirmed game, or the arm stopped.", (90, 90, 255)),
                       ("Put the pieces back as in the confirmed game, then press L.  Esc: quit", white)]
        elif phase == "OVER":
            result += [("Game over.  Esc: quit", white)]
        else:
            result += [("Arm's turn - keep your hands clear of the arm and the board", white)]
        return result

    def redraw(obs=None):
        if obs is not None:
            state["latest"] = obs
        shown = (state["phase"], state["message"])
        if shown != state.get("shown"):
            print(f"[{shown[0]}] {shown[1]}", flush=True)
            state["shown"] = shown
        board_window(state["latest"], hand, lines(), state["phase"] == "HUMAN",
                     time.perf_counter() < state["pressed_until"])
        camera_window(ws, f"{state['latest'].status}: {state['latest'].reason}")

    def observing(obs):
        redraw(obs)
        if cv2.waitKey(1) & 255 == 27:
            raise InterruptedError("Stopped by player")

    def click(event, x, y, flags, param):
        if state["phase"] != "HUMAN" or event != cv2.EVENT_LBUTTONDOWN:
            return
        if np.hypot(x-BUTTON_CENTER[0], y-BUTTON_CENTER[1]) < BUTTON_RADIUS:
            state["press"] = True
        elif 0 <= x < BOARD and 0 <= y < BOARD:
            try:
                state["message"] = hand.click(chess.square(x//100, 7-y//100))
            except ValueError as error:
                state["message"] = str(error)
            state["options"] = []

    def stopped(error):
        state["message"], state["phase"] = str(error), "PAUSED"
        print(error, flush=True)
        if isinstance(error, ArmError):
            try:
                ws.retreat()  # Lift clear of the pieces before anyone touches the board.
            except ArmError as again:
                state["message"] = f"{error} -- and the arm could not retreat: {again}"

    redraw()
    cv2.setMouseCallback(TITLE+" - board", click)
    while True:
        phase = state["phase"]
        if phase == "READ":
            try:
                read_board(ws, game, observing)
                state["message"] = "Camera read the board: it matches the confirmed position."
                state["phase"] = "ARM" if game.board.turn == arm_color else "START_HUMAN"
            except ValueError as error:
                stopped(error)
            continue
        if phase == "ARM":
            try:
                state["message"] = "Stockfish is thinking..."
                redraw()
                cv2.waitKey(1)
                move = engine.choose_move(game.board, args.depth)
                state["message"] = f"Arm is playing {move}..."
                redraw()
                cv2.waitKey(1)
                play_turn(ws, game, move.uci(), animate=True, observer=observing)
                state["message"] = f"Arm played {move} (camera verified)."
                print(f"Arm {move}: camera verified, version {game.version}", flush=True)
                state["phase"] = "OVER" if game.board.is_game_over(claim_draw=True) else "START_HUMAN"
                if state["phase"] == "OVER":
                    state["message"] = "Game over: " + game.board.result(claim_draw=True)
            except (ValueError, TimeoutError, chess.engine.EngineError) as error:
                stopped(error)
            continue
        if phase == "START_HUMAN":
            try:
                state["latest"], state["started"] = start_human_turn(ws, game, hand, observing)
                state["options"], state["phase"] = [], "HUMAN"
            except ValueError as error:
                stopped(error)
            continue

        key = cv2.waitKey(30) & 255
        if key == 27:
            break
        selection = None
        if phase == "HUMAN":
            if key in (13, 10, 32):
                state["press"] = True
            elif key == ord("u"):
                state["message"], state["options"] = hand.undo(), []
            elif key != 255 and chr(key) in "qrbn" and not physical:
                hand.promotion = {"q": "queen", "r": "rook", "b": "bishop", "n": "knight"}[chr(key)]
                state["message"] = f"Promotion piece set to {hand.promotion}"
            elif key != 255 and chr(key).isdigit() and 0 < int(chr(key)) <= len(state["options"]):
                selection = state["options"][int(chr(key))-1]
                state["press"] = True
            if state["press"]:
                state["press"], state["pressed_until"] = False, time.perf_counter()+.4
                if not physical and hand.move is None:
                    state["message"] = "Make your move first, then press DONE"
                    continue
                selection = selection or (None if physical else hand.move.uci())
                try:
                    state["message"] = "DONE pressed - camera is reading your move..."
                    redraw()
                    cv2.waitKey(1)
                    move, _ = press_button(ws, game, hand, state["started"], selection, observing)
                    state["message"], state["options"] = f"Camera read your move: {move}.", []
                    print(f"Human {move}: read by camera after DONE, version {game.version}", flush=True)
                    state["phase"] = "OVER" if game.board.is_game_over(claim_draw=True) else "ARM"
                    if state["phase"] == "OVER":
                        state["message"] = "Game over: " + game.board.result(claim_draw=True)
                except AmbiguousMove as error:
                    state["options"] = error.options
                except ValueError as error:
                    state["message"], state["options"] = str(error) + " - fix the board, then press DONE.", []
                    print(error, flush=True)
                continue
            if time.perf_counter() > state["live_at"]:
                # A live preview while you move pieces; acceptance always uses
                # a fresh, stable snapshot after DONE.
                state["latest"] = ws.detector.process(ws.frame())
                state["live_at"] = time.perf_counter() + LIVE_REFRESH
        elif phase == "PAUSED" and key == ord("l"):
            state["message"], state["phase"] = "Re-reading the board...", "READ"
        redraw()


if __name__ == "__main__":
    main()
