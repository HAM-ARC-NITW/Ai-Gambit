"""End-to-end turn loop: the human's move is never declared, only read by the
camera after the big red button. Run with: python -m pytest test_game_loop.py
"""
import time
import pytest
import chess
import mujoco
from perception_sim import Simulation
from chess_coords import PIECE_REST_Z, square_to_pos
from ai_gambit_perception.perception import Observation, Perception
from chess_game import Coordinator, AmbiguousMove, occupancy
from play_chess import play_turn
from play_game import start_human_turn, press_button
from virtual_hand import VirtualHand

CAPTURE = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"
TWO_CAPTURES = "rnbqkbnr/ppp1pppp/8/3p4/2P1P3/8/PP1P1PPP/RNBQKBNR b KQkq - 0 2"
CASTLE = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"
EP = "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1"
PROMOTE = "4k3/8/8/8/8/8/p7/4K3 b - - 0 1"


@pytest.fixture(scope="module")
def sim():
    instance = Simulation(button=True)
    yield instance
    instance.close()


def setup(sim, fen):
    board = chess.Board(fen)
    sim.load_position(board)
    game = Coordinator(sim.detector, board)
    game.rack = sim.rack_inventory.copy()
    hand = VirtualHand(sim)
    _, started = start_human_turn(sim, game, hand)
    return game, hand, started


def test_button_is_outside_the_camera_view(sim):
    sim.load_position(chess.Board())
    sim.data.qpos[sim.button_qpos] = -.012
    obs = sim.snapshot()
    assert obs.status == "READY" and obs.occupancy == occupancy(chess.Board())
    assert sim.button_down()


@pytest.mark.parametrize("fen,uci", [(CAPTURE, "d5e4"), (CASTLE, "e8g8"), (CASTLE, "e8c8"),
                                     (EP, "d4e3"), (PROMOTE, "a2a1n"), (PROMOTE, "a2a1q")])
def test_camera_reads_undeclared_human_move(sim, fen, uci):
    game, hand, started = setup(sim, fen)
    move = chess.Move.from_uci(uci)
    hand.perform(game.board, move)
    played, obs = press_button(sim, game, hand, started)
    assert played == move and game.version == 1
    assert obs.occupancy == occupancy(game.board)
    # The arm must then grasp the bodies the human actually left behind.
    reply = next(iter(game.board.legal_moves))
    play_turn(sim, game, reply.uci())
    assert game.version == 2


def test_arm_animates_capture_of_piece_the_human_moved(sim):
    game, hand, started = setup(sim, CAPTURE)
    hand.perform(game.board, chess.Move.from_uci("d5e4"))
    press_button(sim, game, hand, started)
    play_turn(sim, game, "d2d3")
    _, started = start_human_turn(sim, game, hand)
    hand.perform(game.board, chess.Move.from_uci("e4d3"))
    press_button(sim, game, hand, started)
    play_turn(sim, game, "c2d3", animate=True)  # Arm captures the hand-moved pawn.
    assert game.version == 4
    assert sim.snapshot().occupancy == occupancy(game.board)


def test_arm_reaches_d3_from_f1(sim):
    # IK seeded from the f1 pick once settled a square short of d3, so the arm
    # set the bishop down from the wrong place and knocked it over on retract.
    board = chess.Board("4k3/8/8/8/8/8/8/4KB2 w - - 0 1")
    sim.load_position(board)
    game = Coordinator(sim.detector, board)
    game.rack = sim.rack_inventory.copy()
    play_turn(sim, game, "f1d3", animate=True)
    assert game.version == 1


def test_expired_observation_is_retaken_not_trusted(sim, monkeypatch):
    # A busy laptop can process a frame for longer than max_frame_age.
    board = chess.Board()
    sim.load_position(board)
    game = Coordinator(sim.detector, board)
    game.rack = sim.rack_inventory.copy()
    real_snapshot, real_move = sim.snapshot, sim.physical_move
    aged, moves = {"left": 2}, []

    def slow_snapshot(*args, **kwargs):
        obs = real_snapshot(*args, **kwargs)
        if aged["left"]:  # The first baseline and the first post-move check.
            aged["left"] -= 1
            obs.timestamp -= 10
        return obs

    monkeypatch.setattr(sim, "snapshot", slow_snapshot)
    monkeypatch.setattr(sim, "physical_move", lambda *a, **k: moves.append(1) or real_move(*a, **k))
    play_turn(sim, game, "e2e4")
    assert game.version == 1 and len(moves) == 1  # The arm moved once.
    aged["left"] = 99  # An always-expired camera is still refused.
    with pytest.raises(ValueError, match="expired"):
        play_turn(sim, game, "e7e5")
    assert game.version == 1


def test_ambiguous_capture_asks_and_then_commits_choice(sim):
    game, hand, started = setup(sim, TWO_CAPTURES)
    hand.perform(game.board, chess.Move.from_uci("d5c4"))
    with pytest.raises(AmbiguousMove) as error:
        press_button(sim, game, hand, started)
    assert set(error.value.options) == {"d5c4", "d5e4"} and game.version == 0
    played, _ = press_button(sim, game, hand, started, selection="d5c4")
    assert played.uci() == "d5c4"
    play_turn(sim, game, next(iter(game.board.legal_moves)).uci())


def test_hand_follows_the_rules_and_locks_after_one_move(sim):
    game, hand, started = setup(sim, chess.STARTING_FEN.replace(" w ", " b "))
    assert "not your piece" in hand.click(chess.E2)
    assert "no legal move" in hand.click(chess.A8)
    hand.click(chess.E7)
    assert "Illegal" in hand.click(chess.E3) and hand.move is None
    # Clicking another own piece switches to it; it never captures its own colour.
    assert "from d7" in hand.click(chess.D7)
    assert sim.body_at(chess.E7) is not None and hand.held_from == chess.D7
    assert "no legal move" in hand.click(chess.D8) and hand.held is None
    assert sim.body_at(chess.D7) is not None
    hand.click(chess.E7)
    hand.click(chess.E5)
    assert hand.move.uci() == "e7e5"
    assert "already played" in hand.click(chess.D7) and hand.held is None
    hand.undo()  # Takes the whole move back.
    assert hand.move is None and sim.body_at(chess.E7) is not None and sim.body_at(chess.E5) is None
    hand.click(chess.G8)
    hand.click(chess.F6)
    played, _ = press_button(sim, game, hand, started)
    assert played.uci() == "g8f6"


def test_camera_rejects_a_knocked_board(sim):
    game, hand, started = setup(sim, chess.STARTING_FEN.replace(" w ", " b "))
    fen = game.board.fen()
    hand.perform(game.board, chess.Move.from_uci("e7e5"))
    # Something the hand did not do: a piece knocked onto another square.
    target = square_to_pos("a5")
    target[2] = PIECE_REST_Z["pawn"]
    sim.put(sim.body_at(chess.A7), target)
    with pytest.raises(ValueError, match="No legal move"):
        press_button(sim, game, hand, started)
    assert game.board.fen() == fen and game.version == 0


def test_held_piece_stays_visible_and_can_be_put_back(sim):
    game, hand, started = setup(sim, chess.STARTING_FEN.replace(" w ", " b "))
    hand.click(chess.E7)
    held = hand.held
    assert sim.data.xpos[held][2] > .05  # Floating above e7, not hidden.
    for _ in range(200):
        mujoco.mj_step(sim.model, sim.data)
    assert sim.data.xpos[held][2] > .05  # Physics does not drop a held piece.
    hand.click(chess.E7)  # Put it back where it came from.
    assert hand.held is None and sim.body_at(chess.E7) == held
    hand.click(chess.E7)
    hand.click(chess.E5)
    played, _ = press_button(sim, game, hand, started)
    assert played.uci() == "e7e5"


def test_button_refused_while_holding_a_piece(sim):
    game, hand, started = setup(sim, chess.STARTING_FEN.replace(" w ", " b "))
    hand.click(chess.G8)
    with pytest.raises(ValueError, match="holding"):
        press_button(sim, game, hand, started)
    assert game.version == 0


def observation(board, rack=None, tray=None):
    return Observation(2, time.time(), status="READY", occupancy=occupancy(board), confidence=[.95]*64,
                       rack=rack or [1]*32, tray=tray or [0]*48, stable=True)


def test_storage_changes_must_agree_with_the_move():
    board = chess.Board()
    after = board.copy()
    after.push_uci("e2e4")
    game = Coordinator(Perception(), board)
    with pytest.raises(ValueError, match="tray/rack"):
        game.read_human_move(observation(after, tray=[1]+[0]*47))  # Quiet move, new tray piece.
    game.tray = [1]+[0]*47
    with pytest.raises(ValueError, match="taken out"):
        game.read_human_move(observation(after))
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    after = board.copy()
    after.push_uci("a7a8r")
    game = Coordinator(Perception(), board)
    rack = [1]*32
    rack[1] = 0  # Slot 1 is labelled rook, so the promotion is not ambiguous.
    move, _, _ = game.read_human_move(observation(after, rack=rack, tray=[1]+[0]*47))
    assert move.uci() == "a7a8r"
