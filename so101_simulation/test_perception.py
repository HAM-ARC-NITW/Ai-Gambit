"""Contract, rules and real rendered-image tests. Run with pytest explicitly;
older diagnostic scripts in this folder are not pytest tests.
The full adversarial rendered suite is run_perception.py --headless.
"""
import copy
import time
import numpy as np
import pytest
import chess
from perception_sim import Simulation
from ai_gambit_perception.perception import Observation, Perception, SQUARES, transform
from chess_game import Coordinator, occupancy, candidates
from run_perception import CASTLE, EP, PROMOTE, CAPTURE
from play_chess import play_turn
from engine_stuff.stockfish_engine import StockfishEngine


def observation(board, sequence=2):
    return Observation(sequence,time.time(),status="READY",occupancy=occupancy(board),
                       confidence=[.95]*64,rack=[1]*32,tray=[0]*48,stable=True)


@pytest.mark.parametrize("fen,uci",[(chess.STARTING_FEN,"e2e4"),(chess.STARTING_FEN,"g1f3"),
                                  (CASTLE,"e1g1"),(CASTLE,"e1c1"),(EP,"e5d6"),
                                  (PROMOTE,"a7a8q"),(PROMOTE,"a7a8r"),(PROMOTE,"a7a8b"),(PROMOTE,"a7a8n")])
def test_legal_outcome_matching(fen,uci):
    board = chess.Board(fen)
    final = board.copy()
    final.push_uci(uci)
    assert uci in candidates(board,observation(final))
    assert board.fen() == fen  # Candidate search must not mutate confirmed state.


def test_capture_and_promotion_are_explicitly_ambiguous():
    board = chess.Board(CAPTURE)
    final = board.copy()
    final.push_uci("e4d5")
    assert set(candidates(board,observation(final))) == {"e4d5","e4f5"}
    board = chess.Board(PROMOTE)
    final = board.copy()
    final.push_uci("a7a8q")
    assert len(candidates(board,observation(final))) == 4
    game = Coordinator(Perception(),board)
    with pytest.raises(ValueError,match="Ambiguous"):
        game.interpret(observation(final))
    assert game.interpret(observation(final),"a7a8n")[0].uci() == "a7a8n"


@pytest.mark.parametrize("status",["MOVING","OCCLUDED","CAMERA_FAILURE","CALIBRATION_INVALID","ROBOT_EXECUTING"])
def test_uncertain_scene_cannot_be_interpreted(status):
    before,after = chess.Board(),chess.Board()
    after.push_uci("e2e4")
    obs = observation(after)
    obs.status = status
    assert candidates(before,obs) == []


def test_unknown_is_not_a_wildcard():
    before,after = chess.Board(),chess.Board()
    after.push_uci("e2e4")
    obs = observation(after)
    obs.occupancy[chess.E4] = 255
    assert not candidates(before,obs)


def test_low_confidence_cannot_be_interpreted():
    before,after = chess.Board(),chess.Board()
    after.push_uci("e2e4")
    obs = observation(after)
    obs.confidence[chess.E4] = .2
    assert not candidates(before,obs)


@pytest.mark.parametrize("fault",["board","rack","tray","version","time","sequence","stale","unstable"])
def test_no_commit_on_failed_verification(fault):
    detector, board = Perception(),chess.Board()
    game = Coordinator(detector,board)
    assert game.baseline(observation(board,1))[0]
    before_fen = game.board.fen()
    after = board.copy()
    after.push_uci("e2e4")
    obs = observation(after)
    started = time.time()-.05
    if fault == "board": obs.occupancy[chess.E4] = 0
    if fault == "rack": obs.rack[0] = 0
    if fault == "tray": obs.tray[0] = 1
    if fault == "version": game.version = 2
    if fault == "time": started = time.time()+1
    if fault == "sequence": obs.observation_id = 1
    if fault == "stale": obs.timestamp -= 10
    if fault == "unstable": obs.stable = False
    assert not game.accept(chess.Move.from_uci("e2e4"),obs,started,[1]*32,[0]*48)[0]
    assert game.board.fen() == before_fen


def test_success_commits_once_and_requires_new_baseline():
    game = Coordinator(Perception())
    assert game.baseline(observation(game.board,1))[0]
    final = game.board.copy()
    final.push_uci("e2e4")
    obs = observation(final)
    assert game.accept(chess.Move.from_uci("e2e4"),obs,time.time()-.05,[1]*32,[0]*48)[0]
    assert game.version == 1
    # Even another legal move needs a baseline for the new version.
    final.push_uci("e7e5")
    assert not game.accept(chess.Move.from_uci("e7e5"),observation(final,3),time.time()-.05,[1]*32,[0]*48)[0]


def test_invalid_baseline_clears_previous_authorization():
    game = Coordinator(Perception())
    assert game.baseline(observation(game.board,1))[0]
    wrong = observation(game.board,2)
    wrong.occupancy[chess.E2] = 0
    assert not game.baseline(wrong)[0]
    assert game.detector.baseline is None


def test_camera_timestamp_contract():
    detector = Perception()
    assert detector.process(None).status == "CAMERA_FAILURE"
    shape = (detector.config["image_height"],detector.config["image_width"],3)
    image = np.zeros(shape,np.uint8)
    assert detector.process(image,time.time()-10).status == "CAMERA_FAILURE"
    assert detector.process(image,time.time()+10).status == "CAMERA_FAILURE"
    detector.last_timestamp = time.time()+1
    assert detector.process(image).status == "CAMERA_FAILURE"
    detector.last_timestamp = -1
    assert detector.process(image[:100]).status == "CAMERA_FAILURE"


@pytest.fixture(scope="module")
def sim():
    instance = Simulation()
    yield instance
    instance.close()


def test_rendered_calibration_and_orientation(sim):
    sim.load_position(chess.Board())
    obs = sim.snapshot()
    assert obs.status == "READY"
    assert obs.occupancy == occupancy(chess.Board())
    assert obs.rack == [1]*32 and obs.tray == [0]*48
    assert len(obs.marker_ids) == 12
    actual = np.array(obs.poses["board"])
    nominal = transform(sim.config["board_rotation"],sim.config["board_origin"])
    # Four corners plus center and edge square centers, compared to independent
    # fixture dimensions, not coordinates fed into the RGB detector at runtime.
    for x,y in ((0,0),(.24,0),(0,.24),(.24,.24),(.015,.015),(.135,.105),(.225,.225)):
        assert np.linalg.norm((actual-nominal) @ [x,y,0,1]) < .004
    assert SQUARES[0] == "a1" and SQUARES[-1] == "h8"


def test_knight_overhang_belongs_to_its_own_square(sim):
    # In perspective a black knight's head on f2 reaches into f1. That sliver
    # once made the rook's square UNKNOWN and blocked a legal Nxf2.
    board = chess.Board("4k3/8/8/8/8/8/5n2/5RK1 w - - 0 1")
    sim.load_position(board)
    obs = sim.snapshot()
    assert obs.status == "READY" and obs.occupancy == occupancy(board)


def test_stable_frames_are_required(sim):
    sim.load_position(chess.Board())
    sim.detector.previous,sim.detector.stable_count = None,0
    statuses = [sim.detector.process(sim.frame()).status for _ in range(3)]
    assert statuses == ["MOVING","MOVING","READY"]


def test_rendered_game_round_trip(sim):
    sim.load_position(chess.Board())
    game = Coordinator(sim.detector)
    play_turn(sim,game,"e2e4")
    engine = StockfishEngine(time_limit=.05)
    try:reply = engine.choose_move(game.board,depth=1)
    finally:engine.close()
    assert reply in game.board.legal_moves
    play_turn(sim,game,reply.uci())
    assert game.version == 2
    assert sim.snapshot().occupancy == occupancy(game.board)


@pytest.mark.parametrize("fen,uci,missing",[(CAPTURE,"e4d5","tray"),(PROMOTE,"a7a8q","rack")])
def test_inventory_failure_does_not_start_an_action(sim,fen,uci,missing):
    board = chess.Board(fen)
    sim.load_position(board)
    if missing == "tray":
        sim.tray_inventory = [1]*48
    else:
        sim.rack_inventory = [0]*32
    positions = sim.data.qpos.copy()
    with pytest.raises(ValueError):
        sim.physical_move(board,chess.Move.from_uci(uci))
    assert np.array_equal(sim.data.qpos,positions)


def test_loading_extra_promoted_piece_uses_inventory(sim):
    board = chess.Board("4k3/8/8/8/8/8/QQ6/4K3 w - - 0 1")
    sim.load_position(board)
    assert sum(sim.rack_inventory) == 31
    obs = sim.snapshot()
    assert obs.status == "READY" and obs.occupancy == occupancy(board)
    assert obs.rack == sim.rack_inventory
