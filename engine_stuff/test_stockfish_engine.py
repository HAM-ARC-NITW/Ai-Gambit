"""Run with the installed engine: python -m pytest engine_stuff/test_stockfish_engine.py -q."""
import chess
import pytest
from engine_stuff.stockfish_engine import StockfishEngine,find_stockfish


@pytest.fixture(scope="module")
def engine():
    instance = StockfishEngine(time_limit=.1,skill=20)
    yield instance
    instance.close()


@pytest.mark.parametrize("fen",[
    chess.STARTING_FEN,
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",  # Castling rights.
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",     # En passant available.
    "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",        # Promotion available.
    "4r1k1/8/8/8/8/8/8/4K3 w - - 0 1",       # Must escape check.
])
def test_real_engine_returns_legal_move_without_mutating_board(engine,fen):
    board = chess.Board(fen)
    original = board.fen(),board.move_stack.copy()
    move = engine.choose_move(board)
    assert move in board.legal_moves
    assert (board.fen(),board.move_stack)==original


def test_real_engine_finds_mate_in_one(engine):
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    move = engine.choose_move(board,depth=8)
    board.push(move)
    assert board.is_checkmate()


@pytest.mark.parametrize("fen",["7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
                                "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"])
def test_terminal_positions_have_no_reply(engine,fen):
    assert engine.choose_move(chess.Board(fen)) is None


def test_confirmed_history_and_draw_are_preserved(engine):
    board = chess.Board()
    for move in ["g1f3","g8f6","f3g1","f6g8"]*2:board.push_uci(move)
    history = board.move_stack.copy()
    assert board.can_claim_threefold_repetition()
    assert engine.choose_move(board) is None and board.move_stack==history


def test_explicit_missing_engine_does_not_fall_back(tmp_path):
    with pytest.raises(FileNotFoundError):find_stockfish(tmp_path/"missing.exe")


@pytest.mark.parametrize("seconds",[0,-1,float("nan"),float("inf")])
def test_invalid_thinking_time_fails_before_starting_process(seconds):
    with pytest.raises(ValueError):StockfishEngine(time_limit=seconds)


def test_invalid_depth_and_position_are_refused(engine):
    with pytest.raises(ValueError):engine.choose_move(chess.Board(),depth=0)
    with pytest.raises(ValueError):engine.choose_move(chess.Board(None))
