"""Rules and turn coordination. Perception never imports this module.

python-chess supplies the chess rules; the small search below is an example
opponent until the team's custom engine is connected. The confirmed board is
changed in exactly one place, after camera verification succeeds.
"""
import time
import chess


def occupancy(board):
    return [int(board.piece_at(i) is not None) for i in range(64)]


def candidates(board, observation):
    """Compare complete legal outcomes. UNKNOWN must never act as a wildcard."""
    if observation.status != "READY" or not observation.stable or 255 in observation.occupancy or min(observation.confidence) < .7:
        return []
    result = []
    for move in board.legal_moves:
        possible = board.copy()
        possible.push(move)
        if occupancy(possible) == observation.occupancy:
            result.append(move.uci())
    return result


class Coordinator:
    def __init__(self, detector, board=None):
        self.detector = detector
        self.board = chess.Board() if board is None else board.copy()
        self.version = 0
        self.rack = [1]*32
        self.tray = [0]*48

    def baseline(self, obs):
        return self.detector.establish_baseline(obs, occupancy(self.board), self.version,
                                                rack=self.rack, tray=self.tray)

    def interpret(self, obs, selection=None):
        options = candidates(self.board, obs)
        if not options:
            raise ValueError("No legal move matches the clear observed board")
        if len(options) > 1:
            # Occupancy cannot distinguish several captures from the same source,
            # or Q/R/B/N promotions. Require an explicit player/coordinator choice.
            if selection not in options:
                raise ValueError("Ambiguous occupancy; choose one of: " + ", ".join(options))
            return chess.Move.from_uci(selection), options
        return chess.Move.from_uci(options[0]), options

    def accept(self, move, obs, operation_started, rack, tray):
        if move not in self.board.legal_moves:
            return False, "Requested move is illegal in the confirmed position"
        expected = self.board.copy()
        expected.push(move)
        ok, reason = self.detector.verify(obs, occupancy(expected), version=self.version,
                                         after=operation_started, rack=rack, tray=tray)
        if not ok:
            return False, reason
        self.board.push(move)
        self.version += 1
        self.rack, self.tray = list(rack), list(tray)
        return True, reason


def choose_move(board, depth=2):
    """Small, deterministic negamax opponent, not a competitive chess engine.
    Scores are from the side-to-move's perspective. All moves come from the rules
    library, including check evasions, castling, en passant and promotion.
    """
    values = {chess.PAWN:100, chess.KNIGHT:320, chess.BISHOP:330,
              chess.ROOK:500, chess.QUEEN:900, chess.KING:0}

    def evaluate(position):
        if position.is_checkmate():
            return -100000
        if position.is_game_over(claim_draw=True):
            return 0
        score = 0
        for square, piece in position.piece_map().items():
            f, r = chess.square_file(square), chess.square_rank(square)
            center = 7-abs(3.5-f)-abs(3.5-r)
            bonus = center*4 if piece.piece_type != chess.KING else 0
            score += (values[piece.piece_type]+bonus) * (1 if piece.color == position.turn else -1)
        return score

    def search(position, remaining, alpha, beta):
        if remaining == 0 or position.is_game_over(claim_draw=True):
            return evaluate(position)
        best = -float("inf")
        moves = sorted(position.legal_moves, key=lambda m: (not position.is_capture(m), m.uci()))
        for move in moves:
            position.push(move)
            score = -search(position, remaining-1, -beta, -alpha)
            position.pop()
            best, alpha = max(best, score), max(alpha, score)
            if alpha >= beta:
                break
        return best

    choices = []
    for move in sorted(board.legal_moves, key=lambda m:m.uci()):
        trial = board.copy()
        trial.push(move)
        choices.append((-search(trial, depth-1, -float("inf"), float("inf")), move))
    return max(choices, key=lambda item:item[0])[1] if choices else None
