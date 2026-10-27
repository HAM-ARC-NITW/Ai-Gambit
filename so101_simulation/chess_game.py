"""Rules and turn coordination. Perception never imports this module.

python-chess supplies the chess rules; Stockfish chooses moves through the
separate engine adapter. The confirmed board is
changed in exactly one place, after camera verification succeeds.
"""
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
