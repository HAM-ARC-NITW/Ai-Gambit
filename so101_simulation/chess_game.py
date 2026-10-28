"""Rules and turn coordination. Perception never imports this module.

python-chess supplies the chess rules; Stockfish chooses moves through the
separate engine adapter. The confirmed board is
changed in exactly one place, after camera verification succeeds.
"""
import chess

UNKNOWN_STORAGE = 255


def occupancy(board):
    return [int(board.piece_at(i) is not None) for i in range(64)]


class AmbiguousMove(ValueError):
    """Several legal moves explain the camera image; someone must choose."""
    def __init__(self, options):
        super().__init__("Ambiguous occupancy; choose one of: " + ", ".join(options))
        self.options = options


def changed_squares(board, observation):
    return [chess.square_name(i) for i, (a, b) in enumerate(zip(occupancy(board), observation.occupancy)) if a != b]


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

    def read_human_move(self, obs, selection=None):
        """Infer an undeclared human move from the camera alone.

        Board occupancy picks the legal candidates. Storage changes must agree:
        a capture may add one piece to the tray (or none, if it left the
        workspace), and a promotion may take one spare of the promoted type.
        Returns the move plus the observed rack/tray to commit with it.
        """
        if obs.status != "READY" or not obs.stable:
            raise ValueError(f"Camera is not ready ({obs.status}: {obs.reason}); clear the board and press again")
        options = candidates(self.board, obs)
        if not options:
            changed = changed_squares(self.board, obs)
            raise ValueError("No legal move matches the board. Changed squares: " + (", ".join(changed) or "none"))
        if UNKNOWN_STORAGE in obs.rack + obs.tray:
            raise ValueError("Rack or capture tray is not clearly visible")
        if any(a and not b for a, b in zip(self.tray, obs.tray)):
            raise ValueError("A piece was taken out of the capture tray")
        if any(b and not a for a, b in zip(self.rack, obs.rack)):
            raise ValueError("A piece was returned to the promotion rack")
        tray_added = sum(b and not a for a, b in zip(self.tray, obs.tray))
        rack_taken = [i for i, (a, b) in enumerate(zip(self.rack, obs.rack)) if a and not b]
        side = 0 if self.board.turn == chess.WHITE else 1
        consistent = []
        for uci in options:
            move = chess.Move.from_uci(uci)
            if tray_added > int(self.board.is_capture(move)) + int(bool(move.promotion)):
                continue
            if len(rack_taken) > int(bool(move.promotion)):
                continue
            # Rack slots are labelled, so a taken spare identifies the promotion.
            if rack_taken and (rack_taken[0] // 16 != side or
                               self.detector.config["rack_types"][rack_taken[0]] != chess.piece_name(move.promotion)):
                continue
            consistent.append(uci)
        if not consistent:
            raise ValueError("Board matches " + ", ".join(options) + ", but the tray/rack changes do not")
        if len(consistent) > 1:
            if selection not in consistent:
                raise AmbiguousMove(consistent)
            return chess.Move.from_uci(selection), list(obs.rack), list(obs.tray)
        return chess.Move.from_uci(consistent[0]), list(obs.rack), list(obs.tray)

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
