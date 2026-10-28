"""The player's hand in simulation: pick up and set down pieces.

The hand follows the rules of the confirmed position: it lifts only the player's
own pieces, sets them down only on legal squares, and locks after one complete
move until the big red button (U takes the move back). It completes castling,
en passant and promotion physically, as a person would.

The hand never tells the game which move was made. After the button, the camera
alone must work out what happened from the board, tray and rack.
"""
import chess
import numpy as np
import mujoco
from ai_gambit_perception.perception import tray_centers
from chess_coords import PIECE_REST_Z, square_to_pos

LIFT = .09  # A held piece floats this high above the square it came from.


class VirtualHand:
    def __init__(self, sim):
        self.sim = sim
        self.promotion = "queen"
        self.begin(chess.Board(), [1]*32, [0]*48)

    def begin(self, board, rack, tray):
        """Start a turn from the confirmed position and storage inventory."""
        self.board = board.copy()
        self.rack, self.tray = list(rack), list(tray)
        self.held = self.held_from = self.move = None
        self.targets = []
        self.history = []

    def _name(self, body):
        return self.sim.model.body(body).name

    def _white(self, body):
        return self._name(body).startswith(("w_", "spare_w_"))

    def _place(self, body, position):
        name = self._name(body)
        p = np.asarray(position, float).copy()
        p[2] = .022 if name.startswith("spare_") else PIECE_REST_Z[name.split("_")[1]]
        self.sim.model.body_gravcomp[body] = 0
        self.sim.put(body, p)

    def _lift(self, body):
        """Keep the held piece visible, floating without gravity or contacts."""
        m = self.sim.model
        p = self.sim.data.xpos[body].copy()
        p[2] = LIFT
        self.sim.put(body, p)
        m.body_gravcomp[body] = 1
        start, count = m.body_geomadr[body], m.body_geomnum[body]
        m.geom_contype[start:start+count] = m.geom_conaffinity[start:start+count] = 0

    def _save(self):
        m, d = self.sim.model, self.sim.data
        self.history.append((d.qpos.copy(), m.geom_contype.copy(), m.geom_conaffinity.copy(),
                             list(self.rack), list(self.tray)))

    def _to_tray(self, body):
        start = 0 if self._white(body) else 24
        slot = next((i for i in range(start, start+24) if not self.tray[i]), None)
        if slot is None:
            raise ValueError("Capture tray is full")
        self.tray[slot] = 1
        self._place(body, tray_centers()[slot])

    def _spare_slot(self):
        start = 0 if self.board.turn == chess.WHITE else 16
        types = self.sim.config["rack_types"]
        return next((i for i in range(start, start+16) if types[i] == self.promotion and self.rack[i]), None)

    def undo(self):
        """Put the held piece back, or take back the whole move."""
        if not self.history:
            return "Nothing to undo"
        m, d = self.sim.model, self.sim.data
        qpos, contype, conaffinity, self.rack, self.tray = self.history.pop()
        d.qpos[:], d.qvel[:] = qpos, 0
        m.geom_contype[:], m.geom_conaffinity[:] = contype, conaffinity
        m.body_gravcomp[self.sim.piece_ids] = 0
        self.held = self.held_from = self.move = None
        self.targets = []
        mujoco.mj_forward(m, d)
        return "Taken back - make your move"

    def click(self, square):
        """Click one of your pieces, then one of its legal squares."""
        name = chess.square_name(square)
        if self.move is not None:
            return f"You already played {self.move.uci()}. Press DONE, or U to take it back"
        if self.held is None:
            piece = self.board.piece_at(square)
            if piece is None:
                return f"No piece on {name}"
            if piece.color != self.board.turn:
                return f"That is not your piece; you play {'White' if self.board.turn else 'Black'}"
            moves = [m for m in self.board.legal_moves if m.from_square == square]
            if not moves:
                return f"The {chess.piece_name(piece.piece_type)} on {name} has no legal move"
            body = self.sim.body_at(square)
            if body is None:
                return f"Cannot find the piece on {name}; press L to reload"
            self._save()
            self.held, self.held_from, self.targets = body, square, moves
            self._lift(body)
            mujoco.mj_forward(self.sim.model, self.sim.data)
            return f"Holding the {chess.piece_name(piece.piece_type)} from {name}; click a green dot"
        if square == self.held_from:
            self.undo()
            return f"Put the piece back on {name}"
        piece = self.board.piece_at(square)
        if piece is not None and piece.color == self.board.turn:
            # Clicking another of your pieces switches to it, as in any chess UI.
            self.undo()
            return self.click(square)
        moves = [m for m in self.targets if m.to_square == square]
        if not moves:
            return f"Illegal: that piece cannot move to {name}"
        promotion = chess.PIECE_NAMES.index(self.promotion)
        move = next((m for m in moves if m.promotion in (None, promotion)), moves[0])
        slot = self._spare_slot() if move.promotion else None
        if move.promotion and slot is None:
            return f"No spare {self.promotion} left in the rack; press Q/R/B/N to pick another"
        body = self.held
        if self.board.is_capture(move):
            captured = move.to_square
            if self.board.is_en_passant(move):
                captured += -8 if self.board.turn == chess.WHITE else 8
            self._to_tray(self.sim.body_at(captured, ignore=body))
        if move.promotion:
            # Retire the pawn and take the labelled spare of the chosen type.
            self._to_tray(body)
            self.rack[slot] = 0
            body = self.sim.model.body(f"spare_{'w' if slot < 16 else 'b'}_{slot % 16:02d}").id
        self._place(body, square_to_pos(name))
        if self.board.is_castling(move):
            rank = chess.square_rank(move.from_square)
            kingside = move.to_square > move.from_square
            rook = self.sim.body_at(chess.square(7 if kingside else 0, rank))
            self._place(rook, square_to_pos(chess.square_name(chess.square(5 if kingside else 3, rank))))
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self.held = self.held_from = None
        self.targets, self.move = [], move
        return f"You played {move.uci()}. Press DONE, or U to take it back"

    def perform(self, board, move):
        """Scripted human for tests and demos: the same clicks a person makes."""
        if move.promotion:
            self.promotion = chess.piece_name(move.promotion)
        self.click(move.from_square)
        return self.click(move.to_square)
