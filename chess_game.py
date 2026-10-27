import chess
import chess.engine

STOCKFISH_PATH = "/usr/games/stockfish"
ELO = 1350  # approximate strength; Stockfish's floor is ~1320-1350


def sq_to_xy(square: int) -> tuple:
    """chess square -> (file, rank), 1-indexed. a1=(1,1), e2=(5,2), h8=(8,8)."""
    return (chess.square_file(square) + 1, chess.square_rank(square) + 1)


class ChessGame:
    def __init__(self, engine_path=STOCKFISH_PATH, elo=ELO, think_time=0.5):
        self.board = chess.Board()  # full game state is kept here
        self.think_time = think_time
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})

    # normal input: opponent's move in UCI (e.g. "e2e4", promotion: "e7e8q")
    def apply_opponent_move(self, uci_move: str) -> bool:
        try:
            move = chess.Move.from_uci(uci_move.strip().lower())
        except ValueError:
            return False
        if move not in self.board.legal_moves:
            return False
        self.board.push(move)
        return True

    # engine reply -> output for the arm
    def get_engine_move(self) -> dict:
        result = self.engine.play(self.board, chess.engine.Limit(time=self.think_time))
        move = result.move

        # must be computed BEFORE pushing the move
        capture = self.board.is_capture(move)  # True for normal captures and en passant
        special = None
        if self.board.is_castling(move):
            special = "castle_kingside" if self.board.is_kingside_castling(move) else "castle_queenside"
        elif self.board.is_en_passant(move):
            special = "en_passant"
        elif move.promotion:
            special = "promotion_" + chess.piece_name(move.promotion)

        output = {
            "source": sq_to_xy(move.from_square),
            "destination": sq_to_xy(move.to_square),
            "capture": capture,
            "special": special,  # None for ordinary moves
        }
        self.board.push(move)
        return output

    # recovery only: use when the camera and the internal board disagree
    def resync(self, fen: str) -> bool:
        """Reset the internal board from a FEN string. Returns False if the FEN is invalid."""
        try:
            self.board.set_fen(fen)
            return True
        except ValueError:
            return False

    def status(self) -> str:
        if self.board.is_checkmate():
            return "checkmate"
        if self.board.is_stalemate():
            return "stalemate"
        if self.board.is_insufficient_material() or self.board.can_claim_draw():
            return "draw"
        if self.board.is_check():
            return "check"
        return "ongoing"

    def close(self):
        self.engine.quit()


if __name__ == "__main__":
    game = ChessGame()
    print("You are White. Enter moves like e2e4 (or 'quit').\n")
    print(game.board, "\n")

    try:
        while game.status() not in ("checkmate", "stalemate", "draw"):
            user = input("Your move: ")
            if user == "quit":
                break
            if not game.apply_opponent_move(user):
                print("Illegal or invalid move, try again.")
                continue
            if game.status() in ("checkmate", "stalemate", "draw"):
                break

            out = game.get_engine_move()
            print(f"source={out['source']}, destination={out['destination']}, "
                  f"capture={out['capture']}")
            if out["special"]:
                print("special:", out["special"])
            print(game.board, "\n")
            print("Status:", game.status(), "\n")
    finally:
        print("Game over:", game.status())
        game.close()