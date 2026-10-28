"""Small Stockfish adapter. It chooses moves; it never changes the game board.

python-chess speaks the UCI protocol for us, including promotions and full move
history. The caller must execute and camera-verify a move before committing it.
"""
import math
import os
from pathlib import Path
import shutil
import subprocess
import chess.engine


def find_stockfish(path=None):
    """Explicit path, environment variable, PATH, then this project's download."""
    requested = path or os.environ.get("STOCKFISH_PATH")
    if requested:
        executable = shutil.which(str(requested))
        if executable:return executable
        raise FileNotFoundError(f"Stockfish executable not found: {requested}")
    installed = shutil.which("stockfish")
    if installed:return installed
    folder = Path(__file__).resolve().parent
    # The project installers unpack Windows .exe or Linux builds under bin/stockfish/.
    local = [p for p in sorted((folder/"bin").glob("stockfish/stockfish*"))
             if (p.suffix == ".exe") == (os.name == "nt")]
    candidates = [folder/"bin/stockfish.exe",*local,Path("/usr/games/stockfish")]
    for candidate in candidates:
        if candidate.is_file():return str(candidate)
    raise FileNotFoundError("Stockfish is missing. Run engine_stuff/install_stockfish.ps1 on Windows or "
                            "engine_stuff/install_stockfish.sh on Linux, or set STOCKFISH_PATH. "
                            "See engine_stuff/README.md.")


class StockfishEngine:
    def __init__(self,path=None,time_limit=.5,skill=10):
        if not math.isfinite(time_limit) or time_limit<=0:
            raise ValueError("Engine thinking time must be finite and positive")
        if isinstance(skill,bool) or not isinstance(skill,int) or not 0<=skill<=20:
            raise ValueError("Stockfish skill must be an integer from 0 to 20")
        self.path = find_stockfish(path)
        self.time_limit = time_limit
        # Hide the engine's console on Windows; MuJoCo/camera windows stay visible.
        options = dict(creationflags=subprocess.CREATE_NO_WINDOW) if os.name=="nt" else {}
        self.engine = chess.engine.SimpleEngine.popen_uci(self.path,timeout=5.,**options)
        try:
            # One thread and a small hash leave resources for rendering/physics.
            self.engine.configure({"Threads":1,"Hash":64,"Skill Level":skill})
        except Exception:
            self.engine.close()
            raise
        self.new_game()

    @property
    def name(self):
        return self.engine.id.get("name","Stockfish")

    def new_game(self):
        """A new token tells UCI to clear old-game search state on the next call."""
        self.game = object()

    def choose_move(self,board,depth=None):
        if not board.is_valid():raise ValueError("Engine needs a valid confirmed chess position")
        if depth is not None and (isinstance(depth,bool) or not isinstance(depth,int) or depth<1):
            raise ValueError("Search depth must be a positive integer")
        if board.is_game_over(claim_draw=True):return None
        # Send the full confirmed position/history. Searching must not push a
        # speculative move onto the coordinator's official board.
        result = self.engine.play(board.copy(stack=True),
            chess.engine.Limit(time=self.time_limit,depth=depth),game=self.game)
        if result.move not in board.legal_moves:
            raise ValueError("Stockfish returned no legal move; game paused")
        return result.move

    def close(self):
        try:self.engine.quit()
        except (chess.engine.EngineError,TimeoutError):self.engine.close()

