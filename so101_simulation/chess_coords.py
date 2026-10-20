"""
Chessboard Coordinate System and Spatial Mapping.
Ai-Gambit SO-101 Robotics Simulation.

Maps standard algebraic chess coordinates (a1..h8) to 3D simulation coordinates (x, y, z).
"""

import numpy as np
from typing import Dict, Optional, Tuple

BOARD_CENTER_X = 0.250
BOARD_CENTER_Y = 0.000
BOARD_SURFACE_Z = 0.010  # Top of chessboard surface
SQUARE_SIZE = 0.030      # 30 mm per square

FILES = {'a': 3.5, 'b': 2.5, 'c': 1.5, 'd': 0.5,
         'e': -0.5, 'f': -1.5, 'g': -2.5, 'h': -3.5}

RANKS = {'1': -3.5, '2': -2.5, '3': -1.5, '4': -0.5,
         '5': 0.5, '6': 1.5, '7': 2.5, '8': 3.5}

# Piece grasp height offsets above table surface (fingertip target)
PIECE_GRASP_HEIGHTS = {
    'pawn': 0.020,
    'rook': 0.020,
    'knight': 0.024,
    'bishop': 0.024,
    'queen': 0.028,
    'king': 0.030,
}

# Piece resting cylinder center heights on the chessboard surface
PIECE_REST_Z = {
    'pawn': 0.024,
    'rook': 0.024,
    'knight': 0.028,
    'bishop': 0.028,
    'queen': 0.031,
    'king': 0.0325,
}

CLEARANCE_HEIGHT_Z = 0.130  # 13 cm clearance while moving pieces over the board
HOME_POSE = np.array([0.0, -0.6, 1.2, -0.6, 0.0, 1.2])
PARK_POSE = np.array([-1.4, -0.8, 1.2, -0.4, 0.0, 0.0])  # Swung clear of overhead camera



def square_to_pos(square: str, z_offset: float = 0.0) -> np.ndarray:
    """
    Convert algebraic chess square (e.g. 'e2', 'e4', 'a8') to 3D world position.
    
    Args:
        square: Algebraic square name ('a1' to 'h8')
        z_offset: Height above board surface
        
    Returns:
        np.ndarray: [x, y, z] position in world coordinates
    """
    square = square.lower().strip()
    if len(square) != 2 or square[0] not in FILES or square[1] not in RANKS:
        raise ValueError(f"Invalid chess square: '{square}'")
        
    f_char = square[0]
    r_char = square[1]
    
    x = BOARD_CENTER_X + RANKS[r_char] * SQUARE_SIZE
    y = BOARD_CENTER_Y + FILES[f_char] * SQUARE_SIZE
    z = BOARD_SURFACE_Z + z_offset
    
    return np.array([x, y, z], dtype=np.float64)


def pos_to_square(x: float, y: float) -> Optional[str]:
    """
    Identify algebraic chess square from world (x, y) coordinates.
    Returns None if (x, y) is outside the 8x8 chessboard.
    """
    dx = x - BOARD_CENTER_X
    dy = y - BOARD_CENTER_Y
    
    # Check within board boundary (+- 0.12 m)
    if abs(dx) > 0.12 or abs(dy) > 0.12:
        return None
        
    rank_idx = int(np.floor((dx / SQUARE_SIZE) + 4.0)) + 1
    file_idx = 7 - int(np.floor((dy / SQUARE_SIZE) + 4.0))
    
    if 1 <= rank_idx <= 8 and 0 <= file_idx <= 7:
        f_char = chr(ord('a') + file_idx)
        return f"{f_char}{rank_idx}"
    return None


def get_graveyard_pos(color: str, index: int) -> np.ndarray:
    """
    Return coordinate for captured pieces off the board.
    White graveyard: y = -0.165
    Black graveyard: y = +0.165
    """
    y = -0.165 if color.lower().startswith('w') else 0.165
    x = 0.145 + (index % 8) * 0.028
    z = BOARD_SURFACE_Z + 0.015
    return np.array([x, y, z], dtype=np.float64)


def get_board_occupancy(model, data) -> Dict[str, Tuple[str, np.ndarray]]:
    """
    Scan all chess piece bodies in simulation and return mapping:
    square -> (piece_body_name, current_world_pos)
    """
    occupancy = {}
    for b in range(model.nbody):
        bname = model.body(b).name
        if bname.startswith('w_') or bname.startswith('b_'):
            pos = data.xpos[b]
            sq = pos_to_square(pos[0], pos[1])
            if sq is not None:
                occupancy[sq] = (bname, pos.copy())
    return occupancy


def get_piece_at_square(model, data, square: str) -> Optional[Tuple[str, np.ndarray]]:
    """
    Find which piece occupies the given square.
    """
    occ = get_board_occupancy(model, data)
    return occ.get(square.lower(), None)
