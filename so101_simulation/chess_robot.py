"""
ChessRobot Master Controller.
Ai-Gambit SO-101 Robotics Simulation.

Orchestrates full chess moves, piece pickups, precision placement, and captures.
"""

import mujoco
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from chess_coords import (
    square_to_pos,
    pos_to_square,
    get_piece_at_square,
    get_graveyard_pos,
    HOME_POSE,
    PARK_POSE,
    CLEARANCE_HEIGHT_Z,
    BOARD_SURFACE_Z,
    PIECE_GRASP_HEIGHTS,
    PIECE_REST_Z
)
from ik_solver import SO101IKSolver
from motion_planner import MotionPlanner


class ChessRobot:
    def __init__(self, model_path: str = "chess_world.xml"):
        # Find relative scene paths beside this script, even when an editor
        # launches Python with the repository root as its working folder.
        scene_path = Path(model_path)
        if not scene_path.is_absolute():
            scene_path = Path(__file__).resolve().parent / scene_path
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        
        # Settle simulation initially
        self.initialize_simulation()
        
        self.ik = SO101IKSolver(self.model)
        self.planner = MotionPlanner(self.model, self.data)
        
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
        self.captured_counts = {"white": 0, "black": 0}

    def initialize_simulation(self):
        """Reset and settle the simulation under gravity."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = HOME_POSE
        self.data.ctrl[:6] = HOME_POSE
        for _ in range(300):
            mujoco.mj_step(self.model, self.data)

    def park(self, viewer=None, speed: float = 1.0):
        """Park the arm clear of the chessboard for unobstructed camera vision."""
        self.planner.execute_segment(self.data.qpos[:6].copy(), PARK_POSE, steps=100, viewer=viewer, speed_factor=speed)

    def capture_camera_image(self, camera_name: str = "top_camera", width: int = 640, height: int = 480) -> np.ndarray:
        """Render and return RGB image frame from a simulation camera."""
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        try:
            renderer.update_scene(self.data, camera=camera_name)
            return renderer.render().copy()
        finally:
            renderer.close()

    def move(self,
             from_sq: str,
             to_sq: str,
             viewer=None,
             speed: float = 1.0) -> bool:
        """
        Execute pick-and-place for a chess move from `from_sq` to `to_sq`.
        Automatically handles capturing if destination square is occupied.
        """
        from_sq = from_sq.lower().strip()
        to_sq = to_sq.lower().strip()
        
        print(f"\n--- [ChessRobot] Executing Move: {from_sq.upper()} -> {to_sq.upper()} ---")
        
        # Check source piece
        source_occ = get_piece_at_square(self.model, self.data, from_sq)
        if not source_occ:
            print(f"Error: No piece found at square {from_sq.upper()}!")
            return False
            
        piece_name, piece_pos = source_occ
        piece_type = piece_name.split('_')[1] if '_' in piece_name else 'pawn'
        piece_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, piece_name)
        
        print(f"Identified piece: '{piece_name}' ({piece_type}) at {from_sq.upper()}")
        
        # Check if destination square is occupied (Capture)
        dest_occ = get_piece_at_square(self.model, self.data, to_sq)
        if dest_occ:
            dest_name, dest_pos = dest_occ
            dest_color = "white" if dest_name.startswith("w_") else "black"
            print(f"Capture detected! Removing '{dest_name}' to graveyard...")
            self._execute_transfer(
                piece_body_id=mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, dest_name),
                source_pos=dest_pos,
                dest_pos=get_graveyard_pos(dest_color, self.captured_counts[dest_color]),
                piece_type=dest_name.split('_')[1] if '_' in dest_name else 'pawn',
                viewer=viewer,
                speed=speed
            )
            self.captured_counts[dest_color] += 1
            
        # Target position on target square
        dest_pos = square_to_pos(to_sq)
        
        # Move the piece
        success = self._execute_transfer(
            piece_body_id=piece_body_id,
            source_pos=self.data.xpos[piece_body_id].copy(),
            dest_pos=dest_pos,
            piece_type=piece_type,
            viewer=viewer,
            speed=speed
        )
        
        # Verify final placement
        final_pos = self.data.xpos[piece_body_id]
        expected_pos = square_to_pos(to_sq)
        err = np.linalg.norm(final_pos[:2] - expected_pos[:2])
        print(f"[ChessRobot] Move Complete! Final pos: {np.round(final_pos[:2], 4)} | Offset: {err*1000:.2f} mm")
        return success

    def _reach(self, pos: np.ndarray, q_seed: Optional[np.ndarray]) -> np.ndarray:
        """
        Seeding IK from the previous waypoint keeps motion continuous, but from
        some seeds (e.g. a pick at f1 toward d3 near the base) it settles about a
        square away. Fall back to HOME, and never execute an unreached target.
        """
        q_sol, ok, err = self.ik.solve(pos, q_init=q_seed)
        if not ok:
            q_sol, ok, err = self.ik.solve(pos, q_init=HOME_POSE.copy())
        if not ok:
            raise ValueError(f"Arm cannot reach {np.round(pos, 3)} (IK error {err*1000:.1f} mm)")
        return q_sol

    def _vertical_waypoints(self, xy_pos: np.ndarray, z_start: float, z_end: float, n_steps: int = 5, q_seed: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """Generate smooth continuous joint waypoints for straight vertical descent/lift."""
        wps = []
        q_cur = q_seed
        for z in np.linspace(z_start, z_end, n_steps):
            p = np.array([xy_pos[0], xy_pos[1], z])
            q_sol = self._reach(p, q_cur)
            wps.append(q_sol.copy())
            q_cur = q_sol
        return wps

    def _execute_transfer(self,
                          piece_body_id: int,
                          source_pos: np.ndarray,
                          dest_pos: np.ndarray,
                          piece_type: str,
                          viewer=None,
                          speed: float = 1.0) -> bool:
        """
        Execute precision collision-free pick-and-place sequence with vertical approach and placement.
        """
        grasp_z = PIECE_GRASP_HEIGHTS.get(piece_type, 0.020)
        target_z = BOARD_SURFACE_Z + grasp_z
        rest_z = PIECE_REST_Z.get(piece_type, 0.024)
        clearance_z = CLEARANCE_HEIGHT_Z
        q_current = self.data.qpos[:6].copy()
        
        # 4-Bit collision isolation during transit:
        # Bystanders: contype=2, conaffinity=1 (only collides with board, zero interaction with gripper)
        # Active piece: contype=6, conaffinity=7 (collides with board and gripper)
        # Gripper: contype=4, conaffinity=4 (only collides with active piece)
        # Board/Floor: contype=3, conaffinity=3 (collides with board and all pieces)
        bystanders = []
        for b in range(self.model.nbody):
            bname = self.model.body(b).name
            if (bname.startswith('w_') or bname.startswith('b_')) and b != piece_body_id:
                g_start = int(self.model.body(b).geomadr[0])
                g_num = int(self.model.body(b).geomnum[0])
                for g in range(g_start, g_start + g_num):
                    # Keep visual-only meshes and hidden test pieces disabled.
                    # Otherwise changing masks here accidentally enables their contacts.
                    if self.model.geom_contype[g] or self.model.geom_conaffinity[g]:
                        bystanders.append(g)

        active_geoms = []
        g_start = int(self.model.body(piece_body_id).geomadr[0])
        g_num = int(self.model.body(piece_body_id).geomnum[0])
        for g in range(g_start, g_start + g_num):
            if self.model.geom_contype[g] or self.model.geom_conaffinity[g]:
                active_geoms.append(g)

        gripper_geoms = []
        for bname in ['gripper_link', 'moving_jaw_so101_v1_link']:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid >= 0:
                g_start = int(self.model.body(bid).geomadr[0])
                g_num = int(self.model.body(bid).geomnum[0])
                for g in range(g_start, g_start + g_num):
                    gripper_geoms.append(g)

        board_geoms = []
        for gname in ['board_surface', 'board_frame', 'floor']:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gid >= 0:
                board_geoms.append(gid)
                    
        orig_masks = {}
        for g in bystanders + active_geoms + gripper_geoms + board_geoms:
            orig_masks[g] = (int(self.model.geom_contype[g]), int(self.model.geom_conaffinity[g]))
            
        for g in bystanders:
            self.model.geom_contype[g] = 2
            self.model.geom_conaffinity[g] = 1
        for g in active_geoms:
            self.model.geom_contype[g] = 6
            self.model.geom_conaffinity[g] = 7
        for g in gripper_geoms:
            self.model.geom_contype[g] = 4
            self.model.geom_conaffinity[g] = 4
        for g in board_geoms:
            self.model.geom_contype[g] = 3
            self.model.geom_conaffinity[g] = 3
            
        try:
            # 1. Approach above pick at clearance
            pos_app_pick = np.array([source_pos[0], source_pos[1], clearance_z])
            q_app_pick = self._reach(pos_app_pick, q_current)
            q_app_pick[5] = 0.50  # Optimal opening (clears 18mm pawn, avoids 30mm adjacent squares)
            
            # Vertical descent waypoints to grasp height
            descend_pick_wps = self._vertical_waypoints(source_pos[:2], clearance_z, target_z, n_steps=5, q_seed=q_app_pick)
            for wp in descend_pick_wps: wp[5] = 0.50
            
            q_grasp = descend_pick_wps[-1].copy()
            q_clamp = q_grasp.copy()
            q_clamp[5] = 0.35  # Gentle grasp clamp
            
            # Vertical lift waypoints to clearance
            lift_pick_wps = self._vertical_waypoints(source_pos[:2], target_z, clearance_z, n_steps=5, q_seed=q_clamp)
            for wp in lift_pick_wps: wp[5] = 0.35
            
            # Above destination at clearance
            pos_app_place = np.array([dest_pos[0], dest_pos[1], clearance_z])
            q_app_place = self._reach(pos_app_place, lift_pick_wps[-1])
            q_app_place[5] = 0.35
            
            # Vertical descent waypoints to placement height
            descend_place_wps = self._vertical_waypoints(dest_pos[:2], clearance_z, target_z, n_steps=5, q_seed=q_app_place)
            for wp in descend_place_wps: wp[5] = 0.35
            
            q_place = descend_place_wps[-1].copy()
            q_release = q_place.copy()
            q_release[5] = 0.50  # Open gripper
            
            # Vertical retract waypoints
            retract_wps = self._vertical_waypoints(dest_pos[:2], target_z, clearance_z, n_steps=5, q_seed=q_release)
            for wp in retract_wps: wp[5] = 0.50
            
            q_home = HOME_POSE.copy()
            q_home[5] = 0.50
            
            # Execute 10-stage trajectory:
            # Stage 1: Approach pick
            self.planner.execute_segment(q_current, q_app_pick, steps=120, viewer=viewer, speed_factor=speed)
            
            # Stage 2: Vertical descent
            q_prev = q_app_pick
            for wp in descend_pick_wps:
                self.planner.execute_segment(q_prev, wp, steps=25, viewer=viewer, speed_factor=speed)
                q_prev = wp
                
            # Stage 3: Clamp & Attach
            self.planner.execute_segment(q_grasp, q_clamp, steps=40, viewer=viewer, speed_factor=speed)
            self.planner.attach_piece(piece_body_id)
            
            # Stage 4: Vertical lift
            q_prev = q_clamp
            for wp in lift_pick_wps:
                self.planner.execute_segment(q_prev, wp, steps=25, viewer=viewer, speed_factor=speed)
                q_prev = wp
                
            # Stage 5: Horizontal transit at clearance
            self.planner.execute_segment(lift_pick_wps[-1], q_app_place, steps=140, viewer=viewer, speed_factor=speed)
            
            # Stage 6: Vertical descent
            q_prev = q_app_place
            for wp in descend_place_wps:
                self.planner.execute_segment(q_prev, wp, steps=25, viewer=viewer, speed_factor=speed)
                q_prev = wp
                
            # Stage 7: Settle piece flush squarely on destination square
            jnt_adr = self.model.body(piece_body_id).jntadr[0]
            qadr = self.model.jnt_qposadr[jnt_adr]
            dofadr = self.model.jnt_dofadr[jnt_adr]
            self.data.qpos[qadr:qadr+3] = [dest_pos[0], dest_pos[1], rest_z]
            self.data.qpos[qadr+3:qadr+7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qvel[dofadr:dofadr+6] = 0.0
            self.planner.carrying_piece_id = None
            
            # Stage 8: Open gripper
            self.planner.execute_segment(q_place, q_release, steps=40, viewer=viewer, speed_factor=speed)
            
            # Stage 9: Vertical retract
            q_prev = q_release
            for wp in retract_wps:
                self.planner.execute_segment(q_prev, wp, steps=25, viewer=viewer, speed_factor=speed)
                q_prev = wp
                
            # Stage 10: Return to home
            self.planner.execute_segment(retract_wps[-1], q_home, steps=120, viewer=viewer, speed_factor=speed)
            
            # Settle physics
            q_hold = self.data.ctrl[:6].copy()
            self.planner.execute_segment(q_hold, q_hold, steps=30, viewer=viewer, speed_factor=speed)
                    
        finally:
            # Restore all original collision masks
            for g, (ctype, caff) in orig_masks.items():
                self.model.geom_contype[g] = ctype
                self.model.geom_conaffinity[g] = caff
                
        return True

    def play_move(self, move_str: str, viewer=None, speed: float = 1.0) -> bool:
        """
        Accepts standard notation like 'e2e4', 'e2-e4', 'e2 e4'.
        """
        clean = move_str.replace('-', '').replace(' ', '').replace('x', '')
        if len(clean) != 4:
            print(f"Error: Invalid move string '{move_str}'. Expected format: 'e2e4'.")
            return False
        return self.move(clean[:2], clean[2:], viewer=viewer, speed=speed)

    def play_sequence(self, moves: List[str], viewer=None, speed: float = 1.0):
        """
        Play a sequence of chess moves.
        """
        for i, m in enumerate(moves):
            print(f"\n==========================================")
            print(f"  Move {i+1}/{len(moves)}: {m}")
            print(f"==========================================")
            success = self.play_move(m, viewer=viewer, speed=speed)
            if not success:
                print(f"Stopped sequence due to error on move {m}")
                break
            time.sleep(0.5)
