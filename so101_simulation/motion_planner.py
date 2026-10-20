"""
Motion Planner and Trajectory Generator for SO-101 Arm.
Ai-Gambit SO-101 Robotics Simulation.

Generates smooth multi-waypoint trajectories for collision-free pick-and-place maneuvers.
"""

import mujoco
import numpy as np
import time
from typing import Callable, List, Optional
from chess_coords import CLEARANCE_HEIGHT_Z, HOME_POSE, PIECE_GRASP_HEIGHTS
from ik_solver import SO101IKSolver


class MotionPlanner:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.ik = SO101IKSolver(model)
        
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
        self.home_pose = HOME_POSE.copy()
        
        # Gripper control angles
        self.GRIPPER_OPEN = 1.0
        self.GRIPPER_CLOSED = 0.35

        # Active piece tracking during transit
        self.carrying_piece_id: Optional[int] = None
        self.carry_offset = np.zeros(3)

    def plan_pick_and_place_waypoints(self,
                                      pick_pos: np.ndarray,
                                      place_pos: np.ndarray,
                                      piece_type: str = "pawn") -> List[np.ndarray]:
        """
        Compute the sequence of joint configurations for pick-and-place:
        1. Approach above pick (clearance)
        2. Descend to pick
        3. Clamp piece
        4. Lift to clearance
        5. Transit to above place
        6. Descend to place
        7. Release piece
        8. Retract to clearance
        9. Return to home
        """
        grasp_z = PIECE_GRASP_HEIGHTS.get(piece_type.lower(), 0.015)
        
        # 1. Approach above pick
        pos_app_pick = pick_pos.copy()
        pos_app_pick[2] = CLEARANCE_HEIGHT_Z
        q_app_pick, ok1, _ = self.ik.solve(pos_app_pick, self.home_pose)
        q_app_pick[5] = self.GRIPPER_OPEN
        
        # 2. Descend to grasp
        pos_grasp = pick_pos.copy()
        pos_grasp[2] = grasp_z
        q_grasp, ok2, _ = self.ik.solve(pos_grasp, q_app_pick)
        q_grasp[5] = self.GRIPPER_OPEN
        
        # 3. Clamp
        q_clamp = q_grasp.copy()
        q_clamp[5] = self.GRIPPER_CLOSED
        
        # 4. Lift to clearance
        q_lift_pick = q_app_pick.copy()
        q_lift_pick[5] = self.GRIPPER_CLOSED
        
        # 5. Above placement
        pos_app_place = place_pos.copy()
        pos_app_place[2] = CLEARANCE_HEIGHT_Z
        q_app_place, ok3, _ = self.ik.solve(pos_app_place, q_lift_pick)
        q_app_place[5] = self.GRIPPER_CLOSED
        
        # 6. Descend to place
        pos_place = place_pos.copy()
        pos_place[2] = grasp_z
        q_place, ok4, _ = self.ik.solve(pos_place, q_app_place)
        q_place[5] = self.GRIPPER_CLOSED
        
        # 7. Release
        q_release = q_place.copy()
        q_release[5] = self.GRIPPER_OPEN
        
        # 8. Retract
        q_retract = q_app_place.copy()
        q_retract[5] = self.GRIPPER_OPEN
        
        # 9. Home
        q_home = self.home_pose.copy()
        q_home[5] = self.GRIPPER_OPEN
        
        return [
            q_app_pick,
            q_grasp,
            q_clamp,
            q_lift_pick,
            q_app_place,
            q_place,
            q_release,
            q_retract,
            q_home
        ]

    def execute_segment(self,
                        q_start: np.ndarray,
                        q_end: np.ndarray,
                        steps: int = 250,
                        step_callback: Optional[Callable] = None,
                        viewer=None,
                        speed_factor: float = 1.0):
        """
        Execute smooth cosine S-curve trajectory between two configurations.
        """
        dt = self.model.opt.timestep
        delay = (dt / speed_factor) if viewer else 0.0
        
        for s in range(steps):
            alpha = (1.0 - np.cos(np.pi * (s + 1) / steps)) / 2.0
            q_des = q_start + alpha * (q_end - q_start)
            
            # Send control signal to actuators
            self.data.ctrl[:6] = q_des
            
            # If carrying a piece, rigidly lock it to pinch site
            if self.carrying_piece_id is not None:
                jnt_adr = self.model.body(self.carrying_piece_id).jntadr[0]
                if jnt_adr >= 0:
                    qadr = self.model.jnt_qposadr[jnt_adr]
                    dofadr = self.model.jnt_dofadr[jnt_adr]
                    self.data.qpos[qadr:qadr + 3] = self.data.site_xpos[self.site_id] + self.carry_offset
                    self.data.qvel[dofadr:dofadr + 6] = 0.0
                    
            mujoco.mj_step(self.model, self.data)
            
            if step_callback:
                step_callback(self.data)
                
            if viewer is not None:
                viewer.sync()
                if delay > 0:
                    time.sleep(delay)

    def attach_piece(self, piece_body_id: int):
        """
        Latch a piece to the pinch site for transit.
        """
        self.carrying_piece_id = piece_body_id
        jnt_adr = self.model.body(piece_body_id).jntadr[0]
        if jnt_adr >= 0:
            qadr = self.model.jnt_qposadr[jnt_adr]
            self.carry_offset = self.data.qpos[qadr:qadr + 3] - self.data.site_xpos[self.site_id]

    def detach_piece(self):
        """
        Release held piece onto target square.
        """
        if self.carrying_piece_id is not None:
            jnt_adr = self.model.body(self.carrying_piece_id).jntadr[0]
            if jnt_adr >= 0:
                dofadr = self.model.jnt_dofadr[jnt_adr]
                self.data.qvel[dofadr:dofadr + 6] = 0.0
            self.carrying_piece_id = None
            self.carry_offset = np.zeros(3)
