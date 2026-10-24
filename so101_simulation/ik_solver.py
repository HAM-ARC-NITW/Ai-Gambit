"""
Inverse Kinematics Solver for SO-101 Robot Arm.
Ai-Gambit SO-101 Robotics Simulation.

Solves joint angles for target 3D end-effector positions using robust
Levenberg-Marquardt kinematics with downward vertical orientation preference.
"""

import mujoco
import numpy as np
from typing import Optional, Tuple
from chess_coords import square_to_pos


class SO101IKSolver:
    def __init__(self, model: mujoco.MjModel, site_name: str = "pinch_site"):
        self.model = model
        self.data = mujoco.MjData(model)
        
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in MuJoCo model!")
            
        self.gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_link")
        
        # Tip direction vector in gripper_link frame (pointing from wrist towards finger pinch)
        self.v_tip_local = np.array([0.018, 0.020, -0.095], dtype=np.float64)
        self.v_tip_local /= np.linalg.norm(self.v_tip_local)
        
        # Arm joints 0..4 (pan, lift, elbow, wrist_flex, wrist_roll)
        self.n_arm_joints = 5
        self.jnt_ranges = model.jnt_range[:self.n_arm_joints]

    def solve(self,
              target_pos: np.ndarray,
              q_init: Optional[np.ndarray] = None,
              prefer_vertical: bool = True,
              max_iter: int = 150,
              tol: float = 5e-4) -> Tuple[np.ndarray, bool, float]:
        """
        Solve Inverse Kinematics for target 3D pinch site position.
        
        Prioritizes top-down vertical hand alignment to prevent gripper collisions
        with surrounding chess pieces.
        
        Args:
            target_pos: Desired [x, y, z] pinch site position in world frame
            q_init: Initial joint guess (6 elements). If None, uses analytic heuristics.
            prefer_vertical: If True, uses nullspace projection to orient fingers vertically downward
            max_iter: Max iterations
            tol: Convergence tolerance in meters (default 0.5 mm)
            
        Returns:
            Tuple of (q_solution, success, residual_error)
        """
        target_pos = np.asarray(target_pos, dtype=np.float64)
        pan = np.arctan2(target_pos[1], target_pos[0])
        dist_xy = np.linalg.norm(target_pos[:2])
        
        if q_init is not None:
            guesses = [q_init.copy().astype(np.float64)]
        elif dist_xy < 0.28:
            # High-crane top-down postures: wrist flexed downward, wrist_roll aligned
            guesses = [
                np.array([pan, -0.3, 0.7, 1.4, -0.79, 0.0], dtype=np.float64),
                np.array([pan, -0.5, 1.0, 1.2, -0.79, 0.0], dtype=np.float64),
                np.array([pan, -0.1, 0.4, 1.6, -0.79, 0.0], dtype=np.float64),
                np.array([pan, -0.7, 1.3, 1.0, -0.79, 0.0], dtype=np.float64),
            ]
        else:
            # Extended reach postures for far squares (ranks 6-8)
            guesses = [
                np.array([pan, -0.7, 1.2, 0.5, -0.79, 0.0], dtype=np.float64),
                np.array([pan, -0.9, 1.5, 0.2, -0.79, 0.0], dtype=np.float64),
                np.array([pan, -0.5, 0.9, 0.8, -0.79, 0.0], dtype=np.float64),
            ]
            
        target_down = np.array([0.0, 0.0, -1.0])
        best_q = None
        best_err = 1e9
        
        for q_seed in guesses:
            q = q_seed.copy()
            for it in range(max_iter):
                self.data.qpos[:len(q)] = q
                mujoco.mj_forward(self.model, self.data)
                
                cur_pos = self.data.site_xpos[self.site_id]
                err_pos = target_pos - cur_pos
                err_norm = np.linalg.norm(err_pos)
                
                if err_norm < best_err:
                    best_err = err_norm
                    best_q = q.copy()
                    
                if err_norm < tol:
                    return q, True, err_norm
                    
                jacp = np.zeros((3, self.model.nv))
                jacr = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
                
                J1 = jacp[:, :self.n_arm_joints]
                lambda_sq = 1e-4 if err_norm > 0.01 else 1e-5
                J1_pinv = J1.T @ np.linalg.inv(J1 @ J1.T + lambda_sq * np.eye(3))
                dq1 = J1_pinv @ err_pos
                
                if prefer_vertical:
                    R_grip = self.data.xmat[self.gripper_body_id].reshape(3, 3)
                    cur_tip = R_grip @ self.v_tip_local
                    # Tip Jacobian: d(tip)/dq = -skew(cur_tip) @ jacr
                    skew_tip = np.array([
                        [0, -cur_tip[2], cur_tip[1]],
                        [cur_tip[2], 0, -cur_tip[0]],
                        [-cur_tip[1], cur_tip[0], 0]
                    ])
                    J_tip = -skew_tip @ jacr[:, :self.n_arm_joints]
                    
                    # Nullspace projector of position task
                    N = np.eye(self.n_arm_joints) - J1_pinv @ J1
                    dq2 = N @ (J_tip.T @ (0.40 * (target_down - cur_tip)))
                    dq = dq1 + dq2
                else:
                    dq = dq1
                    
                step_size = 0.7 if err_norm > 0.005 else 0.8
                q[:self.n_arm_joints] += step_size * dq
                
                # Clip to valid joint limits
                for j in range(self.n_arm_joints):
                    q[j] = np.clip(q[j], self.jnt_ranges[j, 0], self.jnt_ranges[j, 1])
                    
            if best_err < tol:
                break
                
        return best_q, best_err < tol, best_err

    def solve_for_square(self,
                         square: str,
                         z_offset: float = 0.024,
                         q_init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool, float]:
        """
        Solve IK directly for a named algebraic chess square.
        """
        pos = square_to_pos(square, z_offset=z_offset)
        return self.solve(pos, q_init=q_init)
