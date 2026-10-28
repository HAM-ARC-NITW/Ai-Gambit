#!/usr/bin/env python3
"""
SO-101 tic-tac-toe, driven by computer vision (MuJoCo simulation + real robot).

You play X (red pieces), the arm plays O (blue pieces) using a minimax engine
(it never loses).

The robot does NOT read your move from the keyboard or from the physics state.
It looks at the board through a top-down camera, segments red / blue pieces,
works out which cell changed, and only then thinks and moves.

  Simulation:  python so101_tictactoe.py
               Ways to play your X (pick whichever is easiest):
                 1. CLICK a cell in the "robot eyes" window (works with a trackpad).
                 2. Press a digit key 1-9 while the "robot eyes" window is focused.
                 3. Type a cell number 1-9 in the terminal and press Enter.
                 4. Drag an X piece in the MuJoCo viewer (double-click the piece, then
                    Ctrl + right-drag; add Shift to slide it horizontally). This needs a
                    real mouse - it is awkward on a trackpad, use 1-3 instead.
               In every case the robot only reacts to what the camera sees.

  Grasping:    the simulated claw has two jaw pads that collide with the pieces. Pieces are held
               by friction only (nothing is glued to the gripper), and after lifting, the code checks
               that the piece really came up with the claw; otherwise it retries / reports a failure.

  Real robot:  python so101_tictactoe.py --real --port /dev/ttyACM0 --camera 0
               (needs lerobot installed, a top-down webcam, red X / blue O pieces)

  Check reach: python so101_tictactoe.py --check      (no motion, just IK report)

Board cell numbers (robot frame: x forward, y left):

        far from robot
     1 2 3        cell 1 = far + robot's right   (x = BOARD_X + CELL, y = BOARD_Y - CELL)
     4 5 6        cell 3 = far + robot's left
     7 8 9        cell 7 = near + robot's right
        near robot
"""
from __future__ import annotations

import argparse
import json
import queue
import random
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:  # lets the pure-python parts be imported/tested without MuJoCo
    mujoco = None

HERE = Path(__file__).resolve().parent

# =============================================================================
# Geometry  (metres, robot base frame: x forward, y left, z up, base at origin)
# =============================================================================
BOARD_X, BOARD_Y = 0.19, 0.0      # centre of the 3x3 grid
CELL = 0.035                      # cell pitch
BOARD_TOP = 0.006                 # height of the board surface
PIECE_H = 0.024                   # piece height
PIECE_R = 0.011                   # O puck radius
TRAY_Y = 0.095                    # trays sit left (O) / right (X) of the board
TRAY_X0 = 0.135
TRAY_DX = 0.029                   # pitch along x ...
TRAY_STAGGER = 0.030              # ... and odd pieces are pushed outward: neighbours are
                                  # ~42 mm apart so an open gripper never touches them
HOVER_DZ = 0.07                   # clearance above grasp height while travelling
GRASP_FRAC = 0.65                 # grasp at 65 % of piece height (fingertips stay off the table)

X_TRAY: list = []
O_TRAY: list = []


def rebuild_layout():
    global X_TRAY, O_TRAY
    xs = [TRAY_X0 + TRAY_DX * k for k in range(5)]
    off = [TRAY_STAGGER * (k % 2) for k in range(5)]
    X_TRAY = [(x, BOARD_Y - TRAY_Y - o) for x, o in zip(xs, off)]   # human pieces (right of board)
    O_TRAY = [(x, BOARD_Y + TRAY_Y + o) for x, o in zip(xs, off)]   # robot pieces (left of board)


rebuild_layout()


def cell_xy(i):
    r, c = divmod(i, 3)
    return BOARD_X + (1 - r) * CELL, BOARD_Y + (c - 1) * CELL


def z_floor():          # tcp height when grasping a piece standing on the table
    return PIECE_H * GRASP_FRAC


def z_board():          # tcp height when placing a piece on the board (2 mm drop on release)
    return BOARD_TOP + PIECE_H * GRASP_FRAC + 0.002


# =============================================================================
# Tunables
# =============================================================================
CTRL_DT = 0.02                    # control tick (50 Hz)
CAM_W, CAM_H = 800, 600
# gripper joint [rad]: 0 = closed, 0.785 = fully open
G_OPEN = 0.70                     # fully open (travel / rest)
G_PICK = 0.50                     # opening used when approaching a piece
G_REL = 0.42                      # just-enough opening after releasing a piece (don't hit neighbours)
G_HOLD = 0.22                     # closed on a piece
WRIST_ROLL = 0.0
# Claw model used by the simulation: two flat jaw pads on the fingertips that really collide with the
# pieces and hold them by FRICTION (nothing is glued or teleported). Size them to match your real claw.
TCP_LOCAL = (-0.0079, -0.000218121, -0.0981274)   # fingertip point in the gripper_link frame
PAD_T, PAD_W, PAD_HH = 0.003, 0.009, 0.010        # half thickness / half width / half height of a jaw pad [m]
PAD_DZ = 0.001                                    # pad centre is this far "up" the tool axis from the tcp
PAD_KP, PAD_KV, PAD_FMAX = 600.0, 8.0, 6.0        # jaw actuator: squeeze force ~ KP * (piece_half_width - cmd)
MIN_LINK_Z = 0.04                 # reject IK solutions that fold the elbow into the table
WARN_TILT_DEG = 25.0              # --check warns when the gripper is tilted more than this
HOME_Q = np.array([0.0, -1.35, 1.24, 1.6])   # arm folded back over the base, clear of board + camera

LIN_STEP = 0.012                  # straight-line moves are solved every 12 mm
MAX_JOINT_STEP = 0.5              # rad; larger jump between IK waypoints = branch flip -> reject
GRASP_TOL_XY = 0.008              # sim: a held piece must stay within 8 mm (xy) of the fingertips
GRASP_TOL_Z = 0.014
GRASP_RETRIES = 2

GRID_PX = 300                     # rectified board image is GRID_PX x GRID_PX
STABLE_FRAMES = 6                 # board reading must be identical this many times
STABLE_DT = 0.25                  # seconds between readings
MIN_BLOB_AREA = 400               # px^2 in the rectified image
RED_S, RED_V = 110, 70
BLUE_S, BLUE_V, BLUE_H = 110, 60, (95, 130)

# Real-robot mapping MuJoCo radians -> LeRobot degrees (check on first run!)
REAL_SIGN = np.array([1.0, 1.0, 1.0, 1.0])
REAL_OFFSET_DEG = np.array([0.0, 0.0, 0.0, 0.0])
# Gripper 0..100 %, one value per sim gripper angle [0, G_HOLD, G_REL, G_PICK, G_OPEN].
# REAL_G_HOLD_PCT must be BELOW the percentage at which the jaws touch the puck, so the motor
# stalls on the piece and squeezes it (tune so it holds firmly without crushing/overheating).
G_BREAKS = [0.0, G_HOLD, G_REL, G_PICK, G_OPEN]
# Sim claw: half the gap between the jaw pads [m] for each gripper angle above. G_HOLD is deliberately
# narrower than the puck (radius 11 mm) so the jaws squeeze it; the others clear it with margin.
JAW_HALF_GAP = [0.004, 0.008, 0.0135, 0.0165, 0.0215]


def jaw_half_gap(g):
    return float(np.interp(g, G_BREAKS, JAW_HALF_GAP))


REAL_G_PCT = [0.0, 12.0, 35.0, 45.0, 60.0]
REAL_GRASP_MARGIN = 0.06          # rad: jaws stopped this far beyond the command => a piece is in them
REAL_SPEED = 2.0                  # >1 = slower than simulation


class Quit(Exception):
    pass


class Unreachable(Exception):
    pass


class GraspFailed(Exception):
    pass


# =============================================================================
# Game logic + engine
# =============================================================================
WINS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def winner(b):
    for a, c, d in WINS:
        if b[a] and b[a] == b[c] == b[d]:
            return b[a]
    return None


def minimax(b, player, depth=0):
    """Returns (score, [best moves]); score > 0 is good for O (the robot)."""
    w = winner(b)
    if w:
        return (10 - depth if w == "O" else depth - 10), []
    free = [i for i in range(9) if not b[i]]
    if not free:
        return 0, []
    best, moves = None, []
    for i in free:
        b[i] = player
        s, _ = minimax(b, "X" if player == "O" else "O", depth + 1)
        b[i] = None
        better = best is None or (player == "O" and s > best) or (player == "X" and s < best)
        if better:
            best, moves = s, [i]
        elif s == best:
            moves.append(i)
    return best, moves


def best_move(board):
    _, moves = minimax(list(board), "O")
    return random.choice(moves)


def board_str(b):
    s = [(b[i] if b[i] else str(i + 1)) for i in range(9)]
    return "\n".join(" " + " ".join(s[r * 3:r * 3 + 3]) for r in range(3))


# =============================================================================
# Scene (MJCF)
# =============================================================================
def get_scene_xml():
    xl, xw = 0.014, 0.0045
    pieces = []
    for k, (x, y) in enumerate(X_TRAY):
        pieces.append(
            f'<body name="x_piece_{k + 1}" pos="{x:.4f} {y:.4f} {PIECE_H / 2 + 5e-4:.4f}"><freejoint/>'
            f'<geom class="piece" type="box" size="{xl} {xw} {PIECE_H / 2}" euler="0 0 0.785398" material="x_mat" mass="0.006"/>'
            f'<geom class="piece" type="box" size="{xl} {xw} {PIECE_H / 2}" euler="0 0 -0.785398" material="x_mat" mass="0.006"/>'
            f'</body>')
    for k, (x, y) in enumerate(O_TRAY):
        pieces.append(
            f'<body name="o_piece_{k + 1}" pos="{x:.4f} {y:.4f} {PIECE_H / 2 + 5e-4:.4f}"><freejoint/>'
            f'<geom class="piece" type="cylinder" size="{PIECE_R} {PIECE_H / 2}" material="o_mat" mass="0.012"/>'
            f'</body>')
    pieces = "\n        ".join(pieces)
    half = 1.5 * CELL + 0.015
    ln = 1.5 * CELL
    return f"""<mujoco model="so101_tictactoe">
    <compiler angle="radian"/>
    <option timestep="0.002" integrator="implicitfast"/>

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="90" elevation="-55" offwidth="{CAM_W}" offheight="{CAM_H}"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.1"/>
        <material name="yellow_robot" rgba="0.95 0.75 0.1 1" reflectance="0.2" specular="0.3"/>

        <mesh name="base_motor_holder_so101_v1" file="assets/base_motor_holder_so101_v1.stl"/>
        <mesh name="base_so101_v2" file="assets/base_so101_v2.stl"/>
        <mesh name="sts3215_03a_v1" file="assets/sts3215_03a_v1.stl"/>
        <mesh name="waveshare_mounting_plate_so101_v2" file="assets/waveshare_mounting_plate_so101_v2.stl"/>
        <mesh name="motor_holder_so101_base_v1" file="assets/motor_holder_so101_base_v1.stl"/>
        <mesh name="rotation_pitch_so101_v1" file="assets/rotation_pitch_so101_v1.stl"/>
        <mesh name="upper_arm_so101_v1" file="assets/upper_arm_so101_v1.stl"/>
        <mesh name="under_arm_so101_v1" file="assets/under_arm_so101_v1.stl"/>
        <mesh name="motor_holder_so101_wrist_v1" file="assets/motor_holder_so101_wrist_v1.stl"/>
        <mesh name="sts3215_03a_no_horn_v1" file="assets/sts3215_03a_no_horn_v1.stl"/>
        <mesh name="wrist_roll_pitch_so101_v2" file="assets/wrist_roll_pitch_so101_v2.stl"/>
        <mesh name="Wrist_Roll_SO101" file="assets/Wrist_Roll_SO101.stl"/>
        <mesh name="Handle_SO101" file="assets/Handle_SO101.stl"/>
        <mesh name="Trigger_SO101" file="assets/Trigger_SO101.stl"/>

        <material name="board_mat" rgba="0.18 0.18 0.20 1"/>
        <material name="line_mat" rgba="0.85 0.85 0.85 1"/>
        <material name="x_mat" rgba="0.95 0.12 0.12 1"/>
        <material name="o_mat" rgba="0.12 0.40 1.0 1"/>
    </asset>

    <default>
        <!-- The arm is kinematic-looking only: no arm collisions (they caused the blow-ups).
             Only floor, board and pieces collide with each other. -->
        <geom contype="0" conaffinity="0"/>
        <joint armature="0.005" damping="0.05"/>
        <default class="piece">
            <geom contype="1" conaffinity="1" friction="1.2 0.01 0.001" condim="3" solref="0.004 1"/>
        </default>
    </default>

    <worldbody>
        <light pos="0 0 3.5" dir="0 0 -1" directional="true" castshadow="false"/>
        <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane" contype="1" conaffinity="1"/>

        <!-- The robot's eye: straight down over the board. Image up = +x, image right = -y. -->
        <camera name="top_cam" pos="{BOARD_X} {BOARD_Y} 0.40" xyaxes="0 -1 0 1 0 0" fovy="40"/>

        <body name="robot_root" pos="0 0 0">
            <geom pos="-0.00636471 -9.94414e-05 -0.0024" quat="0.499998 0.5 0.500002 0.5" type="mesh" mesh="base_motor_holder_so101_v1" material="yellow_robot"/>
            <geom pos="-0.00636471 -8.97657e-09 -0.0024" quat="0.499998 0.5 0.500002 0.5" type="mesh" mesh="base_so101_v2" material="yellow_robot"/>
            <geom pos="0.0263353 -8.97657e-09 0.0437" quat="1 0 0 0" type="mesh" mesh="sts3215_03a_v1" material="yellow_robot"/>
            <geom pos="-0.0309827 -0.000199441 0.0474" quat="0.499998 0.5 0.500002 0.5" type="mesh" mesh="waveshare_mounting_plate_so101_v2" material="yellow_robot"/>
            <body name="shoulder_link" pos="0.0388353 -8.97657e-09 0.0624" quat="1.76036e-12 1.32679e-06 -1 -1.32679e-06" gravcomp="1">
              <inertial pos="-0.0307604 -1.66727e-05 -0.0252713" quat="0.999865 0.00149568 0.0097084 0.0132043" mass="0.100006" diaginertia="8.37836e-05 8.10388e-05 2.39552e-05"/>
              <joint name="shoulder_pan" range="-1.91986 1.91986" actuatorfrcrange="-10 10"/>
              <geom pos="-0.0303992 0.000422241 -0.0417" quat="0.499998 0.5 0.5 -0.500002" type="mesh" mesh="sts3215_03a_v1" material="yellow_robot"/>
              <geom pos="-0.0675992 -0.000177759 0.0158499" quat="0.499998 0.5 -0.5 0.500002" type="mesh" mesh="motor_holder_so101_base_v1" material="yellow_robot"/>
              <geom pos="0.0122008 2.22413e-05 0.0464" quat="0.707105 -0.707108 0 0" type="mesh" mesh="rotation_pitch_so101_v1" material="yellow_robot"/>
              <body name="upper_arm_link" pos="-0.0303992 -0.0182778 -0.0542" quat="0.499998 -0.5 -0.5 -0.500002" gravcomp="1">
                <inertial pos="-0.0898471 -0.00838224 0.0184089" quat="0.453195 0.454063 0.541891 0.542951" mass="0.103" diaginertia="0.000150873 0.000142487 3.72451e-05"/>
                <joint name="shoulder_lift" range="-1.74533 1.74533" actuatorfrcrange="-10 10"/>
                <geom pos="-0.11257 -0.0155 0.0187" quat="9.38184e-07 -0.707105 0.707108 -9.38187e-07" type="mesh" mesh="sts3215_03a_v1" material="yellow_robot"/>
                <geom pos="-0.065085 0.012 0.0182" quat="1.32679e-06 1 0 0" type="mesh" mesh="upper_arm_so101_v1" material="yellow_robot"/>
                <body name="lower_arm_link" pos="-0.11257 -0.028 0" quat="0.707105 0 0 0.707108" gravcomp="1">
                  <inertial pos="-0.0980701 0.00324376 0.0182831" quat="0.510729 0.517006 0.487966 0.483477" mass="0.104" diaginertia="0.000160262 0.000145304 2.83125e-05"/>
                  <joint name="elbow_flex" range="-1.69 1.69" actuatorfrcrange="-10 10"/>
                  <geom pos="-0.0648499 -0.032 0.0182" quat="1.32679e-06 1 0 0" type="mesh" mesh="under_arm_so101_v1" material="yellow_robot"/>
                  <geom pos="-0.0648499 -0.032 0.018" quat="1.32679e-06 -1 0 0" type="mesh" mesh="motor_holder_so101_wrist_v1" material="yellow_robot"/>
                  <geom pos="-0.1224 0.0052 0.0187" quat="1.76037e-12 -1.32679e-06 1 -1.32679e-06" type="mesh" mesh="sts3215_03a_v1" material="yellow_robot"/>
                  <body name="wrist_link" pos="-0.1349 0.0052 0" quat="0.707105 0 0 -0.707108" gravcomp="1">
                    <inertial pos="-0.000103312 -0.0386143 0.0281156" quat="0.967143 0.254226 0.00162226 -0.000146307" mass="0.079" diaginertia="3.68265e-05 2.74474e-05 1.89434e-05"/>
                    <joint name="wrist_flex" range="-1.65806 1.65806" actuatorfrcrange="-10 10" armature="0.002"/>
                    <geom pos="0 -0.0424 0.0306" quat="0.499998 0.5 0.5 -0.500002" type="mesh" mesh="sts3215_03a_no_horn_v1" material="yellow_robot"/>
                    <geom pos="0 -0.028 0.0181" quat="0.499998 -0.5 -0.5 -0.500002" type="mesh" mesh="wrist_roll_pitch_so101_v2" material="yellow_robot"/>
                    <body name="gripper_link" pos="0 -0.0611 0.0181" quat="0.0172101 -0.0172081 0.706899 0.706896" gravcomp="1">
                      <inertial pos="-0.0146393 0.000237056 -0.0371643" quat="0.621616 0.32141 0.315466 0.640914" mass="0.117594" diaginertia="0.000178862 0.000173985 2.64788e-05"/>
                      <joint name="wrist_roll" range="-2.74385 2.84121" actuatorfrcrange="-10 10" armature="0.002"/>
                      <geom pos="0.0077 0.0001 -0.0234" quat="0.707105 -0.707108 0 0" type="mesh" mesh="sts3215_03a_v1" material="yellow_robot"/>
                      <geom quat="0 1 0 0" type="mesh" mesh="Wrist_Roll_SO101" material="yellow_robot"/>
                      <geom quat="0 1 0 0" type="mesh" mesh="Handle_SO101" material="yellow_robot"/>
                      <!-- Fingertip / tool-centre-point. This is what must reach the pieces. -->
                      <site name="tcp" pos="{TCP_LOCAL[0]} {TCP_LOCAL[1]} {TCP_LOCAL[2]}" size="0.004" rgba="1 0 1 1"/>
                      <!-- Claw jaw pads: they collide with the pieces and hold them by friction. The inner
                           faces sit at tcp +/- (joint position), so joint qpos = half the jaw gap. -->
                      <body name="pad_a" pos="{TCP_LOCAL[0] + PAD_T} {TCP_LOCAL[1]} {TCP_LOCAL[2] + PAD_DZ}" gravcomp="1">
                        <joint name="pad_a" type="slide" axis="1 0 0" range="0 0.03" damping="1" armature="0.0005"/>
                        <geom class="piece" type="box" size="{PAD_T} {PAD_W} {PAD_HH}" rgba="0.12 0.12 0.14 1" friction="2.0 0.02 0.002" mass="0.01"/>
                      </body>
                      <body name="pad_b" pos="{TCP_LOCAL[0] - PAD_T} {TCP_LOCAL[1]} {TCP_LOCAL[2] + PAD_DZ}" gravcomp="1">
                        <joint name="pad_b" type="slide" axis="-1 0 0" range="0 0.03" damping="1" armature="0.0005"/>
                        <geom class="piece" type="box" size="{PAD_T} {PAD_W} {PAD_HH}" rgba="0.12 0.12 0.14 1" friction="2.0 0.02 0.002" mass="0.01"/>
                      </body>
                      <body name="trigger_link" pos="0.0202 0.0188 -0.0234" quat="0.707105 0.707108 -1.85362e-08 1.85363e-08" gravcomp="1">
                        <inertial pos="-0.00561981 -0.0245945 0.00986744" quat="0.620232 0.237303 -0.427704 0.613244" mass="0.00648213" diaginertia="2.76121e-06 2.44232e-06 7.728e-07"/>
                        <joint name="gripper" range="0 0.785398" actuatorfrcrange="-10 10" armature="0.001"/>
                        <geom quat="0 0.34202 0.939693 0" type="mesh" mesh="Trigger_SO101" material="yellow_robot"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
        </body>

        <!-- Board: grid centred at ({BOARD_X}, {BOARD_Y}) -->
        <body name="ttt_board" pos="{BOARD_X} {BOARD_Y} 0">
            <geom name="board_base" type="box" size="{half} {half} {BOARD_TOP / 2}" pos="0 0 {BOARD_TOP / 2}" material="board_mat" contype="1" conaffinity="1"/>
            <geom type="box" size="{ln} 0.0007 0.0004" pos="0 {CELL / 2} {BOARD_TOP + 0.0004}" material="line_mat"/>
            <geom type="box" size="{ln} 0.0007 0.0004" pos="0 {-CELL / 2} {BOARD_TOP + 0.0004}" material="line_mat"/>
            <geom type="box" size="0.0007 {ln} 0.0004" pos="{CELL / 2} 0 {BOARD_TOP + 0.0004}" material="line_mat"/>
            <geom type="box" size="0.0007 {ln} 0.0004" pos="{-CELL / 2} 0 {BOARD_TOP + 0.0004}" material="line_mat"/>
        </body>

        {pieces}
    </worldbody>

    <actuator>
        <position name="shoulder_pan"  joint="shoulder_pan"  kp="200" kv="3"/>
        <position name="shoulder_lift" joint="shoulder_lift" kp="200" kv="3"/>
        <position name="elbow_flex"    joint="elbow_flex"    kp="150" kv="2.5"/>
        <position name="wrist_flex"    joint="wrist_flex"    kp="100" kv="1"/>
        <position name="wrist_roll"    joint="wrist_roll"    kp="60"  kv="0.6"/>
        <position name="gripper"       joint="gripper"       kp="60"  kv="0.5"/>
        <position name="pad_a" joint="pad_a" kp="{PAD_KP}" kv="{PAD_KV}" ctrllimited="true" ctrlrange="0.003 0.03" forcelimited="true" forcerange="-{PAD_FMAX} {PAD_FMAX}"/>
        <position name="pad_b" joint="pad_b" kp="{PAD_KP}" kv="{PAD_KV}" ctrllimited="true" ctrlrange="0.003 0.03" forcelimited="true" forcerange="-{PAD_FMAX} {PAD_FMAX}"/>
    </actuator>
</mujoco>
"""


# =============================================================================
# Computer vision  (identical code for simulation and real camera)
# =============================================================================
class BoardVision:
    """Rectifies the 3x3 grid with a homography and finds red (X) / blue (O) blobs."""

    def __init__(self, corners_px):
        """corners_px: 4x2 pixel positions of the OUTER corners of the 3x3 grid, in order
        corner of cell 1, corner of cell 3, corner of cell 9, corner of cell 7."""
        self.corners = np.float32(corners_px)
        n = GRID_PX
        dst = np.float32([[0, 0], [n, 0], [n, n], [0, n]])
        self.H = cv2.getPerspectiveTransform(self.corners, dst)
        self.kernel = np.ones((3, 3), np.uint8)

    def pixel_to_cell(self, px, py):
        """Camera pixel -> cell index 0..8 (or None if outside the grid). Used for mouse clicks."""
        v = self.H @ np.array([px, py, 1.0])
        if abs(v[2]) < 1e-9:
            return None
        x, y = v[0] / v[2], v[1] / v[2]
        c, r = int(x // (GRID_PX / 3)), int(y // (GRID_PX / 3))
        return r * 3 + c if (0 <= x < GRID_PX and 0 <= y < GRID_PX) else None

    def masks(self, warp_bgr):
        hsv = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, (0, RED_S, RED_V), (10, 255, 255)) | \
              cv2.inRange(hsv, (170, RED_S, RED_V), (180, 255, 255))
        blue = cv2.inRange(hsv, (BLUE_H[0], BLUE_S, BLUE_V), (BLUE_H[1], 255, 255))
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, self.kernel)
        blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, self.kernel)
        return red, blue

    def classify(self, frame_bgr):
        """Returns (board[9] with 'X'/'O'/None, debug image)."""
        n = GRID_PX
        cell = n / 3
        warp = cv2.warpPerspective(frame_bgr, self.H, (n, n))
        red, blue = self.masks(warp)
        board, best = [None] * 9, [0] * 9
        for label, mask in (("X", red), ("O", blue)):
            cnt, _, stats, cent = cv2.connectedComponentsWithStats(mask)
            for k in range(1, cnt):
                area = int(stats[k, cv2.CC_STAT_AREA])
                if area < MIN_BLOB_AREA:
                    continue
                cx, cy = cent[k]
                c, r = int(cx // cell), int(cy // cell)
                if not (0 <= c < 3 and 0 <= r < 3):
                    continue
                idx = r * 3 + c
                if area > best[idx]:
                    best[idx], board[idx] = area, label
        return board, self._debug(frame_bgr, warp, board)

    def _debug(self, frame, warp, board):
        n = GRID_PX
        cell = n // 3
        w = warp.copy()
        for i in range(1, 3):
            cv2.line(w, (i * cell, 0), (i * cell, n), (200, 200, 200), 1)
            cv2.line(w, (0, i * cell), (n, i * cell), (200, 200, 200), 1)
        for i in range(9):
            r, c = divmod(i, 3)
            txt = board[i] or str(i + 1)
            col = (0, 0, 255) if board[i] == "X" else (255, 120, 0) if board[i] == "O" else (160, 160, 160)
            cv2.putText(w, txt, (c * cell + 6, r * cell + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        f = frame.copy()
        cv2.polylines(f, [self.corners.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(f, "1", tuple(int(v) for v in self.corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        scale = 480 / f.shape[0]
        f = cv2.resize(f, (int(round(f.shape[1] * scale)), 480))
        w = cv2.resize(w, (480, 480))
        return np.hstack([f, w])


# =============================================================================
# Cameras
# =============================================================================
class SimCamera:
    def __init__(self, model, data, name="top_cam"):
        self.m, self.d, self.name = model, data, name
        self.cid = model.camera(name).id
        self.r = mujoco.Renderer(model, CAM_H, CAM_W)

    def read(self):
        self.r.update_scene(self.d, camera=self.name)
        return cv2.cvtColor(self.r.render(), cv2.COLOR_RGB2BGR)

    def close(self):
        try:
            self.r.close()
        except Exception:
            pass

    def project(self, pts_world):
        """Pinhole projection of world points -> pixels (uses the model's own camera)."""
        mujoco.mj_forward(self.m, self.d)
        pos = self.d.cam_xpos[self.cid]
        R = self.d.cam_xmat[self.cid].reshape(3, 3)       # columns = camera axes in world
        f = (CAM_H / 2) / np.tan(np.radians(self.m.cam_fovy[self.cid]) / 2)
        out = []
        for p in pts_world:
            c = R.T @ (np.asarray(p) - pos)               # x right, y up, z backwards
            depth = -c[2]
            out.append([CAM_W / 2 + f * c[0] / depth, CAM_H / 2 - f * c[1] / depth])
        return np.array(out, np.float32)

    def grid_corners(self):
        z = BOARD_TOP + PIECE_H * 0.8          # plane where the piece blobs live
        h = 1.5 * CELL
        world = [(BOARD_X + h, BOARD_Y - h, z), (BOARD_X + h, BOARD_Y + h, z),
                 (BOARD_X - h, BOARD_Y + h, z), (BOARD_X - h, BOARD_Y - h, z)]
        return self.project(world)


class RealCamera:
    def __init__(self, index=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {index}")

    def read(self):
        for _ in range(2):          # drop stale buffered frames
            self.cap.grab()
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        return frame

    def close(self):
        self.cap.release()

    def calibrate(self, path: Path):
        """Click the 4 outer corners of the 3x3 grid: cell 1, cell 3, cell 9, cell 7."""
        pts = []

        def cb(ev, x, y, *_):
            if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
                pts.append((x, y))

        win = "calibrate: click corners of cell1, cell3, cell9, cell7 | r=reset  s=save"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, cb)
        while True:
            img = self.read()
            for i, p in enumerate(pts):
                cv2.circle(img, p, 5, (0, 255, 0), -1)
                cv2.putText(img, str(i + 1), (p[0] + 8, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(win, img)
            k = cv2.waitKey(30) & 0xFF
            if k == ord("r"):
                pts.clear()
            if k == ord("s") and len(pts) == 4:
                break
        cv2.destroyWindow(win)
        path.write_text(json.dumps(pts))
        return np.float32(pts)


# =============================================================================
# Kinematics (shared by sim and real: the MJCF is the robot model)
# =============================================================================
class Kinematics:
    JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]

    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        ids = [model.joint(n).id for n in self.JOINTS]
        self.qadr = [int(model.jnt_qposadr[i]) for i in ids]
        self.dadr = [int(model.jnt_dofadr[i]) for i in ids]
        self.lo = np.array([model.jnt_range[i][0] for i in ids])
        self.hi = np.array([model.jnt_range[i][1] for i in ids])
        self.sid = model.site("tcp").id
        self.gid = model.body("gripper_link").id
        self.check_ids = [model.body(n).id for n in ("lower_arm_link", "wrist_link", "gripper_link")]
        self.jp = np.zeros((3, model.nv))
        self.jg = np.zeros((3, model.nv))

    def fk(self, q):
        """Fingertip (tcp) position for joint angles q."""
        self.d.qpos[:] = 0
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.sid].copy()

    def solve(self, target, q0, w_rot=0.35, iters=300):
        """Damped least squares: tcp position + 'gripper points straight down' (soft)."""
        m, d = self.m, self.d
        down = np.array([0.0, 0.0, -1.0])
        q = np.clip(np.asarray(q0, float), self.lo, self.hi)
        d.qpos[:] = 0
        pe = re = 1.0
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p, g = d.site_xpos[self.sid].copy(), d.xpos[self.gid].copy()
            v = p - g
            L = np.linalg.norm(v)
            a = v / L
            e_pos, e_rot = np.asarray(target) - p, down - a
            pe, re = np.linalg.norm(e_pos), np.linalg.norm(e_rot)
            if pe < 5e-4 and re < 0.05:
                break
            mujoco.mj_jacSite(m, d, self.jp, None, self.sid)
            mujoco.mj_jacBody(m, d, self.jg, None, self.gid)
            Jp = self.jp[:, self.dadr]
            Ja = (np.eye(3) - np.outer(a, a)) @ (Jp - self.jg[:, self.dadr]) / L
            J = np.vstack([Jp, w_rot * Ja])
            e = np.concatenate([e_pos, w_rot * e_rot])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.1, 0.1), self.lo, self.hi)
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, d)
        zmin = min(d.xpos[i][2] for i in self.check_ids)
        tilt = np.degrees(2 * np.arcsin(min(1.0, re / 2)))
        return q, pe, tilt, zmin


# =============================================================================
# Drivers: how joint commands reach a (simulated / real / no) robot
#   grasp(piece, g_cmd) -> bool   called after the jaws have closed: did we really get the piece?
#   release()                     called just before the jaws open
# =============================================================================
class SimDriver:
    def __init__(self, model, data, viewer):
        self.m, self.d, self.v = model, data, viewer
        self.names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        self.aid = [model.actuator(n).id for n in self.names]
        self.qadr = [int(model.jnt_qposadr[model.joint(n).id]) for n in self.names]
        self.pad_aid = [model.actuator(n).id for n in ("pad_a", "pad_b")]
        self.pad_qadr = [int(model.jnt_qposadr[model.joint(n).id]) for n in ("pad_a", "pad_b")]
        self.tcp = model.site("tcp").id
        self.sub = max(1, round(CTRL_DT / model.opt.timestep))

    def reset_pose(self, q4, g):
        for k in range(4):
            self.d.qpos[self.qadr[k]] = q4[k]
        self.d.qpos[self.qadr[5]] = g
        for a in self.pad_qadr:
            self.d.qpos[a] = jaw_half_gap(g)
        self.send(q4, g)
        mujoco.mj_forward(self.m, self.d)

    def read_q4(self):
        return np.array([self.d.qpos[a] for a in self.qadr[:4]])

    def read_g(self):
        return float(self.d.qpos[self.qadr[5]])

    def send(self, q4, g):
        for k in range(4):
            self.d.ctrl[self.aid[k]] = q4[k]
        self.d.ctrl[self.aid[4]] = WRIST_ROLL
        self.d.ctrl[self.aid[5]] = g
        for a in self.pad_aid:
            self.d.ctrl[a] = jaw_half_gap(g)

    def _piece_pos(self, body):
        b = self.m.body(body)
        adr = int(self.m.jnt_qposadr[b.jntadr[0]])
        return self.d.qpos[adr:adr + 3].copy()

    def grasp(self, piece, g_cmd):
        """True when the jaws were stopped by something (they cannot reach the commanded gap) AND that
        something is the piece. Pure physics: the claw must really be around it."""
        if piece is None:
            return True
        opening = float(np.mean([self.d.qpos[a] for a in self.pad_qadr]))
        stalled = opening > jaw_half_gap(g_cmd) + 0.0015
        return stalled and self.holding(piece)

    def holding(self, piece):
        """Is the piece still between the fingertips? (After a lift this proves friction really holds it.)"""
        if piece is None:
            return True
        off = self._piece_pos(piece) - self.d.site_xpos[self.tcp]
        ok = np.hypot(off[0], off[1]) < GRASP_TOL_XY and abs(off[2]) < GRASP_TOL_Z
        if not ok:
            print(f"(sim) piece is not in the claw: {np.hypot(off[0], off[1]) * 1000:.1f} mm off in xy, "
                  f"{off[2] * 1000:.1f} mm in z")
        return ok

    def release(self):
        pass

    def tick(self):
        t0 = time.perf_counter()
        for _ in range(self.sub):
            mujoco.mj_step(self.m, self.d)
        if not self.v.is_running():
            raise Quit
        self.v.sync()
        rest = CTRL_DT - (time.perf_counter() - t0)
        if rest > 0:
            time.sleep(rest)


class RealDriver:
    NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    def __init__(self, port, robot_id="my_follower"):
        try:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        except ImportError:
            from lerobot.common.robots.so101_follower import SO101Follower, SO101FollowerConfig
        cfg = SO101FollowerConfig(port=port, id=robot_id, use_degrees=True, max_relative_target=20.0)
        self.robot = SO101Follower(cfg)
        self.robot.connect()
        self.g_cmd = G_PICK

    def read_q4(self):
        obs = self.robot.get_observation()
        deg = np.array([obs[f"{n}.pos"] for n in self.NAMES[:4]])
        return np.radians((deg - REAL_OFFSET_DEG) / REAL_SIGN)

    def read_g(self):
        pct = self.robot.get_observation()["gripper.pos"]
        return float(np.interp(pct, REAL_G_PCT, G_BREAKS))     # exact inverse of send()

    def send(self, q4, g):
        deg = np.degrees(np.asarray(q4)) * REAL_SIGN + REAL_OFFSET_DEG
        a = {f"{n}.pos": float(deg[i]) for i, n in enumerate(self.NAMES[:4])}
        a["wrist_roll.pos"] = float(np.degrees(WRIST_ROLL))
        a["gripper.pos"] = float(np.interp(g, G_BREAKS, REAL_G_PCT))
        self.g_cmd = g
        self.robot.send_action(a)

    def grasp(self, piece, g_cmd):
        """The jaws were commanded to squeeze below the puck width. If a piece is in them the motor
        stalls on it, so the measured opening stays clearly above the command."""
        measured = self.read_g()
        ok = measured > g_cmd + REAL_GRASP_MARGIN
        print(f"(real) gripper command {g_cmd:.2f} rad, measured {measured:.2f} rad -> "
              f"{'piece held' if ok else 'NOTHING in the gripper'}")
        return ok

    def holding(self, piece):
        return self.grasp(piece, self.g_cmd)

    def release(self):
        pass

    def tick(self):
        time.sleep(CTRL_DT)

    def close(self):
        self.robot.disconnect()


class NullDriver:
    """--no-arm: run vision + engine on the real camera without moving anything."""

    def __init__(self):
        self.q, self.g = np.array([0.0, -1.0, 1.0, 0.5]), 0.3

    def read_q4(self):
        return self.q.copy()

    def read_g(self):
        return self.g

    def send(self, q4, g):
        self.q, self.g = np.asarray(q4, float), g

    def grasp(self, piece, g_cmd):
        return True

    def holding(self, piece):
        return True

    def release(self):
        pass

    def tick(self):
        time.sleep(CTRL_DT)


# =============================================================================
# Arm: IK + pick & place
# =============================================================================
class Arm:
    def __init__(self, kin: Kinematics, driver, slow=1.0):
        self.kin, self.drv, self.slow = kin, driver, slow
        self.rng = np.random.default_rng(0)
        self.on_tick = None
        self.q = driver.read_q4()
        self.g = driver.read_g()
        self.q_home = None
        self.at_home = True

    # -- time ---------------------------------------------------------------
    def tick(self):
        self.drv.tick()
        if self.on_tick:
            self.on_tick()

    def follow(self, qs, g, duration):
        """Smoothly run through the joint waypoints qs (and gripper to g) in `duration` seconds."""
        pts = [self.q.copy()] + [np.asarray(q, float) for q in qs]
        seg = len(qs)
        n = max(1, int(duration * self.slow / CTRL_DT))
        g0 = self.g
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)
            f = s * seg
            k = min(int(f), seg - 1)
            u = f - k
            self.drv.send(pts[k] + u * (pts[k + 1] - pts[k]), g0 + s * (g - g0))
            self.tick()
        self.q, self.g = pts[-1].copy(), g

    def move(self, q, g, duration):
        self.follow([q], g, duration)

    def hold_still(self, secs):
        for _ in range(max(1, int(secs * self.slow / CTRL_DT))):
            self.drv.send(self.q, self.g)
            self.tick()

    # -- inverse kinematics with continuity + random restarts ------------------
    def solve(self, target, label="", seed=None):
        first = self.q if seed is None else seed
        seeds = [first] + [self.rng.uniform(self.kin.lo, self.kin.hi) for _ in range(40)]
        best = None
        for s in seeds:
            q, pe, tilt, zmin = self.kin.solve(target, s)
            ok = pe < 2e-3 and zmin > MIN_LINK_Z
            score = pe + 0.002 * tilt + (0.0 if zmin > MIN_LINK_Z else 1.0)
            if best is None or score < best[0]:
                best = (score, q, pe, tilt, zmin)
            if ok:
                break
        _, q, pe, tilt, zmin = best
        if pe > 6e-3 or zmin <= MIN_LINK_Z:
            raise Unreachable(f"{label or target}: err {pe * 1000:.1f} mm, elbow/wrist z {zmin * 1000:.0f} mm")
        return q, pe, tilt

    def goto(self, x, y, z, g, duration=0.8, lin=None):
        """Move the fingertips to (x,y,z). After the first move out of the home pose the path is a
        straight line in Cartesian space (so a carried piece never dips into the board)."""
        tgt = np.array([x, y, z], float)
        label = f"({x:.3f},{y:.3f},{z:.3f})"
        if lin is None:
            lin = not self.at_home
        if not lin:
            q, _, _ = self.solve(tgt, label)
            qs = [q]
        else:
            p0 = self.kin.fk(self.q)
            n = max(1, int(np.ceil(np.linalg.norm(tgt - p0) / LIN_STEP)))
            qs, qprev = [], self.q
            for i in range(1, n + 1):
                q, _, _ = self.solve(p0 + (tgt - p0) * i / n, label, seed=qprev)
                if np.max(np.abs(q - qprev)) > MAX_JOINT_STEP:
                    raise Unreachable(f"{label}: IK branch flip on the straight path")
                qs.append(q)
                qprev = q
        self.follow(qs, g, duration)
        self.at_home = False

    def init_home(self):
        self.q_home = np.clip(HOME_Q, self.kin.lo, self.kin.hi)

    def go_home(self, g=None, duration=1.2):
        self.move(self.q_home, self.g if g is None else g, duration)
        self.at_home = True

    # -- task -----------------------------------------------------------------
    def pick_and_place(self, src, dst, piece=None):
        zp, zb, hz = z_floor(), z_board(), HOVER_DZ
        for attempt in range(GRASP_RETRIES):
            self.goto(src[0], src[1], zp + hz, G_PICK, 1.0)             # above the piece, claw open
            self.goto(src[0], src[1], zp, G_PICK, 0.7)                  # straight down around it
            self.move(self.q, G_HOLD, 0.6)                              # close the claw on the piece
            self.hold_still(0.4)                                        # let the grip settle
            ok = self.drv.grasp(piece, self.g)
            if ok:
                self.goto(src[0], src[1], zp + hz, G_HOLD, 0.8)         # lift straight up
                self.hold_still(0.3)
                ok = self.drv.holding(piece)                            # did friction really hold it?
            if ok:
                break
            print(f"Grasp attempt {attempt + 1} failed" + (" - retrying" if attempt + 1 < GRASP_RETRIES else ""))
            self.move(self.q, G_PICK, 0.5)
            self.goto(src[0], src[1], zp + hz, G_PICK, 0.7)
        else:
            self.go_home(G_PICK)
            raise GraspFailed("could not pick up the O piece")
        self.goto(dst[0], dst[1], zb + hz, G_HOLD, 1.4)                 # carry level over the board
        self.goto(dst[0], dst[1], zb, G_HOLD, 0.8)                      # lower onto the cell
        self.hold_still(0.2)
        self.drv.release()
        self.move(self.q, G_REL, 0.5)                                   # open just enough to let go
        self.hold_still(0.2)
        self.goto(dst[0], dst[1], zb + hz, G_REL, 0.7)                  # straight up, clear of neighbours
        self.go_home(G_PICK)


def check_reach(arm: Arm):
    """Offline IK report for every waypoint the game will ever use."""
    rows = []
    for i in range(9):
        x, y = cell_xy(i)
        rows += [(f"cell {i + 1} hover", (x, y, z_board() + HOVER_DZ)), (f"cell {i + 1} place", (x, y, z_board()))]
    for k, (x, y) in enumerate(O_TRAY):
        rows += [(f"O tray {k + 1} hover", (x, y, z_floor() + HOVER_DZ)), (f"O tray {k + 1} pick", (x, y, z_floor()))]
    bad = warn = 0
    print(f"{'waypoint':<18}{'err mm':>8}{'tilt deg':>10}   status")
    for name, t in rows:
        try:
            _, pe, tilt = arm.solve(t, name)
            flag = "ok"
            if tilt > WARN_TILT_DEG:
                warn += 1
                flag = "ok, but gripper is TILTED (grasp may be unreliable)"
            print(f"{name:<18}{pe * 1000:8.2f}{tilt:10.1f}   {flag}")
        except Unreachable as e:
            bad += 1
            print(f"{name:<18}{'':>8}{'':>10}   UNREACHABLE  ({e})")
    if bad:
        print(f"\n{bad} waypoint(s) unreachable: move the board/trays closer (--board-x) or shrink --cell.")
    elif warn:
        print(f"\nAll waypoints reachable, {warn} with a tilted gripper (> {WARN_TILT_DEG:.0f} deg).")
    else:
        print("\nAll waypoints reachable with a near-vertical gripper.")
    return bad == 0


# =============================================================================
# Simulation helper: a "virtual hand" that drops your X piece on the board
# =============================================================================
class VirtualHand:
    def __init__(self, model, data):
        self.m, self.d = model, data
        self.q = queue.Queue()
        if sys.stdin and sys.stdin.isatty():
            threading.Thread(target=self._reader, daemon=True).start()

    def request(self, cell):
        self.q.put(cell)

    def clear(self):
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass

    def _reader(self):
        while True:
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if not line:
                return
            tok = line.strip().split()
            if tok and tok[-1].isdigit() and 1 <= int(tok[-1]) <= 9:
                self.q.put(int(tok[-1]) - 1)
            elif tok:
                print("Type a cell number 1-9 (or click a cell in the camera window).")

    def poll(self, is_free):
        try:
            cell = self.q.get_nowait()
        except queue.Empty:
            return
        if not is_free(cell):
            print(f"(virtual hand) cell {cell + 1} is already taken")
            return
        # nearest X piece that is still sitting in the tray
        best, bd = None, 0.02
        for k, (tx, ty) in enumerate(X_TRAY):
            b = self.m.body(f"x_piece_{k + 1}")
            adr = int(self.m.jnt_qposadr[b.jntadr[0]])
            dist = np.hypot(self.d.qpos[adr] - tx, self.d.qpos[adr + 1] - ty)
            if dist < bd:
                best, bd = (b, adr), dist
        if not best:
            print("(virtual hand) no X pieces left in the tray")
            return
        b, adr = best
        x, y = cell_xy(cell)
        yaw = random.uniform(-0.4, 0.4)
        self.d.qpos[adr:adr + 3] = [x + random.uniform(-0.004, 0.004), y + random.uniform(-0.004, 0.004),
                                    BOARD_TOP + PIECE_H / 2 + 0.003]
        self.d.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        vadr = int(self.m.jnt_dofadr[b.jntadr[0]])
        self.d.qvel[vadr:vadr + 6] = 0


# =============================================================================
# The game loop: look -> understand -> think -> move -> verify
# =============================================================================
class Game:
    WIN = "robot eyes (q = quit)"

    def __init__(self, camera, vision, arm, hand=None, show=True, robot_first=False, wait_close=False):
        self.cam, self.vis, self.arm, self.hand = camera, vision, arm, hand
        self.show, self.robot_first, self.wait_close = show, robot_first, wait_close
        self.expected = [None] * 9
        self.o_used = 0
        self.geom = None            # (left panel width, scale) of the debug image, for mouse clicks
        arm.on_tick = self.pump
        if self.show:
            try:
                cv2.namedWindow(self.WIN)
                cv2.setMouseCallback(self.WIN, self._on_mouse)
            except cv2.error:
                self.show = False

    # -- UI / time ---------------------------------------------------------------
    def _on_mouse(self, ev, x, y, *_):
        """Left click on a cell (right panel, or the raw camera picture) = place your X there."""
        if ev != cv2.EVENT_LBUTTONDOWN or not self.hand or not self.geom:
            return
        left_w, scale = self.geom
        if x >= left_w:
            c, r = (x - left_w) * 3 // 480, y * 3 // 480
            cell = r * 3 + c if (0 <= c < 3 and 0 <= r < 3) else None
        else:
            cell = self.vis.pixel_to_cell(x / scale, y / scale)
        if cell is not None:
            self.hand.request(cell)

    def pump(self):
        if self.show:
            try:
                k = cv2.waitKey(1) & 0xFF
            except cv2.error:
                self.show = False
                return
            if k == ord("q"):
                raise Quit
            if self.hand and ord("1") <= k <= ord("9"):
                self.hand.request(k - ord("1"))

    def idle(self, secs, human=False):
        """Wait `secs`. The virtual hand only acts while it is the human's turn."""
        for _ in range(max(1, round(secs / CTRL_DT))):
            self.arm.tick()
            if human and self.hand:
                self.hand.poll(lambda c: self.expected[c] is None)

    # -- perception -----------------------------------------------------------------
    def see(self):
        frame = self.cam.read()
        board, dbg = self.vis.classify(frame)
        self.geom = (dbg.shape[1] - 480, 480 / frame.shape[0])
        if self.show:
            try:
                cv2.imshow(self.WIN, dbg)
            except cv2.error:
                print("(no OpenCV GUI available - running without the vision window)")
                self.show = False
        return board

    def stable_board(self, human=False):
        last, n = None, 0
        while True:
            self.idle(STABLE_DT, human)
            b = self.see()
            n = n + 1 if b == last else 1
            last = b
            if n >= STABLE_FRAMES:
                return b

    # -- one full game ------------------------------------------------------------------
    def play(self):
        print("\n--- TIC-TAC-TOE READY ---  you are X (red), the arm is O (blue)")
        if self.hand:
            print("Click a cell in the 'robot eyes' window, press 1-9 there, or type 1-9 here.")
        print(board_str(self.expected), "\n")
        turn = "O" if self.robot_first else "X"
        warned = None
        while not (winner(self.expected) or all(self.expected)):
            if turn == "X":
                obs = self.stable_board(human=True)
                if obs == self.expected:
                    continue
                diff = [i for i in range(9) if obs[i] != self.expected[i]]
                if len(diff) == 1 and self.expected[diff[0]] is None and obs[diff[0]] == "X":
                    c = diff[0]
                    self.expected[c] = "X"
                    print(f"I see your X in cell {c + 1}.")
                    warned, turn = None, "O"
                elif obs != warned:
                    warned = obs
                    print("Board doesn't match the game (expected only ONE new X). "
                          f"I see: {''.join(o or '.' for o in obs)}  expected: {''.join(o or '.' for o in self.expected)}"
                          " - fix the board and I'll re-check.")
            else:
                if self.hand:
                    self.hand.clear()               # ignore clicks made while it is not your turn
                if winner(self.expected) or all(self.expected):
                    break
                move = best_move(self.expected)
                print(f"My move: cell {move + 1}")
                k = self.o_used
                if k >= len(O_TRAY):
                    print("Out of O pieces!")
                    break
                self.o_used += 1
                try:
                    self.arm.pick_and_place(O_TRAY[k], cell_xy(move), f"o_piece_{k + 1}")
                except (Unreachable, GraspFailed) as e:
                    print("Cannot complete my move:", e)
                    try:
                        self.arm.go_home(G_PICK)
                    except Exception:
                        pass
                    break
                self.expected[move] = "O"
                print(board_str(self.expected), "\n")
                turn = "X"
                # closed loop: make sure the camera really sees the O where it should be
                t0, said = time.time(), False
                while True:
                    obs = self.stable_board()
                    if obs == self.expected:
                        break
                    if not said or time.time() - t0 > 8:
                        print("Waiting for the camera to see my O in the right place...")
                        said, t0 = True, time.time()
        w = winner(self.expected)
        print("\n" + board_str(self.expected))
        print(f"\nGame over: {'I win (O)' if w == 'O' else 'you win (X)' if w == 'X' else 'draw'}")
        if self.wait_close:
            while True:
                self.stable_board()


# =============================================================================
# Entry point
# =============================================================================
def load_or_calibrate(cam: RealCamera, recal: bool):
    path = HERE / "board_corners.json"
    if path.exists() and not recal:
        return np.float32(json.loads(path.read_text()))
    return cam.calibrate(path)


def main():
    global BOARD_X, BOARD_Y, CELL, PIECE_H
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true", help="use the real SO-101 + webcam")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--recalibrate", action="store_true", help="re-click the 4 board corners")
    ap.add_argument("--no-arm", action="store_true", help="real camera, but never move the arm")
    ap.add_argument("--check", action="store_true", help="print IK reachability report and exit")
    ap.add_argument("--robot-first", action="store_true")
    ap.add_argument("--no-window", action="store_true", help="hide the OpenCV 'robot eyes' window")
    ap.add_argument("--board-x", type=float, default=BOARD_X)
    ap.add_argument("--board-y", type=float, default=BOARD_Y)
    ap.add_argument("--cell", type=float, default=CELL)
    ap.add_argument("--piece-h", type=float, default=PIECE_H)
    a = ap.parse_args()
    BOARD_X, BOARD_Y, CELL, PIECE_H = a.board_x, a.board_y, a.cell, a.piece_h
    rebuild_layout()

    if mujoco is None:
        sys.exit("MuJoCo is required (also for the real robot: it provides the kinematics).  pip install mujoco")
    if not (HERE / "assets").is_dir():
        sys.exit(f"Missing folder {HERE / 'assets'} with the SO-101 .stl meshes.")

    scene = HERE / "ttt_scene_auto.xml"
    scene.write_text(get_scene_xml())
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    kin = Kinematics(m)
    cam = None
    drv = None

    try:
        if a.real:
            cam = RealCamera(a.camera)
            corners = load_or_calibrate(cam, a.recalibrate)
            drv = NullDriver() if a.no_arm else RealDriver(a.port)
            arm = Arm(kin, drv, slow=REAL_SPEED)
            arm.init_home()
            if a.check:
                arm.q = arm.q_home.copy()
                check_reach(arm)
                return
            if not a.no_arm:
                input("Clear the workspace. Press Enter to move the arm to its home pose...")
            arm.go_home(G_PICK, 3.0)
            Game(cam, BoardVision(corners), arm, None, not a.no_window, a.robot_first).play()
            return

        # ---------------- simulation ----------------
        if a.check:
            arm = Arm(kin, SimDriver(m, d, type("V", (), {"is_running": lambda s: True, "sync": lambda s: None})()))
            arm.init_home()
            arm.q = arm.q_home.copy()
            check_reach(arm)
            return

        with mujoco.viewer.launch_passive(m, d) as v:
            v.cam.lookat[:] = [BOARD_X, 0.0, 0.02]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.75, 90, -55
            drv = SimDriver(m, d, v)
            arm = Arm(kin, drv)
            arm.init_home()
            drv.reset_pose(arm.q_home, G_PICK)
            arm.q, arm.g = arm.q_home.copy(), G_PICK
            cam = SimCamera(m, d)
            vision = BoardVision(cam.grid_corners())
            hand = VirtualHand(m, d)
            Game(cam, vision, arm, hand, not a.no_window, a.robot_first, wait_close=True).play()
    except Quit:
        pass
    finally:
        if drv is not None and hasattr(drv, "close"):
            try:
                drv.close()
            except Exception:
                pass
        if cam is not None and hasattr(cam, "close"):
            cam.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()
