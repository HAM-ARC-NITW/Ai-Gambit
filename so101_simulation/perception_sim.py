"""MuJoCo camera and physical test fixtures, separate from the RGB detector.

Only this adapter reads simulator coordinates. They are used to arrange test
scenes, move virtual human pieces, and score results -- never to classify images.
The generated XML/AprilTag PNGs live in perception_assets so you can inspect them.
"""
from pathlib import Path
import copy
import sys
import time
import os
import tempfile
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "perception_ws/src/ai_gambit_perception"
sys.path.insert(0, str(SOURCE))
from ai_gambit_perception.perception import (  # noqa: E402
    Perception, load_config, camera_calibration, transform, tray_centers, overlay)
from chess_robot import ChessRobot
from chess_coords import PARK_POSE, HOME_POSE, PIECE_REST_Z, square_to_pos


BUTTON_POSITION = (.22, -.45, 0)  # Outside the overhead camera's view and the arm's path.


def make_workspace(button=False):
    """Extend the original scene rather than maintaining a second robot model.
    `button` adds the player's big red "my move is done" button.
    """
    config = load_config()
    folder = HERE / "perception_assets"
    folder.mkdir(exist_ok=True)
    root = ET.parse(HERE / "chess_world.xml").getroot()
    for asset in root.find("asset"):
        if "file" in asset.attrib:
            asset.set("file", str((HERE / asset.get("file")).resolve()))
    visual = root.find("visual")
    ET.SubElement(visual, "quality", shadowsize="1024", offsamples="0")
    ET.SubElement(visual, "map", znear="0.01")
    visual.find("headlight").set("diffuse", "0.35 0.35 0.35")
    visual.find("headlight").set("ambient", "0.2 0.2 0.2")
    ET.SubElement(visual, "global") if visual.find("global") is None else None
    visual.find("global").set("offwidth", str(config["image_width"]))
    visual.find("global").set("offheight", str(config["image_height"]))
    asset, world = root.find("asset"), root.find("worldbody")
    asset.find("texture[@name='chess_grid']").set("rgb1", "0.72 0.62 0.48")
    world.find("light").set("diffuse", "0.55 0.55 0.55")
    world.find("light").set("ambient", "0.05 0.05 0.05")
    camera = world.find("camera[@name='top_camera']")
    camera.set("pos", " ".join(map(str, config["camera_position"])))
    camera.set("fovy", str(config["camera_fovy_degrees"]))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    for group in ("board", "rack", "tray"):
        origin = config["rack_origin" if group == "rack" else "board_origin"]
        rotation = config["rack_rotation" if group == "rack" else "board_rotation"]
        # All frames share the same orientation in this fixed simulation layout.
        body = ET.SubElement(world, "body", name=f"{group}_fiducials", pos=" ".join(map(str, origin)),
                             quat="0.707106781 0 0 -0.707106781")
        for marker, center in config[f"{group}_markers"].items():
            png = folder / f"tag_{marker}.png"
            # A white quiet zone is necessary for reliable tag detection.
            pixels = cv2.aruco.generateImageMarker(dictionary, int(marker), 192)
            pixels = cv2.copyMakeBorder(pixels, 24,24,24,24, cv2.BORDER_CONSTANT, value=255)
            if not png.exists():
                # Atomic replacement lets two simulation windows start together
                # without one reading an image the other is still writing.
                with tempfile.NamedTemporaryFile(dir=folder,suffix=".png",delete=False) as temporary:
                    temp_png = temporary.name
                cv2.imwrite(temp_png,pixels)
                os.replace(temp_png,png)
            ET.SubElement(asset, "texture", name=f"tag_tex_{marker}", type="2d", file=str(png))
            ET.SubElement(asset, "material", name=f"tag_mat_{marker}", texture=f"tag_tex_{marker}",
                          texrepeat="1 1", texuniform="false", emission="1", specular="0", reflectance="0")
            half = config["marker_size"] * 1.25 / 2
            ET.SubElement(body, "geom", name=f"tag_{marker}", type="box", size=f"{half} {half} .0001",
                          pos=f"{center[0]} {center[1]} .0002", material=f"tag_mat_{marker}",
                          contype="0", conaffinity="0")
    # Known fixtures have the same top plane as the board, simplifying calibration.
    ET.SubElement(world, "geom", name="promotion_rack", type="box", size=".020 .165 .005",
                  pos=".415 0 0.005", rgba=".32 .40 .48 1")
    for sign in (-1, 1):
        ET.SubElement(world, "geom", name=f"capture_tray_{sign}", type="box", size=".12 .045 .005",
                      pos=f".243 {sign*.195} .005", rgba=".38 .44 .50 1")
    # Two rows of sixteen labelled spare-piece slots. Compact cylinders are
    # intentional rack proxies, not a learned visual identity classifier.
    for row, color in enumerate(("w", "b")):
        for col in range(16):
            name = f"spare_{color}_{col:02d}"
            p = np.asarray(config["rack_origin"]) + np.asarray(config["rack_rotation"]) @ [col*.02, row*.02, .012]
            body = ET.SubElement(world, "body", name=name, pos=" ".join(map(str, p)))
            ET.SubElement(body, "freejoint", name=name+"_joint")
            ET.SubElement(body, "geom", type="cylinder", size=".006 .012", mass=".02",
                          material="white_mat" if color == "w" else "black_mat")
    # Movable test obstructions begin outside the camera's workspace.
    for name, size, color in (("hand", ".055 .035 .008", ".75 .40 .24 1"),
                              ("foreign", ".025 .025 .025", ".05 .2 .85 1")):
        body = ET.SubElement(world, "body", name=name, mocap="true", pos="0 0 -1")
        ET.SubElement(body, "geom", type="box", size=size, rgba=color, contype="0", conaffinity="0")
    if button:
        # Sprung cap on a slide joint: double-click it, or Ctrl+drag it down,
        # in the 3D viewer. It is appended last so existing qpos indices hold.
        housing = ET.SubElement(world, "body", name="big_red_button_base", pos=" ".join(map(str, BUTTON_POSITION)))
        ET.SubElement(housing, "geom", type="box", size=".045 .045 .015", pos="0 0 .015",
                      rgba=".12 .12 .12 1", contype="0", conaffinity="0")
        cap = ET.SubElement(housing, "body", name="big_red_button", pos="0 0 .03", gravcomp="1")
        ET.SubElement(cap, "joint", name="big_red_button_press", type="slide", axis="0 0 1",
                      range="-.012 0", limited="true", stiffness="40", damping="1.5")
        ET.SubElement(cap, "geom", type="cylinder", size=".032 .008", pos="0 0 .008", mass=".05",
                      rgba=".85 .05 .05 1", contype="0", conaffinity="0")
    path = folder / ("workspace_button.xml" if button else "workspace.xml")
    ET.indent(root)
    with tempfile.NamedTemporaryFile(dir=folder,suffix=".xml",delete=False) as temporary:
        temp_xml = temporary.name
    ET.ElementTree(root).write(temp_xml, encoding="utf-8", xml_declaration=True)
    os.replace(temp_xml,path)
    return path


class Simulation:
    def __init__(self, button=False):
        self.config = load_config()
        self.robot = ChessRobot(str(make_workspace(button)))
        self.model, self.data = self.robot.model, self.robot.data
        self.renderer = mujoco.Renderer(self.model, height=self.config["image_height"], width=self.config["image_width"])
        self.detector = Perception(self.config)
        self.piece_ids = [b for b in range(self.model.nbody)
                          if self.model.body(b).name.startswith(("w_", "b_", "spare_"))]
        self.initial_masks = (self.model.geom_contype.copy(), self.model.geom_conaffinity.copy())
        self.park(instant=True)
        # An independently arranged empty scene supplies the background photograph.
        saved = self.data.qpos.copy()
        for b in self.piece_ids:
            self.put(b, [0,0,-2])
        empty = self.frame()
        cv2.imwrite(str(HERE/"perception_assets/empty_workspace.png"),cv2.cvtColor(empty,cv2.COLOR_RGB2BGR))
        self.detector.set_empty_reference(empty)
        self.data.qpos[:] = saved
        self.model.geom_contype[:], self.model.geom_conaffinity[:] = self.initial_masks
        mujoco.mj_forward(self.model, self.data)
        self.initial_qpos = self.data.qpos.copy()
        self.initial_body_pos = self.model.body_pos.copy()
        self.initial_rgba = self.model.geom_rgba.copy()
        self.initial_geom_pos = self.model.geom_pos.copy()
        self.initial_cam_pos = self.model.cam_pos.copy()
        self.mapping = {}
        self.rack_inventory = [1]*32
        self.tray_inventory = [0]*48
        self.button_ids = ()
        if button:
            self.button_ids = (self.model.body("big_red_button").id, self.model.body("big_red_button_base").id)
            self.button_qpos = self.model.jnt_qposadr[self.model.joint("big_red_button_press").id]

    def button_down(self):
        return bool(self.button_ids) and self.data.qpos[self.button_qpos] < -.006

    def body_at(self, square, ignore=None):
        """The piece body physically standing on a square, or None."""
        import chess
        target = square_to_pos(chess.square_name(square))[:2]
        near = [(np.linalg.norm(self.data.xpos[b][:2]-target), b) for b in self.piece_ids
                if self.data.xpos[b][2] > 0 and b != ignore]
        distance, body = min(near)
        return body if distance < .012 else None

    def sync_mapping(self, board):
        """Simplified grasping attaches a known body, so after a human turn the
        control adapter must know which body the human left on each square.
        This plays the role of the gripper closing on whatever is there.
        """
        mapping = {}
        for square in board.piece_map():
            body = self.body_at(square)
            if body is None:
                raise ValueError("No graspable piece found on a confirmed square")
            mapping[square] = body
        self.mapping = mapping

    def frame(self):
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="top_camera")
        return self.renderer.render().copy()  # RGB, never BGR inside the pipeline.

    def snapshot(self, frames=4, effect=None, on_observation=None, **kwargs):
        """Fresh frames, rather than repeated calls using a cached timestamp."""
        for i in range(frames):
            image = self.frame()
            if effect:
                image = effect(image)
            obs = self.detector.process(image, **kwargs)
            # Refresh the UI between frames only. Window updates after the last
            # frame would age it past max_frame_age before the caller verifies it.
            if on_observation is not None and i < frames-1:
                on_observation(obs)
        return obs

    def reset(self):
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = 0
        self.data.ctrl[:6] = PARK_POSE
        self.model.body_pos[:] = self.initial_body_pos
        self.model.geom_rgba[:] = self.initial_rgba
        self.model.geom_pos[:] = self.initial_geom_pos
        self.model.cam_pos[:] = self.initial_cam_pos
        self.model.geom_contype[:], self.model.geom_conaffinity[:] = self.initial_masks
        self.data.mocap_pos[:] = [0,0,-1]
        self.detector.previous, self.detector.stable_count = None, 0
        self.detector.baseline = None
        self.mapping, self.rack_inventory, self.tray_inventory = {}, [1]*32, [0]*48
        mujoco.mj_forward(self.model, self.data)

    def put(self, body, position, quaternion=None):
        """Virtual human/test action. It must not be called by the RGB detector."""
        b = self.model.body(body).id if isinstance(body, str) else body
        joint = self.model.body_jntadr[b]
        q = self.model.jnt_qposadr[joint]
        dof = self.model.jnt_dofadr[joint]
        self.data.qpos[q:q+3] = position
        self.data.qpos[q+3:q+7] = [1,0,0,0] if quaternion is None else quaternion
        self.data.qvel[dof:dof+6] = 0
        # Hidden test pieces must not collide with the infinite ground plane.
        start, count = self.model.body_geomadr[b], self.model.body_geomnum[b]
        for geom in range(start,start+count):
            self.model.geom_contype[geom] = 0 if position[2] < 0 else self.initial_masks[0][geom]
            self.model.geom_conaffinity[geom] = 0 if position[2] < 0 else self.initial_masks[1][geom]

    def park(self, viewer=None, instant=False):
        if instant:
            self.data.qpos[:6], self.data.ctrl[:6] = PARK_POSE, PARK_POSE
            mujoco.mj_forward(self.model, self.data)
        else:
            self.robot.planner.execute_segment(self.data.qpos[:6].copy(), PARK_POSE, steps=250, viewer=viewer)
            self.robot.planner.execute_segment(PARK_POSE, PARK_POSE, steps=200, viewer=viewer)

    def load_position(self, board):
        """Arrange a rules-engine position for tests; identities are known inventory."""
        import chess
        self.reset()
        available = self.piece_ids[:32]
        for b in available:
            self.put(b, [0,0,-2])
        for square, piece in board.piece_map().items():
            prefix = ("w" if piece.color else "b") + "_" + chess.piece_name(piece.piece_type)
            candidates = [b for b in available if self.model.body(b).name.startswith(prefix)]
            if not candidates:
                start = 0 if piece.color else 16
                slot = next((i for i in range(start,start+16) if self.config["rack_types"][i] == chess.piece_name(piece.piece_type) and self.rack_inventory[i]),None)
                if slot is None:
                    raise ValueError("FEN exceeds available original and promotion-piece inventory")
                b = self.model.body(f"spare_{'w' if piece.color else 'b'}_{slot%16:02d}").id
                self.rack_inventory[slot] = 0
            else:
                b = candidates[0]
                available.remove(b)
            self.mapping[square] = b
            rest = .022 if self.model.body(b).name.startswith("spare_") else PIECE_REST_Z[chess.piece_name(piece.piece_type)]
            self.put(b, square_to_pos(chess.square_name(square),rest-.01))
        mujoco.mj_forward(self.model, self.data)

    def storage_slot(self, color):
        start = 0 if color else 24
        return next((i for i in range(start,start+24) if self.tray_inventory[i] == 0), None)

    def physical_move(self, board, move, animate=False, viewer=None, speed=1, observation=None):
        """Control adapter: execute the engine's ordered transfers, including all
        chess special moves. The caller commits rules/inventory only after vision
        verifies the final scene. Returns expected inventory, not observed data.
        """
        import chess
        mapping = self.mapping.copy()
        rack, tray = self.rack_inventory.copy(), self.tray_inventory.copy()
        sources = {}
        board_pose = np.array(observation.poses["board"]) if observation is not None else transform(self.config["board_rotation"],self.config["board_origin"])

        def board_target(square, measured_offset=False):
            local = np.array([(chess.square_file(square)+.5)*.03,
                              (chess.square_rank(square)+.5)*.03,0,1.])
            if measured_offset and observation is not None:
                local[:2] += observation.offsets[square]
            return (board_pose @ local)[:3]

        def tray_target(slot):
            target = tray_centers()[slot]
            if observation is not None:
                nominal = transform(self.config["board_rotation"],self.config["board_origin"])
                target = (np.array(observation.poses["tray"]) @ np.linalg.inv(nominal) @ [*target,1])[:3]
            return target

        if animate and (observation is None or observation.status != "READY"):
            raise ValueError("Animated robot motion requires a clear calibrated camera observation")
        if observation is not None:
            # Identities come from the coordinator's inventory; grasp positions
            # come from camera geometry, not simulator body coordinates.
            sources = {body:board_target(square,True) for square,body in mapping.items()}
        operations = []
        piece = board.piece_at(move.from_square)
        if piece is None:
            raise ValueError("No known piece at source")
        if board.is_capture(move):
            capture_square = move.to_square
            if board.is_en_passant(move):
                capture_square += -8 if piece.color else 8
            captured = board.piece_at(capture_square)
            slot = self.storage_slot(captured.color)
            if slot is None:
                raise ValueError("Capture storage full")
            b = mapping.pop(capture_square)
            operations.append((b, tray_target(slot), chess.piece_name(captured.piece_type)))
            tray[slot] = 1
        b = mapping.pop(move.from_square)
        dest = board_target(move.to_square)
        operations.append((b, dest, chess.piece_name(piece.piece_type)))
        mapping[move.to_square] = b
        if board.is_castling(move):
            rank = 0 if piece.color else 7
            rook_from = chess.square(7 if move.to_square > move.from_square else 0, rank)
            rook_to = chess.square(5 if move.to_square > move.from_square else 3, rank)
            rook = mapping.pop(rook_from)
            operations.append((rook, board_target(rook_to), "rook"))
            mapping[rook_to] = rook
        if move.promotion:
            # Types are assigned by the coordinator's labelled inventory, not
            # inferred from cylinder appearance. Edit rack_types to change loading.
            start = 0 if piece.color else 16
            slot = next((i for i in range(start,start+16) if self.config["rack_types"][i] == chess.piece_name(move.promotion) and rack[i]), None)
            tray_start = 0 if piece.color else 24
            retired_slot = next((i for i in range(tray_start,tray_start+24) if tray[i] == 0), None)
            if slot is None or retired_slot is None:
                raise ValueError("No verified spare piece or retirement storage available")
            operations.append((b, tray_target(retired_slot), "pawn"))
            spare = self.model.body(f"spare_{'w' if piece.color else 'b'}_{slot%16:02d}").id
            if observation is not None:
                rack_pose = np.array(observation.poses["rack"])
                sources[spare] = (rack_pose @ [slot%16*.02,slot//16*.02,0,1])[:3]
            operations.append((spare, dest, "pawn"))  # Rack proxies have pawn-sized grasp geometry.
            mapping[move.to_square] = spare
            rack[slot], tray[retired_slot] = 0, 1
        # Validate every target before starting, so unreachable rack slots cannot
        # cause a partially executed promotion. Motion planning remains Control's job.
        if animate:
            for body, target, kind in operations:
                for position in (sources[body],target):
                    for height in (.13,.01 + {"pawn":.02,"rook":.02,"knight":.024,"bishop":.024,"queen":.028,"king":.03}[kind]):
                        point = position.copy()
                        point[2] = height
                        _, ok, _ = self.robot.ik.solve(point)
                        if not ok:
                            raise ValueError("Requested approach/grasp target is outside validated arm reach")
            # The original planner expects approach from HOME; returning here also
            # prevents a parked-to-pick swing through nearby pieces.
            self.robot.planner.execute_segment(self.data.qpos[:6].copy(), HOME_POSE, steps=250, viewer=viewer, speed_factor=speed)
            self.robot.planner.execute_segment(HOME_POSE, HOME_POSE, steps=200, viewer=viewer, speed_factor=speed)
        for body, target, kind in operations:
            if animate:
                self.robot._execute_transfer(body, sources[body], target, kind, viewer=viewer, speed=speed)
                sources[body] = target  # A promotion transfers its pawn twice.
            else:
                p = np.asarray(target).copy()
                p[2] = .022 if self.model.body(body).name.startswith("spare_") else PIECE_REST_Z[kind]
                self.put(body, p)
        if animate:
            self.park(viewer)
        mujoco.mj_forward(self.model, self.data)
        # Pending control mapping is returned, then committed by the coordinator.
        return mapping, rack, tray

    def close(self):
        self.renderer.close()


def show_observation(sim, obs, title="Perception", message=""):
    """Two visible views: raw overhead RGB plus the measured square grid."""
    rgb = sim.frame()
    cv2.putText(rgb, f"{obs.status}: {obs.reason}", (15,30), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,220,60), 2)
    cv2.putText(rgb, message, (15,58), cv2.FONT_HERSHEY_SIMPLEX, .50, (255,255,255), 1)
    cv2.imshow(title+" - overhead camera", cv2.cvtColor(cv2.resize(rgb,(960,720)), cv2.COLOR_RGB2BGR))
    cv2.imshow(title+" - measured occupancy", cv2.cvtColor(overlay(obs),cv2.COLOR_RGB2BGR))
