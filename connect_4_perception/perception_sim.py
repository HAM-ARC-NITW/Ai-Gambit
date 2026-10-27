"""MuJoCo adapter: arrange scenes, render RGB, and score the independent detector.

Only this file knows physical disc positions. Perception sees pixels/calibration.
The arm remains parked: releases are scripted to test perception, not a claim of
MoveIt manipulation. Discs then fall under gravity into separate collision guides.
"""
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(ROOT/"perception_ws/src/ai_gambit_perception"))
from ai_gambit_perception.connect4 import (Perception,load_config,transform,EMPTY,RED,YELLOW)

PARK = [-1.4,-.8,1.2,-.4,0,.1]
FRAME_QUAT = [.5,.5,.5,.5]  # Local X=world Y, local Y=world Z, normal=world X.


def workspace(folder=None):
    """Add readable test fixtures to the supplied model; keep the original STLs."""
    folder = Path(folder or HERE/"perception_assets").resolve()
    folder.mkdir(parents=True,exist_ok=True)
    cfg = load_config()
    root = ET.parse(HERE/"connect4_new.xml").getroot()
    root.find("include").set("file",str(ROOT/"simulation/so101_new_calib.xml"))
    root.find("compiler").set("meshdir",str(ROOT/"simulation/assets"))
    root.find("option").set("integrator","implicitfast")
    asset,world = root.find("asset"),root.find("worldbody")
    for mesh in asset.findall("mesh"):
        mesh.set("file",str(HERE/("connect4_board.stl" if mesh.get("name")=="board_mesh" else "connect4_token.stl")))
    # Original disc diameter is 15 mm but the board's row pitch is 19.4076 mm.
    # Match its diameter to the pitch so a real gravity stack aligns with holes.
    # Keep its 11 mm thickness: uniformly scaling it would exceed the board gap.
    disc_scale = cfg["disc_radius"]/.0075
    asset.find("mesh[@name='token_mesh']").set("scale",f"{disc_scale*.001} {disc_scale*.001} .001")
    asset.find("material[@name='board_mat']").set("rgba",".04 .18 .65 1")
    ET.SubElement(asset,"material",name="yellow_disk",rgba="1 .85 .04 1",specular="0")
    world.remove(world.find("body[@name='board']"))
    world.remove(world.find("body[@name='red_token_1']"))
    root.remove(root.find("keyframe"))  # Free discs add joints; initialize explicitly.
    visual = root.find("visual")
    visual.find("global").set("offwidth",str(cfg["image_width"]))
    visual.find("global").set("offheight",str(cfg["image_height"]))
    # Both the RGB renderer and passive viewer allocate GL buffers. Shadows are
    # disabled below; keep their unused buffer small on Windows laptops too.
    ET.SubElement(visual,"quality",offsamples="0",shadowsize="256")
    camera = world.find("camera[@name='top_camera']")
    camera.set("name","front_camera")
    camera.set("pos"," ".join(map(str,cfg["camera_position"])))
    camera.set("xyaxes","0 1 0 0 0 1")
    camera.set("fovy",str(cfg["camera_fovy"]))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

    def box(parent,name,pos,size,visible=False,**extra):
        return ET.SubElement(parent,"geom",name=name,type="box",pos=" ".join(map(str,pos)),
            size=" ".join(map(str,size)),rgba=".13 .13 .13 1" if visible else "0 0 0 0",**extra)

    for group in ("board","feeder"):
        fixture = ET.SubElement(world,"body",name=group+"_fixture",
            pos=" ".join(map(str,cfg[group+"_position"])),quat=".5 .5 .5 .5")
        for marker,center in cfg[group+"_tags"].items():
            image = cv2.aruco.generateImageMarker(dictionary,int(marker),192)
            image = cv2.copyMakeBorder(image,24,24,24,24,cv2.BORDER_CONSTANT,value=255)
            path = folder/f"tag_{marker}.png"
            cv2.imwrite(str(path),image)
            ET.SubElement(asset,"texture",name="tag_tex_"+marker,type="2d",file=str(path))
            ET.SubElement(asset,"material",name="tag_mat_"+marker,texture="tag_tex_"+marker,
                emission="1",specular="0",texrepeat="1 1",texuniform="false")
            half = cfg["marker_size"]*1.25/2
            ET.SubElement(fixture,"geom",name="tag_"+marker,type="box",size=f"{half} {half} .0001",
                pos=f"{center[0]} {center[1]} .0003",material="tag_mat_"+marker,contype="0",conaffinity="0")
        if group=="board":
            # STL coordinates use an offset CAD origin. This transform puts frame
            # (0,0,0) at the measured centre of the bottom-left front opening.
            ET.SubElement(fixture,"geom",name="board_visual",type="mesh",mesh="board_mesh",
                pos=".123949 -.0183108 -.050328518",quat=".707106781 -.707106781 0 0",
                material="board_mat",contype="0",conaffinity="0")
            px,py = cfg["pitch_x"],cfg["pitch_y"]
            box(fixture,"board_backing",[3*px,2.5*py,-.032],[.105,.067,.001],True)
            # MuJoCo mesh collisions use a convex hull, which would seal every
            # hole/entry. Independent hidden walls preserve open vertical channels.
            radius = cfg["disc_radius"]
            bottom,top = -radius,6*py+.04
            for c in range(8):
                box(fixture,f"rail_{c}",[(c-.5)*px,(top+bottom)/2,-.00886],[.0021,(top-bottom)/2,.008])
            box(fixture,"bottom_stop",[3*px,bottom-.001,-.00886],[3.5*px,.001,.008])
            for face,z in (("front",-.001),("back",-.01672)):
                box(fixture,"guide_"+face,[3*px,(top+bottom)/2,z],[3.5*px,(top-bottom)/2,.0005])
        else:
            pitch = cfg["feeder_pitch"]
            box(fixture,"feeder_backing",[2.5*pitch,3*pitch,-.032],[.080,.091,.001],True)
            for r in range(7):
                box(fixture,f"supply_ledge_{r}",[2.5*pitch,r*pitch-cfg["disc_radius"]-.001,-.00886],
                    [3*pitch,.001,.007])

    rotation = np.asarray(cfg["frame_rotation"])
    for slot in range(42):
        colour = RED if slot<21 else YELLOW
        position = np.asarray(cfg["feeder_position"])+rotation @ [slot%6*cfg["feeder_pitch"],slot//6*cfg["feeder_pitch"],-.00886]
        body = ET.SubElement(world,"body",name=f"disc_{slot}",pos=" ".join(map(str,position)),
            quat=".5 .5 .5 .5",gravcomp="1")
        # A planar guide keeps discs upright while allowing XYZ translation.
        # Gravity/contact determine the landing height. This avoids pretending
        # the coarse collision guide validates realistic wobble or jamming.
        for axis,direction in zip("xyz",("1 0 0","0 1 0","0 0 1")):
            ET.SubElement(body,"joint",name=f"disc_{slot}_{axis}",type="slide",axis=direction,damping=".02")
        material = "red_disk" if colour==RED else "yellow_disk"
        center = [.0725*disc_scale,-.03889148*disc_scale,-.016337412]
        ET.SubElement(body,"geom",name=f"disc_visual_{slot}",type="mesh",mesh="token_mesh",
            pos=" ".join(map(str,center)),material=material,contype="0",conaffinity="0")
        ET.SubElement(body,"geom",name=f"disc_collision_{slot}",type="cylinder",
            size=f"{cfg['disc_radius']} {cfg['disc_thickness']/2}",mass=".01",rgba="0 0 0 0",
            friction=".3 .002 .0001",solref=".008 1")
    # Obstructions are separate physical scene objects, not painted detector output.
    body = ET.SubElement(world,"body",name="obstruction",mocap="true",pos="0 0 -2")
    ET.SubElement(body,"geom",name="hand",type="box",size=".008 .035 .030",rgba=".72 .42 .24 1",contype="0",conaffinity="0")
    path = folder/"workspace.xml"
    ET.indent(root)
    ET.ElementTree(root).write(path,encoding="utf-8",xml_declaration=True)
    return path


class Simulation:
    def __init__(self,folder=None):
        self.config = load_config()
        self.folder = Path(folder or HERE/"perception_assets").resolve()
        self.model = mujoco.MjModel.from_xml_path(str(workspace(self.folder)))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model,self.config["image_height"],self.config["image_width"])
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        self.detector = Perception(self.config)
        self.discs = [self.model.body(f"disc_{i}").id for i in range(42)]
        self.colours = [RED]*21+[YELLOW]*21
        self.arm_home = np.array(PARK)
        self.data.qpos[:6],self.data.ctrl[:6] = self.arm_home,self.arm_home
        for i,b in enumerate(self.discs):
            self.put(b,[3+i*.03,3,.1])
        # Acquire the reference with the arm already settled in its parked pose.
        # Otherwise its initial gravity sag looks like a new apron obstruction.
        for joint in range(6):
            self.model.body_gravcomp[self.model.jnt_bodyid[joint]] = 1
        mujoco.mj_step(self.model,self.data,nstep=300)
        self.arm_home = self.data.qpos[:6].copy()
        self.data.ctrl[:6] = self.arm_home
        mujoco.mj_forward(self.model,self.data)
        empty,stamp = self.frame()
        cv2.imwrite(str(self.folder/"empty_workspace.png"),cv2.cvtColor(empty,cv2.COLOR_RGB2BGR))
        self.detector.set_empty_reference(empty)
        self.initial_body_pos,self.initial_body_quat = self.model.body_pos.copy(),self.model.body_quat.copy()
        self.initial_rgba,self.initial_camera = self.model.geom_rgba.copy(),self.model.cam_pos.copy()
        self.initial_material = self.model.geom_matid.copy()
        self.initial_geom_pos,self.initial_geom_size = self.model.geom_pos.copy(),self.model.geom_size.copy()
        self.load(np.zeros((6,7),np.uint8))

    def pose(self,group):
        return transform(self.config["frame_rotation"],self.config[group+"_position"])

    def point(self,group,x,y,depth=-.00886):
        return (self.pose(group) @ [x,y,depth,1])[:3]

    def put(self,body,position):
        joint = int(self.model.body_jntadr[body])
        q = self.model.jnt_qposadr[joint]
        v = self.model.jnt_dofadr[joint]
        # Slider axes follow the disc's fixed local frame.
        self.data.qpos[q:q+3] = np.asarray(self.config["frame_rotation"]).T @ (np.asarray(position)-self.model.body_pos[body])
        self.data.qvel[v:v+3] = 0

    def load(self,cells):
        """Arrange a known test board and remaining supply; never feed it to CV."""
        cells = np.asarray(cells).reshape(6,7)
        self.model.body_pos[:],self.model.body_quat[:] = self.initial_body_pos,self.initial_body_quat
        self.model.geom_rgba[:],self.model.cam_pos[:] = self.initial_rgba,self.initial_camera
        self.model.geom_matid[:] = self.initial_material
        self.model.geom_pos[:],self.model.geom_size[:] = self.initial_geom_pos,self.initial_geom_size
        self.data.mocap_pos[:] = [0,0,-2]
        self.data.qpos[:6],self.data.ctrl[:6] = self.arm_home,self.arm_home
        self.data.qvel[:] = 0
        self.colours = [RED]*21+[YELLOW]*21
        self.supply = self.colours.copy()
        for i,body in enumerate(self.discs):
            self.put(body,self.point("feeder",i%6*self.config["feeder_pitch"],i//6*self.config["feeder_pitch"]))
            self.model.body_gravcomp[body] = 1
        for r,c in zip(*np.nonzero(cells)):
            colour = int(cells[r,c])
            slot = next((i for i,value in enumerate(self.supply) if value==colour),None)
            if slot is None:
                raise ValueError("A test position asks for more than 21 discs of one colour")
            self.supply[slot] = EMPTY
            self.put(self.discs[slot],self.point("board",c*self.config["pitch_x"],r*self.config["pitch_y"]))
            # Test arrangements can deliberately include floating discs. Their
            # geometry stays fixed until an animated physics test is requested.
            self.model.body_gravcomp[self.discs[slot]] = 1
        mujoco.mj_forward(self.model,self.data)
        self.detector.reset_stability()

    def frame(self):
        mujoco.mj_forward(self.model,self.data)
        captured_at = time.time()  # Exposure precedes rendering, not its completion.
        self.renderer.update_scene(self.data,camera="front_camera")
        return self.renderer.render().copy(),captured_at

    def snapshot(self,effect=None,robot_executing=False,frames=12):
        for _ in range(frames):
            rgb,stamp = self.frame()
            if effect:
                rgb = effect(rgb)
            obs = self.detector.process(rgb,stamp,robot_executing=robot_executing)
            self.last_rgb,self.last_observation = rgb,obs
            if obs.status=="READY":
                break
            time.sleep(.035)
        return obs

    def release(self,slot,column):
        """Scripted release above the TOP entry, not at a cell centre."""
        if not 0<=column<7 or not 0<=slot<42 or not self.supply[slot]:
            raise ValueError("Invalid column or unavailable source disc")
        self.supply[slot] = EMPTY
        for i,existing in enumerate(self.discs):
            if not self.supply[i]:
                self.model.body_gravcomp[existing] = 0
        body = self.discs[slot]
        self.put(body,self.point("board",column*self.config["pitch_x"],6*self.config["pitch_y"]+.025))
        self.model.body_gravcomp[body] = 0
        mujoco.mj_forward(self.model,self.data)

    def step(self,seconds=1.0,on_frame=None):
        """500 Hz physics, optional 30 Hz display; observation sees falling discs."""
        end = self.data.time+seconds
        while self.data.time<end:
            start = time.perf_counter()
            mujoco.mj_step(self.model,self.data,nstep=16)
            if on_frame:
                on_frame()
                time.sleep(max(0,.032-(time.perf_counter()-start)))

    def truth(self):
        """Independent scoring only: measured body positions -> nearest grid cell."""
        board,feeder = [EMPTY]*42,[EMPTY]*42
        mujoco.mj_forward(self.model,self.data)
        for i,body in enumerate(self.discs):
            for group,cols,rows,px,py,output in (
                ("board",7,6,self.config["pitch_x"],self.config["pitch_y"],board),
                ("feeder",6,7,self.config["feeder_pitch"],self.config["feeder_pitch"],feeder)):
                local = np.linalg.inv(self.pose(group)) @ [*self.data.xpos[body],1]
                c,r = round(local[0]/px),round(local[1]/py)
                if 0<=c<cols and 0<=r<rows and abs(local[2]+.00886)<.008 and np.linalg.norm(local[:2]-[c*px,r*py])<.003:
                    output[r*cols+c] = self.colours[i]
        return board,feeder

    def hide_tags(self,ids):
        for marker in ids:
            self.model.geom_rgba[self.model.geom(f"tag_{marker}").id,3] = 0

    def block(self,row=2,column=3):
        position = self.point("board",column*self.config["pitch_x"],row*self.config["pitch_y"],.03)
        self.data.mocap_pos[0] = position

    def close(self):
        self.renderer.close()
