"""Generate the block scene using the robot/meshes already in this repository."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROBOT = HERE.parent/"simulation/so101_new_calib.xml"
OUT = HERE/"generated/row_scene.xml"
# Editable scene settings, in metres. The original target is retained.
N_BLOCKS,ROW_X,GAP = 4,.19,.04
SQUARE_SIDE,SQUARE_X,SQUARE_Y = .07,.09,.17


def build_scene():
    root = ET.parse(ROBOT).getroot()
    root.set("model","block_stacking")
    root.find("compiler").set("meshdir",str(ROBOT.parent/"assets"))
    ET.SubElement(root,"option",timestep=".002",integrator="implicitfast")
    visual = ET.SubElement(root,"visual")
    ET.SubElement(visual,"headlight",ambient=".25 .25 .25",diffuse=".3 .3 .3",specular="0 0 0")
    ET.SubElement(visual,"global",offwidth="640",offheight="480")
    ET.SubElement(visual,"quality",offsamples="0",shadowsize="256")
    # The ROS colour detector assumes an unsaturated arm. Change appearance only;
    # the shared robot file, mesh geometry and dynamics are left untouched.
    for material in root.findall("asset/material"):
        rgba = list(map(float,material.get("rgba","1 1 1 1").split()))
        if max(rgba[:3])-min(rgba[:3])>.2:material.set("rgba",".65 .65 .65 1")
    world = root.find("worldbody")
    ET.SubElement(world,"light",name="top_light",pos=".19 0 1.2",dir="0 0 -1",diffuse=".5 .5 .5",specular="0 0 0")
    ET.SubElement(world,"geom",name="table_floor",type="plane",size="1 1 .05",rgba=".30 .33 .38 1")
    ET.SubElement(world,"geom",name="white_square",type="box",size=f"{SQUARE_SIDE/2} {SQUARE_SIDE/2} .001",
                  pos=f"{SQUARE_X} {SQUARE_Y} .001",rgba="1 1 1 1")
    ET.SubElement(world,"camera",name="topdown",pos=f"{ROW_X} 0 .65",quat="1 0 0 0",fovy="45")
    ET.SubElement(world,"camera",name="overview",pos=".6 -.5 .45",xyaxes=".781 .625 0 -.349 .436 .830")
    bw,bd,bh = json.loads((HERE/"block_dims.json").read_text())["size"]
    colours = (".9 .15 .15 1",".15 .75 .25 1",".2 .35 .9 1",".95 .8 .1 1")
    for i in range(N_BLOCKS):
        y = (i-(N_BLOCKS-1)/2)*(bd+GAP)
        body = ET.SubElement(world,"body",name=f"bk{i}",pos=f"{ROW_X} {y} {bh/2+.0005}")
        ET.SubElement(body,"freejoint",name=f"bk{i}_j")
        ET.SubElement(body,"geom",name=f"bk{i}_geom",type="box",size=f"{bw/2} {bd/2} {bh/2}",
                      mass=".02",rgba=colours[i%4],friction="1.5 .02 .002",condim="4")
    OUT.parent.mkdir(parents=True,exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(OUT,encoding="utf-8",xml_declaration=True)
    return OUT


if __name__=="__main__":print("Built:",build_scene())
