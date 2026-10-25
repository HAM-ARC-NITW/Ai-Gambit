import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SO101 = os.path.join(HERE, "SO-ARM100", "Simulation", "SO101")
ROBOT_XML = "so101_new_calib.xml"
OUT = os.path.join(SO101, "row_scene.xml")

# ---------- settings you can change ----------
N_BLOCKS = 4
ROW_X = 0.22          # how far in front of the world origin the row sits (m)
GAP = 0.04            # empty space between neighbouring blocks (m)
SQUARE_SIDE = 0.07    # white square side length (m)
SQUARE_MARGIN = 0.09  # distance from the last block to the square's centre (m)
# ---------------------------------------------

dims_file = os.path.join(HERE, "block_dims.json")
if os.path.exists(dims_file):
    BW, BD, BH = json.load(open(dims_file))["size"]
else:
    BW = BD = BH = 0.025
print(f"block size (x, y, z): {BW*1000:.1f} x {BD*1000:.1f} x {BH*1000:.1f} mm")

pitch = BD + GAP                                   # centre-to-centre distance along the row
ys = [(i - (N_BLOCKS - 1) / 2) * pitch for i in range(N_BLOCKS)]
sq_x, sq_y = ROW_X, ys[-1] + SQUARE_MARGIN
colours = ["0.9 0.15 0.15 1", "0.15 0.75 0.25 1", "0.2 0.35 0.9 1", "0.95 0.8 0.1 1"]

bodies = ""
for i, y in enumerate(ys):
    bodies += f"""
    <body name="bk{i}" pos="{ROW_X} {y:.4f} {BH/2 + 0.0005:.4f}">
      <freejoint name="bk{i}_j"/>
      <geom name="bk{i}_geom" type="box" size="{BW/2} {BD/2} {BH/2}" mass="0.02"
            rgba="{colours[i % 4]}" friction="1.5 0.02 0.002" condim="4"/>
    </body>"""

xml = f"""<mujoco model="row_scene">
  <include file="{ROBOT_XML}"/>
  <worldbody>
    <light name="top_light" pos="{ROW_X} 0 1.2" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="table_floor" type="plane" size="1 1 0.05" rgba="0.30 0.33 0.38 1"
          friction="1 0.005 0.0001"/>
    <geom name="white_square" type="box" size="{SQUARE_SIDE/2} {SQUARE_SIDE/2} 0.001"
          pos="{sq_x} {sq_y:.4f} 0.001" rgba="1 1 1 1"/>
    <camera name="topdown" pos="{ROW_X} 0 0.65" quat="1 0 0 0" fovy="45"/>
    <camera name="overview" pos="0.6 -0.5 0.45"
            xyaxes="0.781 0.625 0 -0.349 0.436 0.830"/>{bodies}
  </worldbody>
</mujoco>
"""
open(OUT, "w").write(xml)
print("wrote", OUT)
print("block centres y:", [round(y, 3) for y in ys])
print(f"white square centre: x={sq_x:.3f} y={sq_y:.3f}")