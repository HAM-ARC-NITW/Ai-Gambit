import numpy as np
from PIL import Image
from env import RowEnv
from vision import TopCamera, detect

env = RowEnv()
env.reset()
cam = TopCamera(env.model)
img = env.get_image("topdown")
scene = detect(img, cam)

truth = env.block_positions()
print(f"blocks the camera found: {len(scene['blocks'])} (the scene has {len(truth)})")
for b in scene["blocks"]:
    err = np.min(np.linalg.norm(truth[:, :2] - b["xy"], axis=1)) * 1000
    print(f"   {b['color']:6s} x={b['xy'][0]:.3f} y={b['xy'][1]:.3f}  yaw={np.degrees(b['yaw']):5.1f} deg  "
          f"area={b['area_cm2']:.1f} cm2   error vs true position: {err:.1f} mm")

sq = scene["square"]
true_sq = env.data.geom_xpos[env.sq][:2]
if sq is None:
    print("white square: NOT FOUND")
else:
    err = np.linalg.norm(sq["xy"] - true_sq) * 1000
    print(f"white square found at x={sq['xy'][0]:.3f} y={sq['xy'][1]:.3f}, side {sq['side'] * 100:.1f} cm, "
          f"error vs true position: {err:.1f} mm")

dbg = img.copy()
for b in scene["blocks"]:
    u, v = cam.to_pixel(b["xy"], cam.bh * 0.75, img.shape)
    dbg[max(v - 4, 0):v + 5, max(u - 4, 0):u + 5] = [255, 0, 255]
if sq is not None:
    u, v = cam.to_pixel(sq["xy"], 0.002, img.shape)
    dbg[max(v - 4, 0):v + 5, max(u - 4, 0):u + 5] = [0, 255, 255]
Image.fromarray(img).save("camera_view.png")
Image.fromarray(dbg).save("vision_debug.png")
print("saved camera_view.png and vision_debug.png (magenta dots = blocks found, cyan dot = square)")