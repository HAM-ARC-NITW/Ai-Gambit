import json, os
import numpy as np
import mujoco
from scipy import ndimage
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
BW, BD, BH = json.load(open(os.path.join(HERE, "block_dims.json")))["size"]
GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)


class DepthPerception:
    def __init__(self, model, cam="topdown", h=480, w=640):
        self.m, self.cam = model, cam
        self.cid = model.camera(cam).id
        self.r = mujoco.Renderer(model, h, w)
        self.f = (h / 2) / np.tan(np.deg2rad(model.cam_fovy[self.cid]) / 2)
        # geoms that belong to the arm (anything not world and not a bk block)
        self.is_arm = np.array([
            model.geom_bodyid[g] != 0 and not model.body(int(model.geom_bodyid[g])).name.startswith("bk")
            for g in range(model.ngeom)])

    def aim(self, xy, height=0.75):
        """Point the camera straight down over xy."""
        self.m.cam_pos[self.cid] = [xy[0], xy[1], height]
        self.m.cam_quat[self.cid] = [1.0, 0.0, 0.0, 0.0]

    @staticmethod
    def _yaw(px, py):
        best, ang = 1e9, 0.0
        P = np.stack([px, py])
        for a in np.deg2rad(np.arange(0, 90, 2)):
            R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
            q = R @ P
            s = np.ptp(q[0]) * np.ptp(q[1])
            if s < best:
                best, ang = s, a
        return ang - np.pi / 2 if ang > np.pi / 4 else ang

    def detect(self, data, table_z=0.0, save=None):
        mujoco.mj_kinematics(self.m, data)
        self.r.enable_depth_rendering()
        self.r.update_scene(data, camera=self.cam)
        depth = self.r.render().copy()
        self.r.disable_depth_rendering()
        self.r.enable_segmentation_rendering()
        self.r.update_scene(data, camera=self.cam)
        seg = self.r.render().copy()
        self.r.disable_segmentation_rendering()

        # pixels showing the arm are ignored (treated as empty floor)
        gid = seg[..., 0]
        ok = (gid >= 0) & (gid < self.m.ngeom) & (seg[..., 1] == GEOM)
        arm_px = np.zeros(depth.shape, bool)
        arm_px[ok] = self.is_arm[gid[ok]]
        arm_px = ndimage.binary_dilation(arm_px, iterations=2)

        cx, cy, cz = self.m.cam_pos[self.cid]
        H, W = depth.shape
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        X = cx + (uu - (W - 1) / 2) * depth / self.f
        Y = cy - (vv - (H - 1) / 2) * depth / self.f
        height = cz - depth - table_z          # height above table
        height[arm_px] = 0.0
        pix_area = (depth / self.f) ** 2       # m^2 covered by each pixel

        lab, n = ndimage.label(height > 0.5 * BH)
        print(f"   perception: {int(arm_px.sum())} arm pixels masked, {n} bumps found")
        out = []
        for k in range(1, n + 1):
            comp = lab == k
            if comp.sum() < 30:
                continue
            htop = np.percentile(height[comp], 95)
            top = comp & (height > htop - 0.3 * BH)       # top face only
            area = pix_area[top].sum()
            if not (0.4 * BW * BD < area < 1.6 * BW * BD):  # reject junk
                print(f"   skipped a bump: top area {area*1e4:.1f} cm2 (block is {BW*BD*1e4:.1f} cm2)")
                continue
            px, py = X[top], Y[top]
            c = np.array([px.mean(), py.mean()])
            out.append(dict(xy=c, yaw=self._yaw(px - c[0], py - c[1]),
                            level=int(round(htop / BH)), top_z=htop + table_z,
                            pix=(uu[top].mean(), vv[top].mean())))
        if save:
            g = np.clip(height / (3 * BH), 0, 1)
            img = (np.stack([g] * 3, -1) * 255).astype(np.uint8)
            img[arm_px] = [0, 0, 120]
            for b in out:
                u, v = int(b["pix"][0]), int(b["pix"][1])
                img[max(v - 4, 0):v + 5, max(u - 4, 0):u + 5] = [255, 0, 0]
            Image.fromarray(img).save(save)
        return out