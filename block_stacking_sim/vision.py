import numpy as np
from scipy import ndimage


class TopCamera:
    """Knows where the top-down camera is (calibration) and converts pixels <-> table coordinates."""

    def __init__(self, model, name="topdown"):
        cid = model.camera(name).id
        self.pos = model.cam_pos[cid].copy()
        self.fovy = float(model.cam_fovy[cid])
        g = model.geom("bk0_geom").id
        self.bw, self.bd, self.bh = [float(2 * x) for x in model.geom_size[g]]
        self.sq_side = float(2 * model.geom_size[model.geom("white_square").id][0])

    def f(self, H):
        return (H / 2) / np.tan(np.deg2rad(self.fovy) / 2)

    def to_world(self, u, v, z, shape):
        H, W = shape[:2]
        depth = self.pos[2] - z
        x = self.pos[0] + (u - (W - 1) / 2) * depth / self.f(H)
        y = self.pos[1] - (v - (H - 1) / 2) * depth / self.f(H)
        return x, y

    def to_pixel(self, xy, z, shape):
        H, W = shape[:2]
        depth = self.pos[2] - z
        u = (xy[0] - self.pos[0]) * self.f(H) / depth + (W - 1) / 2
        v = (H - 1) / 2 - (xy[1] - self.pos[1]) * self.f(H) / depth
        return int(round(u)), int(round(v))


def rgb_to_hsv(img):
    x = img.astype(float) / 255
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    ok = d > 1e-6
    rc = (mx == r) & ok
    gc = (mx == g) & ok & ~rc
    bc = (mx == b) & ok & ~rc & ~gc
    h[rc] = ((g - b)[rc] / d[rc]) % 6
    h[gc] = (b - r)[gc] / d[gc] + 2
    h[bc] = (r - g)[bc] / d[bc] + 4
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0.0)
    return h * 60, s, mx


def colour_name(hue):
    if hue < 20 or hue > 340:
        return "red"
    if 40 <= hue <= 75:
        return "yellow"
    if 90 <= hue <= 165:
        return "green"
    if 190 <= hue <= 265:
        return "blue"
    return "other"


def block_yaw(px, py):
    """Rotation of the block's square top (radians, between -45 and +45 degrees)."""
    P = np.stack([px - px.mean(), py - py.mean()])
    best, ang = 1e9, 0.0
    for a in np.deg2rad(np.arange(0, 90, 2)):
        R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        q = R @ P
        area = np.ptp(q[0]) * np.ptp(q[1])
        if area < best:
            best, ang = area, a
    return ang - np.pi / 2 if ang > np.pi / 4 else ang


def _components(mask, min_px):
    lab, n = ndimage.label(mask)
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        if len(xs) >= min_px:
            yield xs, ys


def detect(img, cam):
    """img: RGB image from the top-down camera. Returns the blocks and the white square it can see."""
    hue, sat, val = rgb_to_hsv(img)
    H = img.shape[0]
    out = {"blocks": [], "square": None}

    z_top = cam.bh * 0.75
    px_area = ((cam.pos[2] - cam.bh) / cam.f(H)) ** 2
    expected = cam.bw * cam.bd
    for xs, ys in _components((sat > 0.45) & (val > 0.25), 40):
        wx, wy = cam.to_world(xs.astype(float), ys.astype(float), z_top, img.shape)
        area = len(xs) * px_area
        if not (0.5 * expected < area < 2.2 * expected):
            continue
        out["blocks"].append({
            "xy": np.array([wx.mean(), wy.mean()]),
            "yaw": float(block_yaw(wx, wy)),
            "color": colour_name(float(np.median(hue[ys, xs]))),
            "area_cm2": area * 1e4,
        })

    # white square: must be the right size and squarish (arm parts are neither)
    white = (sat < 0.18) & (val > 0.80)
    sq_px_area = ((cam.pos[2] - 0.002) / cam.f(H)) ** 2
    best = None
    for xs, ys in _components(white, 150):
        wx, wy = cam.to_world(xs.astype(float), ys.astype(float), 0.002, img.shape)
        w, h = np.ptp(wx), np.ptp(wy)
        fill = len(xs) * sq_px_area / max(w * h, 1e-9)
        if 0.7 * cam.sq_side < max(w, h) < 1.3 * cam.sq_side and fill > 0.6:
            if best is None or fill > best[0]:
                best = (fill, wx, wy)
    if best is not None:
        _, wx, wy = best
        out["square"] = {"xy": np.array([(wx.min() + wx.max()) / 2, (wy.min() + wy.max()) / 2]),
                         "side": float(max(np.ptp(wx), np.ptp(wy)))}
    out["blocks"].sort(key=lambda b: b["xy"][1])
    return out