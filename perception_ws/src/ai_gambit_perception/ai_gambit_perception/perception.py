"""RGB-only perception. No MuJoCo state or chess rules are used in this file.

Read from top to bottom: configuration -> localization -> occupancy -> stability
-> versioned verification. All arrays use a1, b1, ... h1, a2, ... h8 order.
An empty-board photograph is required; never learn it from an occupied board.
"""
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import cv2
import numpy as np

EMPTY, OCCUPIED, UNKNOWN = 0, 1, 255
SQUARES = [f"{f}{r}" for r in range(1, 9) for f in "abcdefgh"]


def load_config(path=None):
    return json.loads(Path(path or Path(__file__).with_name("workspace.json")).read_text())


def transform(rotation, position):
    """4x4 matrix: multiply it by [x, y, z, 1] to change coordinate frames."""
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, position
    return result


def camera_calibration(config):
    """Ideal pinhole calibration for the simulated, vertically mounted camera.

    Hardware must supply measured intrinsics, distortion and base-to-optical
    transform instead. These defaults are not a hardware calibration procedure.
    """
    w, h = config["image_width"], config["image_height"]
    focal = h / (2 * np.tan(np.deg2rad(config["camera_fovy_degrees"]) / 2))
    K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
    base_camera = np.asarray(config["base_camera_optical"],float) if "base_camera_optical" in config else transform(np.diag([1, -1, -1]), config["camera_position"])
    return K, np.zeros(5), base_camera


def tray_centers():
    """48 capture/retirement slots: 24 per colour in three rows of eight.
    Each colour can retire eight promoted pawns and lose up to 23 total physical
    pieces (original pieces plus replacement pieces, excluding its king).
    """
    return np.array([[0.145 + (i % 8) * .028, sign * (.165 + (i // 8) * .030), .01]
                     for sign in (-1, 1) for i in range(24)])


@dataclass
class Observation:
    observation_id: int
    timestamp: float
    status: str = "INITIALIZING"
    reason: str = ""
    occupancy: list = field(default_factory=lambda: [UNKNOWN] * 64)
    confidence: list = field(default_factory=lambda: [0.] * 64)
    offsets: list = field(default_factory=lambda: [[0., 0.] for _ in range(64)])
    rack: list = field(default_factory=lambda: [UNKNOWN] * 32)
    tray: list = field(default_factory=lambda: [UNKNOWN] * 48)
    rack_confidence: list = field(default_factory=lambda: [0.] * 32)
    tray_confidence: list = field(default_factory=lambda: [0.] * 48)
    poses: dict = field(default_factory=dict)
    marker_ids: list = field(default_factory=list)
    reprojection_error: float = 0.
    stable: bool = False
    board_image: object = None
    foreground: object = None

    def as_dict(self):
        """Images are saved separately; everything else can go into JSON or ROS."""
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.__dict__.items() if k not in ("board_image", "foreground")}


class Perception:
    def __init__(self, config=None, K=None, distortion=None, base_camera=None):
        self.config = config or load_config()
        default_K, default_D, default_T = camera_calibration(self.config)
        self.K = default_K if K is None else np.asarray(K, float)
        self.distortion = default_D if distortion is None else np.asarray(distortion, float)
        self.base_camera = default_T if base_camera is None else np.asarray(base_camera, float)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self.reference = None
        self.sequence = 0
        self.last_timestamp = -float("inf")
        self.previous = None
        self.stable_count = 0
        self.baseline = None
        self.state = "INITIALIZING"

    def _localize(self, image):
        """Estimate each fixture independently; a hidden rack cannot borrow the
        board pose. RANSAC rejects corners inconsistent with the other markers.
        """
        corners, ids, _ = self.detector.detectMarkers(image)
        seen = {} if ids is None else {int(i): c.reshape(4, 2) for i, c in zip(ids.flat, corners)}
        results = {}
        size = self.config["marker_size"] / 2
        # Texture top-left, top-right, bottom-right, bottom-left in local XY.
        delta = np.array([[-size, size], [size, size], [size, -size], [-size, -size]])
        for group in ("board", "rack", "tray"):
            world, pixels = [], []
            for marker, center in self.config[f"{group}_markers"].items():
                if int(marker) in seen:
                    world.extend(np.asarray(center) + delta)
                    pixels.extend(seen[int(marker)])
            if len(world) < 8:  # Require two independent tags, not a guessed pose.
                continue
            world, pixels = np.asarray(world, float), np.asarray(pixels, float)
            H, inliers = cv2.findHomography(world, pixels, cv2.RANSAC, 2.5)
            if H is None or inliers.sum() < 8 or inliers.mean() < .75:
                continue
            keep = inliers.ravel().astype(bool)
            points = np.column_stack([world[keep], np.zeros(keep.sum())])
            ok, rvec, tvec = cv2.solvePnP(points, pixels[keep], self.K, np.zeros(5))
            if not ok or tvec[2, 0] <= 0:
                continue
            predicted, _ = cv2.projectPoints(points, rvec, tvec, self.K, np.zeros(5))
            error = float(np.sqrt(np.mean((predicted.reshape(-1, 2) - pixels[keep]) ** 2)))
            if error > self.config["max_reprojection_error"]:
                continue
            pose = self.base_camera @ transform(cv2.Rodrigues(rvec)[0], tvec.ravel())
            results[group] = (H, pose, error)
        return results, sorted(seen)

    def _warp(self, image, H, origin, width, height, pixels_per_meter):
        # H maps local fixture metres to camera pixels; its inverse rectifies.
        scale = np.array([[pixels_per_meter, 0, -origin[0] * pixels_per_meter],
                          [0, -pixels_per_meter, (origin[1] + height) * pixels_per_meter],
                          [0, 0, 1]])
        return cv2.warpPerspective(image, scale @ np.linalg.inv(H),
                                   (round(width * pixels_per_meter), round(height * pixels_per_meter)))

    def _regions(self, image, locations):
        p = self.config["board_pixels"]
        ppm = p / .24
        board = self._warp(image, locations["board"][0], [0, 0], .24, .24, ppm)
        rack = self._warp(image, locations["rack"][0], [-.01, -.01], .32, .04, ppm)
        # The tray frame shares board axes and origin; its tags surround both sides.
        tray = self._warp(image, locations["tray"][0], [-.14, 0], .52, .24, ppm)
        apron = self._warp(image, locations["board"][0], [-.16, 0], .56, .33, 1000)
        return board, rack, tray, apron

    def set_empty_reference(self, rgb):
        """Explicit calibration step, normally a saved photograph of empty fixtures."""
        image = cv2.undistort(rgb, self.K, self.distortion)
        locations, _ = self._localize(image)
        if len(locations) != 3:
            raise ValueError("Empty-reference calibration needs visible board, rack and tray tags")
        self.reference = self._regions(image, locations)
        self.previous, self.stable_count = None, 0

    def _foreground(self, image, reference):
        # Median channel levels are dominated by empty fixture pixels. Correct a
        # moderate global exposure change before differencing (not local shadows).
        # Fit background pixels robustly: half a starting board contains pieces,
        # so a plain whole-image median would confuse pieces with an exposure change.
        source, target = image[::8,::8].reshape(-1,3).astype(float), reference[::8,::8].reshape(-1,3).astype(float)
        gain, bias = np.ones(3), np.zeros(3)
        for _ in range(3):
            residual = np.max(np.abs(source*gain+bias-target),axis=1)
            keep = residual <= np.percentile(residual,60)
            for c in range(3):
                x, y = source[keep,c], target[keep,c]
                if np.std(x) > 8 and np.std(y) > 8:
                    a, b = np.linalg.lstsq(np.column_stack([x,np.ones(len(x))]),y,rcond=None)[0]
                    gain[c], bias[c] = np.clip(a,.65,1.6), np.clip(b,-40,40)
                else:
                    bias[c] = np.clip(np.median(y-x),-40,40)
        normalized = np.clip(image.astype(float) * gain + bias, 0, 255).astype(np.uint8)
        difference = np.max(cv2.absdiff(normalized, reference), axis=2)
        mask = (difference > self.config["difference_threshold"]).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        return mask, normalized

    @staticmethod
    def _split_necks(mask, max_area):
        """In perspective two tall neighbours (a knight on e2 before the king on
        e1) can touch through a thin neck and read as one oversized blob. Cut the
        neck only when that leaves two or more piece-sized parts and removes
        little else; a hand or arm loses most of its area and stays oversized.
        """
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
        out = mask.copy()
        for i in range(1, count):
            area = stats[i, cv2.CC_STAT_AREA]
            if not max_area < area <= 2*max_area:
                continue
            blob = (labels == i).astype(np.uint8)
            opened = cv2.morphologyEx(blob, cv2.MORPH_OPEN, kernel)
            n, _, parts, _ = cv2.connectedComponentsWithStats(opened, 8)
            sizes = parts[1:, cv2.CC_STAT_AREA]
            if n > 2 and np.all(sizes > 280) and sizes.max() <= max_area and sizes.sum() >= .85*area:
                out[blob > 0] = 0
                out[opened > 0] = 1
        return out

    def _occupancy(self, mask, centers, radius, max_area, linked_slots=False):
        """A piece must have a compact foreground component near its slot centre.
        Large components (hand/arm), boundary pieces and uncertain shapes stay UNKNOWN.
        This is a controlled-workspace heuristic, not a general object classifier.
        """
        if not linked_slots:
            mask = self._split_necks(mask, max_area)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        values, confidence, offsets = [], [], []
        grid = np.asarray(centers, float)
        for x, y in centers:
            x, y = int(x), int(y)
            patch = labels[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1]
            candidates = []
            for i in np.unique(patch):
                if not i or stats[i, cv2.CC_STAT_AREA] <= 280:
                    continue
                # In perspective a tall piece can overhang the next square (a
                # knight's head beside a back-rank rook). A sliver of a component
                # centred on another square belongs to that square. Larger
                # intrusions still make this square UNKNOWN.
                distances = np.hypot(*(grid - centroids[i]).T)
                if (not linked_slots and np.hypot(*(centroids[i] - (x, y))) > radius
                        and distances.min() < radius and np.count_nonzero(patch == i) < .15*patch.size):
                    continue
                candidates.append(int(i))
            area = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in candidates)
            offset = [0., 0.]
            if not candidates:
                value, score = EMPTY, .97
            elif len(candidates) != 1 or area > max_area:
                value, score = UNKNOWN, .1
            else:
                cx, cy = centroids[candidates[0]]
                if linked_slots:
                    # In the compact rack, shadows can join adjacent rows. Measure
                    # the visible part inside this slot, while still rejecting a
                    # component large enough to cover several slots.
                    yy, xx = np.nonzero(patch == candidates[0])
                    cx, cy = x-radius+xx.mean(), y-radius+yy.mean()
                offset = [(cx-x) * .24/self.config["board_pixels"],
                          -(cy-y) * .24/self.config["board_pixels"]]
                value, score = OCCUPIED, .95
                if area < 150 or np.linalg.norm(offset) > self.config["max_center_offset"]:
                    value, score = UNKNOWN, .25
            values.append(value)
            confidence.append(score)
            offsets.append(offset)
        # Components outside every allowed slot indicate a dropped piece or foreign object.
        stray = False
        for i in range(1, count):
            if stats[i, cv2.CC_STAT_AREA] < 100:
                continue
            if not any(np.linalg.norm(centroids[i] - c) < radius * 1.6 for c in centers):
                stray = True
        return values, confidence, offsets, stray

    def process(self, rgb, timestamp=None, now=None, robot_executing=False):
        timestamp = time.time() if timestamp is None else timestamp
        now = time.time() if now is None else now
        self.sequence += 1
        obs = Observation(self.sequence, timestamp)

        def fail(status, reason):
            obs.status, obs.reason = status, reason
            self.state, self.stable_count, self.previous = status, 0, None
            return obs

        if rgb is None or timestamp <= self.last_timestamp or now-timestamp > self.config["max_frame_age"] or timestamp > now+.1:
            return fail("CAMERA_FAILURE", "Missing, stale, future or out-of-order frame")
        self.last_timestamp = timestamp
        if rgb.shape != (self.config["image_height"], self.config["image_width"], 3):
            return fail("CAMERA_FAILURE", "Image dimensions disagree with camera calibration")
        image = cv2.undistort(rgb, self.K, self.distortion)
        locations, obs.marker_ids = self._localize(image)
        if len(locations) != 3:
            return fail("CALIBRATION_INVALID", "Need at least two valid tags per fixture")
        obs.poses = {k: v[1].tolist() for k, v in locations.items()}
        obs.reprojection_error = max(v[2] for v in locations.values())
        # Fixtures are mechanically fixed in V1. Do not silently follow a shifted board.
        for name, origin in (("board", "board_origin"), ("rack", "rack_origin"), ("tray", "board_origin")):
            rotation = self.config["rack_rotation" if name == "rack" else "board_rotation"]
            pose = locations[name][1]
            angle = np.rad2deg(np.arccos(np.clip((np.trace(np.asarray(rotation).T @ pose[:3,:3])-1)/2, -1, 1)))
            if np.linalg.norm(pose[:3,3] - self.config[origin]) > self.config["max_pose_shift"] or angle > self.config["max_pose_angle_degrees"]:
                return fail("CALIBRATION_INVALID", f"{name} moved; validate fixtures and recalibrate")
        if self.reference is None:
            return fail("INITIALIZING", "An empty-workspace reference photograph is required")
        board, rack, tray, apron = self._regions(image, locations)
        masks, normalized = zip(*(self._foreground(im, ref) for im, ref in zip((board, rack, tray, apron), self.reference)))
        # The wide tray crop also contains the board. Observe only its side strips.
        masks[2][:, round(.14*self.config['board_pixels']/.24):round(.38*self.config['board_pixels']/.24)] = 0
        # Remove thin checker-edge artifacts without cutting real pieces at
        # square boundaries. Cutting those boundaries would hide a fallen or
        # badly centred piece's true size and centroid.
        masks[0][:] = cv2.morphologyEx(masks[0],cv2.MORPH_OPEN,np.ones((9,9),np.uint8))
        p, cell = self.config["board_pixels"], self.config["board_pixels"] // 8
        centers = [(cell*(f+.5), cell*(7-r+.5)) for r in range(8) for f in range(8)]
        obs.occupancy, obs.confidence, obs.offsets, stray_board = self._occupancy(masks[0], centers, 37, 6200)
        ppm = p/.24
        rack_centers = [((f*.02+.01)*ppm, (.03-r*.02)*ppm) for r in range(2) for f in range(16)]
        obs.rack, obs.rack_confidence, _, stray_rack = self._occupancy(masks[1], rack_centers, 24, 6000, linked_slots=True)
        T = transform(self.config["board_rotation"], self.config["board_origin"])
        local = (np.linalg.inv(T) @ np.column_stack([tray_centers(), np.ones(48)]).T).T
        tray_pixels = [((v[0]+.14)*ppm, (.24-v[1])*ppm) for v in local]
        # Tall captured pieces can join adjacent silhouettes through shadows.
        # As in the compact rack, score each slot locally but reject components
        # large enough to cover several slots as an obstruction.
        obs.tray, obs.tray_confidence, _, stray_tray = self._occupancy(masks[2], tray_pixels, 33, 20000, linked_slots=True)
        # Also inspect the gaps around the fixtures. Remove regions already
        # handled above; an extra object in the remaining apron blocks new moves.
        apron_mask = masks[3].copy()
        for x0,y0,x1,y1 in ((-.005,0,.245,.245),(-.045,.260,.285,.320),
                             (-.14,0,-.02,.245),(.26,0,.38,.245)):
            left,right = max(0,round((x0+.16)*1000)),round((x1+.16)*1000)
            top,bottom = max(0,round((.33-y1)*1000)),round((.33-y0)*1000)
            apron_mask[top:bottom,left:right] = 0
        _,_,apron_stats,_ = cv2.connectedComponentsWithStats(apron_mask,8)
        stray_apron = any(area > 300 for area in apron_stats[1:,cv2.CC_STAT_AREA])
        obs.board_image, obs.foreground = board, masks[0]
        if robot_executing:
            return fail("ROBOT_EXECUTING", "Intermediate robot positions cannot be committed")
        if UNKNOWN in obs.occupancy + obs.rack + obs.tray or stray_board or stray_rack or stray_tray or stray_apron:
            parts = [name for name, bad in (("board",UNKNOWN in obs.occupancy or stray_board),
                                           ("rack",UNKNOWN in obs.rack or stray_rack),("tray",UNKNOWN in obs.tray or stray_tray),
                                           ("workspace apron",stray_apron)) if bad]
            return fail("OCCLUDED", "Unknown object or invalid placement in " + ", ".join(parts))
        # Compare all fixtures, not just occupancy bits: a moving hand can leave
        # occupancy unchanged for several frames. Downsampling suppresses sensor noise.
        small = np.concatenate([cv2.resize(im, (80, 80)).reshape(-1).astype(float) for im in normalized])
        difference = float("inf") if self.previous is None else np.mean(np.abs(small-self.previous))
        self.previous = small
        self.stable_count = self.stable_count+1 if difference < 3.0 else 1
        obs.stable = self.stable_count >= self.config["stable_frames"]
        obs.status = "READY" if obs.stable else "MOVING"
        obs.reason = f"{self.stable_count}/{self.config['stable_frames']} consistent frames"
        self.state = "OBSERVATION_READY" if obs.stable else "WAITING_FOR_STABILITY"
        return obs

    def establish_baseline(self, obs, expected, version, rack=None, tray=None, now=None):
        ok, reason = self.verify(obs, expected, rack=rack, tray=tray, now=now)
        if ok:
            self.baseline = (obs.observation_id, version, obs.timestamp, list(obs.occupancy))
            self.state = "WAITING_FOR_HUMAN"
        else:
            self.baseline = None
            self.state = "ERROR"
        return ok, reason

    def verify(self, obs, expected, version=None, after=None, rack=None, tray=None, now=None):
        if obs.status != "READY" or not obs.stable:
            return False, f"Observation not ready: {obs.status}: {obs.reason}"
        if len(obs.occupancy) != 64 or len(obs.confidence) != 64 or len(expected) != 64:
            return False, "Board arrays must contain exactly 64 entries"
        if UNKNOWN in obs.occupancy or min(obs.confidence) < .7:
            return False, "Unknown or low-confidence board square"
        if (time.time() if now is None else now)-obs.timestamp > self.config["max_frame_age"]:
            return False, "Observation expired"
        if after is not None and obs.timestamp <= after:
            return False, "Observation predates operation"
        if version is not None and (self.baseline is None or self.baseline[1] != version or obs.observation_id <= self.baseline[0]):
            return False, "Stale baseline or game-state version"
        wrong = [SQUARES[i] for i, (a, b) in enumerate(zip(obs.occupancy, expected)) if a != b]
        if len(expected) != 64 or wrong:
            return False, f"Board mismatch: {', '.join(wrong)}"
        for name, actual, wanted in (("rack", obs.rack, rack), ("tray", obs.tray, tray)):
            if wanted is not None and list(actual) != list(wanted):
                return False, f"{name} inventory mismatch"
        return True, "Fresh board and requested inventories match"


def overlay(obs):
    """Green=occupied, grey=empty, orange=unknown. Labels are observations only."""
    image = np.zeros((800, 800, 3), np.uint8) if obs.board_image is None else obs.board_image.copy()
    cell = image.shape[0] // 8
    for i, name in enumerate(SQUARES):
        f, r = i%8, i//8
        x, y = f*cell, (7-r)*cell
        color = {EMPTY:(150,150,150), OCCUPIED:(40,230,80), UNKNOWN:(255,150,20)}[obs.occupancy[i]]
        cv2.rectangle(image, (x+2,y+2), (x+cell-2,y+cell-2), color, 2)
        cv2.putText(image, f"{name} {obs.confidence[i]:.2f}", (x+4,y+18), cv2.FONT_HERSHEY_SIMPLEX, .37, color, 1)
    return image
