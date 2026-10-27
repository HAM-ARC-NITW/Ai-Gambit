"""Connect 4: RGB perception followed by a separate, versioned game coordinator.

Read top-to-bottom: geometry -> tags -> colour -> stability -> legal verification.
The detector never imports MuJoCo or reads simulator positions. Rows run bottom
to top, columns left to right; flattened index = row * 7 + column.
"""
from dataclasses import dataclass,field
from collections import deque
from pathlib import Path
import json
import time
import uuid
import cv2
import numpy as np

EMPTY,RED,YELLOW,UNKNOWN = 0,1,2,255


def load_config(path=None):
    return json.loads(Path(path or Path(__file__).with_name("connect4.json")).read_text())


def transform(rotation,position):
    T = np.eye(4)
    T[:3,:3],T[:3,3] = rotation,position
    return T


def calibration(config):
    """Ideal simulated calibration; real cameras must supply measured K, D and TF."""
    w,h = config["image_width"],config["image_height"]
    f = h/(2*np.tan(np.deg2rad(config["camera_fovy"])/2))
    return (np.array([[f,0,w/2],[0,f,h/2],[0,0,1.]]),np.zeros(5),
            transform([[0,0,-1],[1,0,0],[0,-1,0]],config["camera_position"]))


def drop(board,column,colour):
    """Rules helper: return a NEW board; gravity chooses the landing row."""
    board = np.asarray(board,dtype=np.uint8).reshape(6,7)
    if (isinstance(column,(bool,np.bool_)) or not isinstance(column,(int,np.integer))
            or colour not in (RED,YELLOW) or not 0<=column<7):
        raise ValueError("Invalid column or colour")
    empty = np.flatnonzero(board[:,column]==EMPTY)
    if not len(empty):
        raise ValueError("Column is full")
    row = int(empty[0])
    result = board.copy()
    result[row,column] = colour
    return result,row


def winners(board):
    board = np.asarray(board).reshape(6,7)
    found = set()
    for r in range(6):
        for c in range(7):
            for dr,dc in ((0,1),(1,0),(1,1),(1,-1)):
                if 0<=r+3*dr<6 and 0<=c+3*dc<7:
                    values = [int(board[r+k*dr,c+k*dc]) for k in range(4)]
                    if values[0] in (RED,YELLOW) and len(set(values))==1:
                        found.add(values[0])
    return found


def board_error(board):
    """Structural checks are separate from colour classification."""
    board = np.asarray(board).reshape(6,7)
    if not np.isin(board,[EMPTY,RED,YELLOW]).all():
        return "Unknown/invalid cells"
    for col in board.T:
        empty_seen = False
        for value in col:
            if value==EMPTY:
                empty_seen = True
            elif empty_seen:
                return "Floating disc: gravity is inconsistent"
    nr,ny = int(np.sum(board==RED)),int(np.sum(board==YELLOW))
    if nr not in (ny,ny+1) or nr>21 or ny>21:
        return "Impossible colour counts (Red starts)"
    win = winners(board)
    if len(win)>1 or (RED in win and nr!=ny+1) or (YELLOW in win and nr!=ny):
        return "Impossible winning position"
    return ""


@dataclass
class Observation:
    observation_id: int
    timestamp: float
    status: str = "INITIALIZING"
    reason: str = ""
    cells: list = field(default_factory=lambda:[UNKNOWN]*42)
    confidence: list = field(default_factory=lambda:[0.]*42)
    feeder: list = field(default_factory=lambda:[UNKNOWN]*42)
    feeder_confidence: list = field(default_factory=lambda:[0.]*42)
    poses: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)
    reprojection_error: float = 0.
    stable: bool = False
    images: dict = field(default_factory=dict)

    def as_dict(self):
        return {k:v for k,v in self.__dict__.items() if k!="images"}


class Perception:
    def __init__(self,config=None,K=None,D=None,base_camera=None):
        self.config = config or load_config()
        k,d,t = calibration(self.config)
        self.K,self.D,self.base_camera = (k if K is None else np.asarray(K,float),
            d if D is None else np.asarray(D,float),t if base_camera is None else np.asarray(base_camera,float))
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.tags = cv2.aruco.ArucoDetector(dictionary,params)
        self.reference,self.reference_poses = {},{}
        self.board_regions,self.board_allowed = [],None
        self.sequence,self.last_timestamp = 0,-float("inf")
        self.reset_stability()

    def reset_stability(self):
        self.previous,self.previous_images = None,{}
        self.stable_count,self.stable_since = 0,None

    def localize(self,rgb):
        corners,ids,_ = self.tags.detectMarkers(rgb)
        seen = {} if ids is None else {int(i):c.reshape(4,2) for i,c in zip(ids.flat,corners)}
        locations = {}
        half = self.config["marker_size"]/2
        delta = np.array([[-half,half],[half,half],[half,-half],[-half,-half]])
        for group in ("board","feeder"):
            points,pixels = [],[]
            for marker,center in self.config[group+"_tags"].items():
                if int(marker) in seen:
                    points.extend(np.asarray(center)+delta)
                    pixels.extend(seen[int(marker)])
            minimum = self.config["minimum_tags"]*4
            if len(points)<minimum:  # Three tags by default; one can be hidden.
                continue
            points,pixels = np.asarray(points,float),np.asarray(pixels,float)
            # Two tags on one narrow edge give poor extrapolation across the
            # entire board. Require useful spread in BOTH physical directions.
            if np.min(np.ptp(points,axis=0))<self.config["marker_size"]*2:
                continue
            H,inliers = cv2.findHomography(points,pixels,cv2.RANSAC,2.)
            if H is None or inliers.sum()<minimum or inliers.mean()<.75:
                continue
            keep = inliers.ravel().astype(bool)
            if np.min(np.ptp(points[keep],axis=0))<self.config["marker_size"]*2:
                continue
            object_points = np.c_[points[keep],np.zeros(keep.sum())]
            ok,rvec,tvec = cv2.solvePnP(object_points,pixels[keep],self.K,np.zeros(5))
            if not ok or tvec[2,0]<=0:
                continue
            predicted,_ = cv2.projectPoints(object_points,rvec,tvec,self.K,np.zeros(5))
            error = float(np.sqrt(np.mean((predicted.reshape(-1,2)-pixels[keep])**2)))
            if error<=self.config["max_reprojection_error"]:
                T = self.base_camera @ transform(cv2.Rodrigues(rvec)[0],tvec.ravel())
                locations[group] = (H,T,error)
        return locations,sorted(seen)

    def regions(self,rgb,group,H):
        """Rectify to 100 pixels/cell, with a margin to detect jams above the grid."""
        cols,rows = (7,6) if group=="board" else (6,7)
        px,py = (self.config["pitch_x"],self.config["pitch_y"]) if group=="board" else (self.config["feeder_pitch"],)*2
        # A one-cell border includes the top entry area, not just the 42 holes.
        scale = np.array([[100/px,0,150],[0,-100/py,(rows+.5)*100],[0,0,1.]])
        image = cv2.warpPerspective(rgb,scale @ np.linalg.inv(H),((cols+2)*100,(rows+2)*100))
        return image,cols,rows,px,py

    def set_empty_reference(self,rgb):
        """Explicitly photograph EMPTY board and feeder; never learn occupied cells."""
        image = cv2.undistort(rgb,self.K,self.D)
        locations,_ = self.localize(image)
        if len(locations)!=2:
            raise ValueError("Empty calibration needs board AND feeder tags")
        self.reference = {g:self.regions(image,g,H)[0] for g,(H,_,_) in locations.items()}
        self.reference_poses = {g:T.copy() for g,(_,T,_) in locations.items()}
        # The thick/tapered STL apertures look slightly off-centre from an oblique
        # camera. Calibrate the visible openings from the EMPTY photograph, not
        # from whatever coloured object happens to be present in the live frame.
        reference = self.reference["board"]
        hsv = cv2.cvtColor(reference,cv2.COLOR_RGB2HSV)
        neutral = ((hsv[:,:,1]<80)&(hsv[:,:,2]<180)).astype(np.uint8)
        self.board_regions = []
        self.board_allowed = np.zeros(neutral.shape,np.uint8)
        for r in range(6):
            for c in range(7):
                x,y = int((c+1.5)*100),int((6-r+.5)*100)
                patch = neutral[y-45:y+46,x-45:x+46]
                count,labels,stats,_ = cv2.connectedComponentsWithStats(patch)
                if count<2:
                    raise ValueError(f"Empty calibration cannot see hole {r},{c}")
                component = 1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
                hole = (labels==component).astype(np.uint8)
                distance = cv2.distanceTransform(hole,cv2.DIST_L2,5)
                _,radius,_,center = cv2.minMaxLoc(distance)
                if radius<10 or radius>40:
                    raise ValueError(f"Empty opening {r},{c} is obstructed or poorly calibrated")
                self.board_regions.append((x-45+center[0],y-45+center[1],max(8,round(radius*.72))))
                self.board_allowed[y-45:y+46,x-45:x+46] |= cv2.dilate(hole,np.ones((7,7),np.uint8))
        self.reset_stability()

    def classify(self,image,reference,group):
        """Colour/empty evidence must dominate each physical circular opening."""
        cols,rows = (7,6) if group=="board" else (6,7)
        px,py = (self.config["pitch_x"],self.config["pitch_y"]) if group=="board" else (self.config["feeder_pitch"],)*2
        radius = self.config["hole_radius"]*.70 if group=="board" else self.config["disc_radius"]*.68
        hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        hue,sat,value = cv2.split(hsv)
        vivid = (sat>=self.config["min_saturation"]) & (value>=self.config["min_value"])
        red = np.zeros(hue.shape,bool)
        for low,high in self.config["red_hue"]:
            red |= (hue>=low)&(hue<=high)
        red &= vivid
        low,high = self.config["yellow_hue"]
        yellow = (hue>=low)&(hue<=high)&vivid
        # Compare empty appearance after moderate global exposure normalization.
        valid = (reference>10)&(image>10)
        # Median of corresponding-pixel ratios stays anchored to the unchanged
        # frame/backing, even when the board goes from empty to completely full.
        gain = np.clip(np.median(reference[valid].astype(float)/image[valid]) if valid.any() else 1.,.65,1.6)
        normalized = np.clip(image.astype(float)*gain,0,255).astype(np.uint8)
        difference = np.max(cv2.absdiff(normalized,reference),axis=2)
        empty = (difference<self.config["empty_difference"]) & (sat<80) & (value<180)
        values,scores = [],[]
        allowed = np.zeros(image.shape[:2],np.uint8)
        for r in range(rows):
            for c in range(cols):
                x,y = (c+1.5)*100,(rows-r+.5)*100
                mask = np.zeros(image.shape[:2],np.uint8)
                if group=="board":
                    x,y,calibrated_radius = self.board_regions[r*cols+c]
                    cv2.circle(mask,(x,y),calibrated_radius,1,-1)
                else:
                    cv2.ellipse(mask,(int(x),int(y)),(round(radius/px*100),round(radius/py*100)),0,0,360,1,-1)
                keep = mask.astype(bool)
                evidence = [float(np.mean(a[keep])) for a in (empty,red,yellow)]
                winner = int(np.argmax(evidence));score = evidence[winner]
                competing = sorted(evidence)[-2]>.15
                values.append(winner if score>=self.config["min_confidence"] and not competing else UNKNOWN)
                scores.append(score if values[-1]!=UNKNOWN else min(score,.4))
                # Supply discs have visible 3D sides from an oblique view. Their
                # permitted footprint is wider than their central colour sample.
                outer = self.config["hole_radius"]*1.35 if group=="board" else self.config["disc_radius"]*1.55
                cv2.ellipse(allowed,(int((c+1.5)*100),int((rows-r+.5)*100)),
                            (round(outer/px*100),round(outer/py*100)),0,0,360,1,-1)
        if group=="board":
            # A disc is nearer the front face than the empty backing, so it can
            # fill more of a tapered aperture than the empty-reference silhouette.
            allowed |= self.board_allowed
        # A redundant tag can be covered without obscuring a cell. Tag visibility
        # is already checked by localization; do not call its changed pixels a
        # stray disc when two other reliable tags still locate the fixture.
        half = self.config["marker_size"]*.75
        for tx,ty in self.config[group+"_tags"].values():
            x,y = tx/px*100+150,(rows+.5)*100-ty/py*100
            cv2.rectangle(allowed,(round(x-half/px*100),round(y-half/py*100)),
                          (round(x+half/px*100),round(y+half/py*100)),1,-1)
        # Ignore the far border containing tags; monitor the board/top and feeder.
        changed = (difference>45).astype(np.uint8)
        changed[allowed.astype(bool)] = 0
        changed[:35],changed[-35:],changed[:,:35],changed[:,-35:] = 0,0,0,0
        changed = cv2.morphologyEx(changed,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        _,_,stats,_ = cv2.connectedComponentsWithStats(changed)
        stray = bool(any(s[cv2.CC_STAT_AREA]>self.config["stray_area"] for s in stats[1:]))
        return values,scores,stray

    def process(self,rgb,timestamp=None,now=None,robot_executing=False):
        timestamp = time.time() if timestamp is None else float(timestamp)
        now = time.time() if now is None else float(now)
        self.sequence += 1
        obs = Observation(self.sequence,timestamp)

        def fail(status,reason):
            self.reset_stability()
            obs.status,obs.reason = status,reason
            return obs

        if (not np.isfinite([timestamp,now]).all() or timestamp<=self.last_timestamp or
            now-timestamp>self.config["max_frame_age"] or timestamp>now+.05):
            return fail("CAMERA_FAILURE","Stale, duplicate, future or out-of-order exposure")
        self.last_timestamp = timestamp
        if rgb is None or rgb.shape!=(self.config["image_height"],self.config["image_width"],3) or rgb.dtype!=np.uint8:
            return fail("CAMERA_FAILURE","Missing image, wrong dimensions or encoding")
        if robot_executing:
            return fail("ROBOT_EXECUTING","Intermediate robot/hand activity cannot be accepted")
        gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        if cv2.Laplacian(gray,cv2.CV_64F).var()<self.config["min_sharpness"]:
            return fail("CAMERA_QUALITY","Image is too blurred or featureless")
        if not self.reference:
            return fail("CALIBRATION_INVALID","An independently empty reference is required")
        image = cv2.undistort(rgb,self.K,self.D)
        locations,obs.tags = self.localize(image)
        stray = False
        for group,(H,T,error) in locations.items():
            obs.poses[group] = T.tolist()
            obs.reprojection_error = max(obs.reprojection_error,error)
            reference_pose = self.reference_poses[group]
            angle = np.rad2deg(np.arccos(np.clip((np.trace(T[:3,:3]@reference_pose[:3,:3].T)-1)/2,-1,1)))
            if np.linalg.norm(T[:3,3]-reference_pose[:3,3])>self.config["max_pose_shift"] or angle>self.config["max_pose_angle"]:
                return fail("CALIBRATION_INVALID",group+" moved; explicit recalibration is required")
            rectified = self.regions(image,group,H)[0]
            obs.images[group] = rectified
            values,scores,unexpected = self.classify(rectified,self.reference[group],group)
            if group=="board":
                obs.cells,obs.confidence = values,scores
                if unexpected:
                    # A broad object may itself have a disc-like colour (skin
                    # can fall in a red HSV interval). Invalidate the fixture,
                    # rather than publishing those apparent colours as reliable.
                    obs.cells,obs.confidence = [UNKNOWN]*42,[0.]*42
            else:
                obs.feeder,obs.feeder_confidence = values,scores
                if unexpected:obs.feeder,obs.feeder_confidence = [UNKNOWN]*42,[0.]*42
            stray |= unexpected
        if len(locations)!=2:
            return fail("CALIBRATION_INVALID","Insufficient reliable, well-spread board or feeder tags")
        if UNKNOWN in obs.cells+obs.feeder or stray:
            return fail("OCCLUDED","Uncertain colour/empty appearance or obstruction in monitored region")
        # Do not interpret discs while falling, bouncing, or while exposure changes.
        changed = tuple(obs.cells+obs.feeder)
        motion = any(float(np.mean(cv2.absdiff(im,self.previous_images[g])))>self.config["motion_difference"]
                     for g,im in obs.images.items() if g in self.previous_images)
        if changed!=self.previous or motion:
            self.stable_count,self.stable_since = 1,timestamp
        else:
            self.stable_count += 1
        self.previous,self.previous_images = changed,{g:im.copy() for g,im in obs.images.items()}
        obs.stable = self.stable_count>=self.config["stable_frames"] and timestamp-self.stable_since>=self.config["settling_seconds"]
        obs.status = "READY" if obs.stable else "MOVING"
        obs.reason = "Stable board and feeder" if obs.stable else "Waiting for consecutive settled exposures"
        if obs.stable and board_error(obs.cells):
            return fail("INVALID_BOARD",board_error(obs.cells))
        return obs


def usable(obs,config,after=-float("inf"),sequence=-1):
    return (obs.status=="READY" and obs.stable and obs.observation_id>sequence and obs.timestamp>after
            and -.05<=time.time()-obs.timestamp<=config["max_frame_age"]
            and min(obs.confidence+obs.feeder_confidence)>=config["min_confidence"]
            and UNKNOWN not in obs.cells+obs.feeder and set(obs.poses)=={"board","feeder"})


class Gate:
    """Shared native/ROS contract. Vision never changes the official game state."""
    def __init__(self,config=None):
        self.config = config or load_config()
        self.baseline = None
        self.completed = set()

    def begin(self,obs,cells,feeder,version,operation):
        self.baseline = None  # A failed request revokes older authorization.
        if not operation or operation in self.completed or not usable(obs,self.config):
            raise ValueError("Fresh valid baseline and a new operation ID are required")
        if list(cells)!=obs.cells or list(feeder)!=obs.feeder or board_error(cells):
            raise ValueError("Baseline differs from the confirmed board or supply")
        self.baseline = (operation,version,obs.observation_id,obs.timestamp,list(cells),list(feeder))

    def verify(self,obs,cells,feeder,version,operation,after):
        saved,self.baseline = self.baseline,None
        if saved is None or saved[:2]!=(operation,version) or operation in self.completed:
            raise ValueError("Missing, replayed or wrong-version baseline")
        if after<saved[3] or not usable(obs,self.config,max(after,saved[3]),saved[2]):
            raise ValueError("Verification needs a fresh settled exposure after completion")
        if list(cells)!=obs.cells or list(feeder)!=obs.feeder:
            raise ValueError("Observed board/supply differs from the requested outcome")
        # Compare against the confirmed baseline, not the previous camera frame.
        # Repeated exposures of the same disc must never become extra moves.
        changed = [i for i,(old,new) in enumerate(zip(saved[4],obs.cells)) if old!=new]
        if len(changed)!=1:
            raise ValueError("Engine update requires exactly one changed board cell")
        index = changed[0]
        row,column = divmod(index,7)
        colour = obs.cells[index]
        expected,_ = drop(saved[4],column,colour)
        supply_changes = [i for i,(old,new) in enumerate(zip(saved[5],obs.feeder)) if old!=new]
        if (saved[4][index]!=EMPTY or expected.ravel().tolist()!=obs.cells or
                board_error(obs.cells) or winners(saved[4]) or len(supply_changes)!=1):
            raise ValueError("Engine update requires a legal single-disc drop and supply change")
        slot = supply_changes[0]
        if saved[5][slot]!=colour or obs.feeder[slot]!=EMPTY:
            raise ValueError("Supply change does not match the observed disc colour")
        self.completed.add(operation)
        # JSON-compatible engine contract: zero-based coordinates, bottom row=0.
        # Version is the resulting board version, not the baseline version.
        return dict(row=row,column=column,index=index,colour=colour,
                    operation_id=operation,game_version=version+1,
                    observation_id=obs.observation_id,timestamp=obs.timestamp)


class Coordinator:
    """Small rules adapter; only this class commits a verified turn."""
    def __init__(self,config=None,human_colour=RED):
        if human_colour not in (RED,YELLOW):
            raise ValueError("Human colour must be RED or YELLOW")
        self.config = config or load_config()
        self.human_colour = human_colour
        self.gate = Gate(self.config)
        self.cells = [EMPTY]*42
        self.feeder = [RED]*21+[YELLOW]*21
        self.version,self.pending,self.paused = 0,None,False
        self.moves = deque()

    def take_move(self):
        """Engine polls this after finish(): one verified coordinate, or None.

        Only consume this in the engine adapter, once. Invalid/noisy frames and
        resets produce no moves. Keep operation_id for deduplication in transport.
        """
        return self.moves.popleft() if self.moves else None

    @property
    def turn(self):
        return RED if self.cells.count(RED)==self.cells.count(YELLOW) else YELLOW

    @property
    def terminal(self):
        return bool(winners(self.cells)) or EMPTY not in self.cells

    def begin(self,obs,actor,intended=None):
        if self.pending or self.paused or self.terminal:
            raise ValueError("Busy, paused or terminal game; no new turn allowed")
        colour = self.human_colour if actor=="human" else 3-self.human_colour
        if actor not in ("human","robot") or colour!=self.turn:
            raise ValueError("Wrong player turn")
        if actor=="robot" and intended is None:
            raise ValueError("Robot verification requires its commanded column")
        if intended is not None:
            drop(self.cells,intended,colour)  # Refuse full/invalid columns before release.
        slot = next((i for i,c in enumerate(self.feeder) if c==colour),None)
        if slot is None:
            raise ValueError("Disc supply is empty")
        operation = str(uuid.uuid4())
        self.gate.begin(obs,self.cells,self.feeder,self.version,operation)
        self.pending = dict(operation=operation,colour=colour,slot=slot,intended=intended)
        return self.pending.copy()

    def finish(self,obs,completed_at):
        if self.pending is None:
            raise ValueError("No operation is pending")
        request = self.pending
        try:
            matches = []
            for column in range(7):
                try:
                    expected,row = drop(self.cells,column,request["colour"])
                except ValueError:
                    continue
                if expected.ravel().tolist()==obs.cells:
                    matches.append((column,row,expected.ravel().tolist()))
            if len(matches)!=1:
                raise ValueError("No unique legal single-disc outcome; existing discs may have changed")
            column,row,expected = matches[0]
            if request["intended"] is not None and column!=request["intended"]:
                raise ValueError(f"Wrong column: observed {column}, requested {request['intended']}")
            feeder = self.feeder.copy()
            feeder[request["slot"]] = EMPTY
            event = self.gate.verify(obs,expected,feeder,self.version,request["operation"],completed_at)
            self.cells,self.feeder = expected,feeder
            self.version += 1
            self.moves.append(event)
            return column,row
        except ValueError:
            self.paused = True
            self.gate.baseline = None
            raise
        finally:
            self.pending = None

    def cancel(self):
        self.pending,self.gate.baseline,self.paused = None,None,True

    def reset(self,obs,completed_at):
        # Reset is explicit, after the release mechanism has finished. Empty cells
        # prove an empty grid, not that a real slider is locked or stock replenished.
        if not usable(obs,self.config,completed_at) or any(obs.cells):
            raise ValueError("Reset failed: need a fresh, visibly empty board")
        if any(c not in (EMPTY,RED,YELLOW) for c in obs.feeder):
            raise ValueError("Reset failed: feeder is not observed")
        self.cells,self.feeder = obs.cells.copy(),obs.feeder.copy()
        self.version,self.pending,self.paused = 0,None,False
        self.gate.baseline = None  # Preserve consumed operation IDs across resets.
        self.moves.clear()  # Old-game coordinates must not reach a new engine game.


def overlay(obs):
    """One beginner-readable debug view: rows, columns, colour and confidence."""
    panels = []
    colours = {EMPTY:(150,150,150),RED:(255,65,65),YELLOW:(255,220,30),UNKNOWN:(255,140,30)}
    for group,values,scores in (("board",obs.cells,obs.confidence),("feeder",obs.feeder,obs.feeder_confidence)):
        cols,rows = (7,6) if group=="board" else (6,7)
        image = obs.images.get(group,np.zeros(((rows+2)*100,(cols+2)*100,3),np.uint8)).copy()
        for index,(value,score) in enumerate(zip(values,scores)):
            r,c = divmod(index,cols)
            x,y = int((c+1.5)*100),int((rows-r+.5)*100)
            cv2.circle(image,(x,y),34,colours[value],2)
            label = f"{r},{c} {'.RY?'[(EMPTY,RED,YELLOW,UNKNOWN).index(value)]} {score:.2f}"
            cv2.putText(image,label,(x-45,y-39),cv2.FONT_HERSHEY_SIMPLEX,.33,colours[value],1)
        cv2.putText(image,group+": "+obs.status,(12,26),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
        panels.append(cv2.resize(image,(500,500)))
    return np.hstack(panels)
