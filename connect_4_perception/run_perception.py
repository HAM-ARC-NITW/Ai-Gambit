"""Visual Connect 4 perception experiments and a small interactive test board.

Default: 3D MuJoCo + front RGB + annotated cells; Space/N advances a case.
--headless writes the complete HTML/JSON gallery. --play accepts columns 1..7.
Robot releases in this perception demo are scripted; the arm remains parked.
"""
import argparse
import html
import json
from pathlib import Path
import time
import cv2
import mujoco
import mujoco.viewer
import numpy as np
from perception_sim import Simulation,HERE
from ai_gambit_perception.connect4 import (Coordinator,Perception,overlay,drop,winners,
                                          EMPTY,RED,YELLOW,UNKNOWN)


# Names are the test plan. Each normal grid case is an actual gravity drop, so
# the mapping is checked at every one of the 42 landing cells, including edges.
CASES = [f"drop_r{r}_c{c}" for r in range(6) for c in range(7)]+[
    "human_yellow","human_free_column","robot_wrong_column","two_red_discs","two_colours",
    "wrong_colour","disc_removed","disc_recoloured","slider_opened","failed_pickup",
    "disc_lost","supply_mismatch","empty_supply","wrong_supply_colour","full_column",
    "hand_on_board","hand_on_supply","one_tag_hidden","two_tags_hidden","same_edge_tags_only","board_tags_lost",
    "feeder_tags_lost","bad_tag_geometry","board_shifted","camera_shifted","mirrored_camera",
    "blurred_camera","camera_disconnected","stale_frame","duplicate_frame","out_of_order_frame",
    "future_frame","wrong_image_size","robot_executing","first_frame_unsettled","disc_falling",
    "jammed_at_top","off_centre_disc","orange_red","white_yellow","glare","moderate_exposure",
    "sensor_noise","low_light","floating_disc","impossible_counts","game_over","full_board_draw",
    "winning_drop","reset_empty","reset_leftover","reset_cached_frame","out_of_turn",
    "cancelled_drop","behind_backing","rgb_depth_limit"]


class Display:
    def __init__(self,sim,headless=False,seconds=None):
        self.sim,self.viewer,self.seconds = sim,None,seconds
        self.start = time.monotonic()
        if not headless:
            self.viewer = mujoco.viewer.launch_passive(sim.model,sim.data)
            self.viewer.cam.lookat[:] = [.24,-.08,.11]
            self.viewer.cam.distance,self.viewer.cam.azimuth,self.viewer.cam.elevation = .70,0,-15
            self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

    def show(self,obs=None,label=""):
        if self.seconds is not None and time.monotonic()-self.start>self.seconds:
            raise KeyboardInterrupt
        if self.viewer is None:
            return -1
        if not self.viewer.is_running():
            raise KeyboardInterrupt
        if obs is None:
            rgb,stamp = self.sim.frame()
            obs = self.sim.detector.process(rgb,stamp)
            self.sim.last_rgb = rgb
        rgb = self.sim.last_rgb.copy()
        cv2.putText(rgb,label[:125],(15,28),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),2)
        cv2.putText(rgb,obs.status+": "+obs.reason[:110],(15,54),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,220,80),1)
        cv2.imshow("Connect 4 - front camera",cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        cv2.imshow("Connect 4 - perception",cv2.cvtColor(overlay(obs),cv2.COLOR_RGB2BGR))
        self.viewer.sync()
        key = cv2.waitKey(1)&255
        if key==27:
            raise KeyboardInterrupt
        return key

    def wait(self,obs,label,automatic=False):
        if self.viewer is None:
            return
        deadline = time.monotonic()+2 if automatic else float("inf")
        while time.monotonic()<deadline:
            if self.show(obs,label+" | Space/N: next, Esc: exit") in (32,ord('n')):
                break
            time.sleep(.03)

    def close(self):
        if self.viewer:
            self.viewer.close()
        cv2.destroyAllWindows()


def set_game(sim,board,human=RED):
    sim.load(board)
    game = Coordinator(sim.config,human)
    game.cells,game.feeder = np.asarray(board).ravel().tolist(),sim.supply.copy()
    game.version = sum(c!=EMPTY for c in game.cells)
    return game


def paint(sim,slot,colour):
    geom = sim.model.geom(f"disc_visual_{slot}").id
    sim.model.geom_matid[geom] = -1
    sim.model.geom_rgba[geom] = colour


def run_case(sim,name,folder,display,automatic=False):
    board = np.zeros((6,7),np.uint8)
    human = RED
    expected,should_commit,limitation = {"READY"},False,""
    if name.startswith("drop_r"):
        row,column = int(name[6]),int(name[9])
        board[:row,column] = [RED if r%2==0 else YELLOW for r in range(row)]
        should_commit = True
    else:
        row,column = 0,3
    if name in ("human_yellow","robot_wrong_column"):
        board[0,3] = RED
        human = YELLOW if name=="human_yellow" else RED
    if name in ("disc_removed","disc_recoloured","slider_opened","reset_empty","reset_leftover","reset_cached_frame"):
        board[0,2:4] = [RED,YELLOW]
    if name=="full_column":
        board[:,3] = [RED,YELLOW]*3
    if name=="game_over":
        board[0,:4],board[1,:3] = RED,YELLOW
    if name=="winning_drop":
        board[0,:3],board[1,:3] = RED,YELLOW
        column,should_commit = 3,True
    if name=="full_board_draw":
        board = np.array([[1+(c//2+r)%2 for c in range(7)] for r in range(6)],np.uint8)
    game = set_game(sim,board,human)
    before = sim.snapshot()
    if before.status!="READY":
        raise AssertionError(name+": baseline not ready: "+before.status+" "+before.reason)
    initial = game.cells.copy()
    folder.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(folder/"before.png"),cv2.cvtColor(sim.last_rgb,cv2.COLOR_RGB2BGR))
    display.show(before,name+": BEFORE")
    error,accepted,actual = "",False,None
    effect,robot = None,False
    completed_at = time.time()
    actor = "human" if game.turn==game.human_colour else "robot"
    requested = None

    def begin(intended=column):
        nonlocal requested
        requested = game.begin(before,actor,None if actor=="human" else intended)
        return requested

    def release(slot,col):
        sim.release(slot,col)
        sim.step(1.0,lambda:display.show(label=name+": DISC FALLING (scripted release)")) if display.viewer else sim.step(1.)

    if name.startswith("drop_r") or name in ("human_yellow","human_free_column","robot_wrong_column","winning_drop"):
        begin()
        column = 1 if name=="human_free_column" else 2 if name=="robot_wrong_column" else column
        release(requested["slot"],column)
        should_commit = name!="robot_wrong_column"
    elif name in ("two_red_discs","two_colours","wrong_colour","disc_lost","supply_mismatch","failed_pickup","cancelled_drop"):
        begin()
        if name=="wrong_colour":
            release(21,column)
            expected = {"INVALID_BOARD"}
        elif name not in ("failed_pickup","disc_lost"):
            release(requested["slot"],column)
            if name in ("two_red_discs","two_colours"):
                release(1 if name=="two_red_discs" else 21,column)
                if name=="two_red_discs":expected = {"INVALID_BOARD"}
            if name=="supply_mismatch":
                sim.put(sim.discs[1],[3,3,.1]);sim.supply[1] = EMPTY
            if name=="cancelled_drop":game.cancel()
        if name=="disc_lost":
            sim.put(sim.discs[requested["slot"]],[3,3,.1]);sim.supply[requested["slot"]] = EMPTY
    elif name in ("disc_removed","disc_recoloured","slider_opened"):
        begin()
        used = [i for i,c in enumerate(sim.supply) if c==EMPTY]
        for slot in used if name=="slider_opened" else used[:1]:
            if name=="disc_recoloured":paint(sim,slot,[1,.85,.04,1]);sim.colours[slot] = YELLOW;expected = {"INVALID_BOARD"}
            else:sim.put(sim.discs[slot],[3,3,.1])
        if name=="disc_removed":expected = {"INVALID_BOARD"}
    elif name in ("empty_supply","wrong_supply_colour","full_column","game_over","full_board_draw","out_of_turn"):
        if name=="empty_supply":
            for slot in range(21):sim.put(sim.discs[slot],[3,3,.1]);sim.supply[slot] = EMPTY
            game.feeder = sim.supply.copy()
        if name=="wrong_supply_colour":paint(sim,0,[1,.85,.04,1]);sim.colours[0] = YELLOW
        after = sim.snapshot()
        try:game.begin(after,"robot" if name=="out_of_turn" else actor,column)
        except ValueError as exc:error = str(exc)
        else:raise AssertionError(name+": invalid operation was admitted")
    elif name in ("hand_on_board","hand_on_supply"):
        if name=="hand_on_board":sim.block()
        else:sim.data.mocap_pos[0] = sim.point("feeder",2*sim.config["feeder_pitch"],3*sim.config["feeder_pitch"],.03)
        expected = {"OCCLUDED"}
    elif name in ("one_tag_hidden","two_tags_hidden","same_edge_tags_only","board_tags_lost","feeder_tags_lost","bad_tag_geometry"):
        if name=="bad_tag_geometry":
            for marker in (20,21):sim.model.geom_pos[sim.model.geom(f"tag_{marker}").id,0] += .04
        else:
            ids = {"one_tag_hidden":[20],"two_tags_hidden":[20,22],"same_edge_tags_only":[20,21],"board_tags_lost":[20,21,22],"feeder_tags_lost":[24,25,26]}[name]
            sim.hide_tags(ids)
        if name in ("two_tags_hidden","same_edge_tags_only","board_tags_lost","feeder_tags_lost","bad_tag_geometry"):expected = {"CALIBRATION_INVALID"}
    elif name in ("board_shifted","camera_shifted"):
        if name=="board_shifted":sim.model.body_pos[sim.model.body("board_fixture").id,1] += .02
        else:sim.model.cam_pos[sim.model.camera("front_camera").id,1] += .03
        expected = {"CALIBRATION_INVALID"}
    elif name in ("mirrored_camera","blurred_camera","moderate_exposure","sensor_noise","low_light"):
        rng = np.random.default_rng(4)
        effect = {"mirrored_camera":lambda im:im[:,::-1].copy(),
            "blurred_camera":lambda im:cv2.GaussianBlur(im,(41,41),0),
            "moderate_exposure":lambda im:(im.astype(float)*.78).astype(np.uint8),
            "sensor_noise":lambda im:np.clip(im.astype(np.int16)+rng.integers(-3,4,im.shape),0,255).astype(np.uint8),
            "low_light":lambda im:(im.astype(float)*.04).astype(np.uint8)}[name]
        if name in ("mirrored_camera","blurred_camera","low_light"):expected = {"CAMERA_QUALITY","CALIBRATION_INVALID"}
    elif name in ("robot_executing","first_frame_unsettled"):
        robot = name=="robot_executing"
        expected = {"ROBOT_EXECUTING"} if robot else {"MOVING"}
        sim.detector.reset_stability()
    elif name in ("disc_falling","jammed_at_top","off_centre_disc","orange_red","white_yellow","glare"):
        slot = 21 if name=="white_yellow" else 0
        if slot==21:
            board[0,0] = RED;game = set_game(sim,board);before = sim.snapshot()
        sim.release(slot,column)
        if name=="disc_falling":sim.step(.05);expected = {"OCCLUDED","INVALID_BOARD","MOVING"}
        elif name=="jammed_at_top":
            sim.put(sim.discs[slot],sim.point("board",column*sim.config["pitch_x"],6*sim.config["pitch_y"]+.005))
            sim.model.body_gravcomp[sim.discs[slot]] = 1;expected = {"OCCLUDED"}
        else:
            sim.step(1.)
            if name=="off_centre_disc":
                sim.put(sim.discs[slot],sim.point("board",column*sim.config["pitch_x"]+.012,0));expected = {"OCCLUDED"}
            if name=="orange_red":paint(sim,slot,[.9,.30,.01,1])
            if name=="white_yellow":paint(sim,slot,[.8,.8,.65,1]);expected = {"OCCLUDED"}
            if name=="glare":paint(sim,slot,[1,1,1,1]);expected = {"OCCLUDED"}
    elif name in ("floating_disc","impossible_counts"):
        board[2,3] = RED if name=="floating_disc" else EMPTY
        if name=="impossible_counts":board[0,:2] = RED
        sim.load(board);expected = {"INVALID_BOARD"}
    elif name in ("reset_empty","reset_leftover","reset_cached_frame"):
        if name!="reset_leftover":
            for i,c in enumerate(sim.supply):
                if not c:sim.put(sim.discs[i],[3,3,.1])
    elif name in ("behind_backing","rgb_depth_limit"):
        # Same-colour material between board and backing can mimic a seated disc.
        # An RGB-only classifier cannot prove depth from an identical projection.
        geom = sim.model.geom("hand").id
        sim.model.geom_size[geom] = [.001,.014,.014]
        sim.model.geom_rgba[geom] = [.9,.1,.1,1]
        sim.data.mocap_pos[0] = sim.point("board",3*sim.config["pitch_x"],0,-.05 if name=="behind_backing" else -.025)
        if name=="rgb_depth_limit":limitation = "RGB sees red material through one hole but cannot prove it is a seated disc. Keep the backing gap protected; use another view/sensor if depth verification is required."

    completed_at = time.time()
    if name in ("camera_disconnected","stale_frame","duplicate_frame","out_of_order_frame","future_frame","wrong_image_size"):
        rgb,stamp = sim.frame()
        if name=="camera_disconnected":rgb = None
        if name=="stale_frame":stamp -= 5
        if name in ("duplicate_frame","out_of_order_frame"):stamp = sim.detector.last_timestamp-(1 if name=="out_of_order_frame" else 0)
        if name=="future_frame":stamp += 5
        if name=="wrong_image_size":rgb = rgb[:100]
        after = sim.detector.process(rgb,stamp)
        sim.last_rgb = sim.frame()[0]
        expected = {"CAMERA_FAILURE"}
    else:
        after = sim.snapshot(effect,robot,frames=1 if name=="first_frame_unsettled" else 12)
    if name.startswith("reset_"):
        try:
            game.reset(before if name=="reset_cached_frame" else after,completed_at)
            accepted = True
        except ValueError as exc:error = str(exc)
        should_commit = name=="reset_empty"
    elif requested and game.pending:
        try:actual = game.finish(after,completed_at);accepted = True
        except ValueError as exc:error = str(exc)
    elif name=="cancelled_drop":
        try:game.finish(after,completed_at)
        except ValueError as exc:error = str(exc)

    truth,stock = sim.truth()
    known = [i for i,v in enumerate(after.cells) if v!=UNKNOWN]
    correct = sum(after.cells[i]==truth[i] for i in known)
    correct_supply = sum(a==b for a,b in zip(after.feeder,stock) if a!=UNKNOWN)
    # The depth case is scored as an explicit LIMIT, never counted as a successful
    # detection of the physically unobservable fault.
    passed = after.status in expected and accepted==should_commit
    if not limitation and after.status=="READY":
        passed &= after.cells==truth and after.feeder==stock
    if limitation:
        passed &= after.cells[3]==RED
    if requested and not accepted:
        passed &= game.cells==initial and game.paused
    engine_move = game.take_move()
    # An accepted reset is not a move. All rejected scenarios must emit nothing.
    passed &= (engine_move is not None)==(actual is not None and accepted)
    result = dict(name=name,passed=bool(passed),expected_status=sorted(expected),actual_status=after.status,
        expected_commit=should_commit,committed=accepted,reason=error or after.reason,actual_move=actual,
        engine_move=engine_move,
        version=game.version,known_cells=len(known),correct_cells=correct,correct_supply=correct_supply,
        truth=truth,truth_feeder=stock,observation=after.as_dict(),limitation=limitation)
    if after.poses.get("board"):
        estimated = np.asarray(after.poses["board"])
        nominal = sim.pose("board")
        points = [[c*sim.config["pitch_x"],r*sim.config["pitch_y"],0,1] for r in (0,5) for c in (0,6)]
        result["max_cell_corner_error_mm"] = max(float(np.linalg.norm((estimated-nominal)@p)*1000) for p in points)
    cv2.imwrite(str(folder/"after.png"),cv2.cvtColor(sim.last_rgb,cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(folder/"perception.png"),cv2.cvtColor(overlay(after),cv2.COLOR_RGB2BGR))
    (folder/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"{'LIMIT' if limitation else 'PASS' if passed else 'FAIL'} {name}: {after.status}; commit={accepted}; {correct}/{len(known)} board cells",flush=True)
    display.wait(after,name+": "+result["reason"],automatic)
    return result


def gallery(results,folder):
    folder.mkdir(parents=True,exist_ok=True)
    cards = []
    for r in results:
        title = "LIMIT" if r["limitation"] else "PASS" if r["passed"] else "FAIL"
        images = ''.join(f'<figure><img src="{r["name"]}/{p}.png"><figcaption>{label}</figcaption></figure>'
            for p,label in (("before","Before"),("after","Front RGB after"),("perception","Actual detector output")))
        cards.append(f'<article><h2>{title}: {r["name"]}</h2><p>{html.escape(r["reason"])}</p>'
            f'<p>Status {r["actual_status"]}; committed {r["committed"]}; {r["correct_cells"]}/{r["known_cells"]} known board cells correct.</p>'
            f'<p>{html.escape(r["limitation"])}</p><div>{images}</div><a href="{r["name"]}/result.json">Measurements and ground truth</a></article>')
    detectable = [r for r in results if not r["limitation"]]
    limits = len(results)-len(detectable)
    page = ('<!doctype html><meta charset="utf-8"><title>Connect 4 perception tests</title>'
        '<style>body{font:16px system-ui;background:#15202d;color:white;margin:25px}article{padding:20px;background:#25364a;margin:20px 0}article div{display:flex}figure{width:33%;margin:5px}img{width:100%}a{color:#9ce0ff}</style>'
        f'<h1>Connect 4 RGB perception: {sum(r["passed"] for r in detectable)}/{len(detectable)} checked outcomes; {limits} documented RGB limit</h1>'
        '<p>Blocking an uncertain scene is a success. LIMIT cases expose RGB observability limits. Discs fall in MuJoCo; the arm is parked and release is scripted. These are synthetic-scene measurements, not hardware grasp validation.</p>'+''.join(cards))
    (folder/"index.html").write_text(page,encoding="utf-8")
    (folder/"summary.json").write_text(json.dumps(results,indent=2),encoding="utf-8")


def play(sim,display,human,output):
    game = Coordinator(sim.config,human)
    # This represents the engine's own board. It receives coordinates only,
    # rather than reading detector cells or simulator body positions.
    engine_board = np.zeros((6,7),np.uint8)
    output.mkdir(parents=True,exist_ok=True)
    move_log = output/"engine_moves.jsonl"
    move_log.write_text("",encoding="utf-8")
    sim.load(np.zeros((6,7),np.uint8))
    while True:
        before = sim.snapshot()
        key = display.show(before,f"1..7: human column | R: reset | version {game.version} | Red starts | scripted releases")
        if key==ord('r'):
            sim.load(np.zeros((6,7),np.uint8));game = Coordinator(sim.config,human)
            engine_board.fill(EMPTY)
            move_log.write_text("",encoding="utf-8")
            continue
        if game.terminal or game.paused:
            time.sleep(.03);continue
        actor = "human" if game.turn==human else "robot"
        column = key-ord('1') if actor=="human" and ord('1')<=key<=ord('7') else None
        if actor=="robot":
            # Temporary test opponent: take an immediate win, block one, then
            # prefer the centre. The team's real engine plugs in at this point.
            options = [c for c in (3,2,4,1,5,0,6) if engine_board[5,c]==EMPTY]
            column = options[0]
            for colour in (game.turn,3-game.turn):
                found = next((c for c in options if colour in winners(drop(engine_board,c,colour)[0])),None)
                if found is not None:column = found;break
        if column is not None:
            try:
                drop(game.cells,column,game.turn)
                request = game.begin(before,actor,column if actor=="robot" else None)
                sim.release(request["slot"],column)
                sim.step(1.1,lambda:display.show(label=actor+": scripted release; arm control pending"))
                completed = time.time();after = sim.snapshot();game.finish(after,completed)
                event = game.take_move()
                engine_board[event["row"],event["column"]] = event["colour"]
                print("Perception -> engine:",json.dumps(event),flush=True)
                with move_log.open("a",encoding="utf-8") as log:
                    log.write(json.dumps(event)+"\n")
                display.show(after,f"Verified column {column+1}; version {game.version}")
            except ValueError as exc:
                if game.pending:game.cancel()
                print("Move refused:",exc,flush=True)
        time.sleep(.03)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--case",choices=CASES,nargs="+")
    parser.add_argument("--demo",action="store_true",help="Advance visual cases automatically")
    parser.add_argument("--play",action="store_true",help="Virtual human vs test opponent; arm motion not implemented")
    parser.add_argument("--human-colour",choices=("red","yellow"),default="red")
    parser.add_argument("--seconds",type=float,help="Close the visual demo after this many wall-clock seconds")
    parser.add_argument("--output",type=Path,default=HERE/"perception_results")
    args = parser.parse_args()
    if args.play and args.headless:parser.error("--play needs the visual windows")
    sim,display,results = Simulation(),None,[]
    try:
        display = Display(sim,args.headless,args.seconds)
        if args.play:
            play(sim,display,RED if args.human_colour=="red" else YELLOW,args.output)
        else:
            for name in args.case or CASES:
                results.append(run_case(sim,name,args.output/name,display,args.demo))
    except KeyboardInterrupt:
        print("Simulation closed.")
    finally:
        if display:display.close()
        sim.close()
        if not args.play:gallery(results,args.output)
    if not args.play:
        print("Open:",args.output/"index.html")
        if any(not r["passed"] for r in results):raise SystemExit(1)


if __name__=="__main__":
    main()
