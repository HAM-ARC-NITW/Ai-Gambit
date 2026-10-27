"""Run explicitly: python -m pytest connect_4_perception/test_perception.py -q.
The larger rendered scenario suite is run_perception.py --headless.
"""
import copy
import json
import time
import cv2
import numpy as np
import pytest
from perception_sim import Simulation
from ai_gambit_perception.connect4 import (Observation,Perception,Coordinator,Gate,
    EMPTY,RED,YELLOW,UNKNOWN,drop,board_error,winners,load_config,transform)


def observation(cells=None,feeder=None,sequence=1):
    return Observation(sequence,time.time(),status="READY",stable=True,
        cells=[0]*42 if cells is None else np.asarray(cells).ravel().tolist(),
        confidence=[.99]*42,feeder=[RED]*21+[YELLOW]*21 if feeder is None else feeder.copy(),
        feeder_confidence=[.99]*42,poses={"board":np.eye(4).tolist(),"feeder":np.eye(4).tolist()})


@pytest.mark.parametrize("row",range(6))
@pytest.mark.parametrize("column",range(7))
def test_gravity_mapping_and_colour(row,column):
    board = np.zeros((6,7),np.uint8)
    board[:row,column] = [RED if r%2==0 else YELLOW for r in range(row)]
    original = board.copy()
    result,actual_row = drop(board,column,RED if row%2==0 else YELLOW)
    assert actual_row==row and result[row,column]!=EMPTY
    np.testing.assert_array_equal(board,original)
    assert not board_error(result)


@pytest.mark.parametrize("dr,dc",[(0,1),(1,0),(1,1),(1,-1)])
def test_engine_wins_in_all_directions(dr,dc):
    board = np.zeros((6,7),np.uint8)
    start_col = 3 if dc<0 else 0
    for k in range(4):board[k*dr,start_col+k*dc] = YELLOW
    assert winners(board)=={YELLOW}


def test_full_column_and_invalid_arguments():
    board = np.zeros((6,7),np.uint8);board[:,2] = [RED,YELLOW]*3
    for col,colour in ((2,RED),(-1,RED),(7,RED),(3,UNKNOWN),(2.5,RED),(True,RED),("3",RED)):
        with pytest.raises(ValueError):drop(board,col,colour)


@pytest.mark.parametrize("fault",["status","unknown","confidence","stale","future","sequence","time","version","operation","board","feeder"])
def test_failed_verification_is_atomic(fault):
    game = Coordinator();before = observation();request = game.begin(before,"human")
    expected,_ = drop(game.cells,3,RED);stock=game.feeder.copy();stock[request["slot"]]=EMPTY
    after = observation(expected,stock,2);after_time = before.timestamp
    if fault=="status":after.status="MOVING"
    if fault=="unknown":after.cells[4]=UNKNOWN
    if fault=="confidence":after.confidence[3]=.4
    if fault=="stale":after.timestamp-=5
    if fault=="future":after.timestamp+=5
    if fault=="sequence":after.observation_id=before.observation_id
    if fault=="time":after_time=after.timestamp+1
    if fault=="version":game.version=99
    if fault=="operation":game.pending["operation"]="another-operation"
    if fault=="board":after.cells[4]=YELLOW
    if fault=="feeder":after.feeder[1]=EMPTY
    initial=game.cells.copy()
    with pytest.raises(ValueError):game.finish(after,after_time)
    assert game.cells==initial and game.paused and game.pending is None
    assert game.take_move() is None


@pytest.mark.parametrize("row,column",[(0,0),(0,6),(5,6),(2,3)])
def test_engine_receives_verified_coordinate_once(row,column):
    colour = RED if row%2==0 else YELLOW
    game = Coordinator(human_colour=colour)
    for r in range(row):game.cells[r*7+column] = RED if r%2==0 else YELLOW
    for value in (RED,YELLOW):
        for _ in range(game.cells.count(value)):game.feeder[game.feeder.index(value)] = EMPTY
    game.version = row
    before = observation(game.cells,game.feeder)
    req = game.begin(before,"human")
    assert game.take_move() is None  # A baseline is not a detected move.
    expected,_ = drop(game.cells,column,colour)
    stock = game.feeder.copy();stock[req["slot"]] = EMPTY
    after = observation(expected,stock,2)
    game.finish(after,before.timestamp)
    event = game.take_move()
    assert event==dict(row=row,column=column,index=row*7+column,colour=colour,
        operation_id=req["operation"],game_version=row+1,observation_id=2,timestamp=after.timestamp)
    assert game.take_move() is None
    with pytest.raises(ValueError):game.finish(after,before.timestamp)
    assert game.take_move() is None


@pytest.mark.parametrize("indices",[[],[0,1]])
def test_ros_gate_does_not_emit_no_change_or_multiple_cells(indices):
    gate = Gate();before = observation();gate.begin(before,before.cells,before.feeder,0,"move")
    cells,stock = before.cells.copy(),before.feeder.copy()
    for index in indices:cells[index]=RED;stock[index]=EMPTY
    after = observation(cells,stock,2)
    with pytest.raises(ValueError,match="exactly one"):
        gate.verify(after,cells,stock,0,"move",before.timestamp)
    assert "move" not in gate.completed


def test_reset_discards_unconsumed_engine_moves():
    game = Coordinator();before = observation();req = game.begin(before,"human")
    cells,_ = drop(game.cells,3,RED);stock=game.feeder.copy();stock[req["slot"]]=EMPTY
    game.finish(observation(cells,stock,2),before.timestamp)
    empty = observation(sequence=3)
    game.reset(empty,before.timestamp)
    assert game.take_move() is None


def test_single_commit_wrong_turn_and_replay():
    game=Coordinator();before=observation();req=game.begin(before,"human")
    expected,_=drop(game.cells,6,RED);stock=game.feeder.copy();stock[req["slot"]]=EMPTY
    after=observation(expected,stock,2)
    assert game.finish(after,before.timestamp)==(6,0)
    assert game.version==1
    with pytest.raises(ValueError):game.finish(after,before.timestamp)
    with pytest.raises(ValueError):game.begin(after,"human")
    with pytest.raises(ValueError):game.gate.begin(after,game.cells,game.feeder,1,req["operation"])


def test_wrong_robot_column_cannot_be_committed():
    game=Coordinator();game.cells[3]=RED;game.feeder[0]=EMPTY
    before=observation(game.cells,game.feeder);req=game.begin(before,"robot",3)
    expected,_=drop(game.cells,2,YELLOW);stock=game.feeder.copy();stock[req["slot"]]=EMPTY
    with pytest.raises(ValueError,match="Wrong column"):
        game.finish(observation(expected,stock,2),before.timestamp)
    assert game.version==0 and game.cells[2]==EMPTY and game.paused


def test_human_colour_is_configurable():
    game=Coordinator(human_colour=YELLOW)
    with pytest.raises(ValueError):game.begin(observation(),"human")
    assert game.begin(observation(),"robot",3)["colour"]==RED


def test_baseline_failure_revokes_earlier_authorization():
    gate=Gate();obs=observation();gate.begin(obs,obs.cells,obs.feeder,0,"one")
    wrong=obs.cells.copy();wrong[0]=RED
    with pytest.raises(ValueError):gate.begin(obs,wrong,obs.feeder,0,"two")
    assert gate.baseline is None


def test_reset_needs_post_completion_exposure_and_preserves_replay_guard():
    game=Coordinator();obs=observation();game.gate.completed.add("old")
    with pytest.raises(ValueError):game.reset(obs,obs.timestamp+1)
    game.reset(obs,obs.timestamp-.1)
    assert "old" in game.gate.completed
    assert game.version==0 and not game.paused


@pytest.mark.parametrize("fault",["none","shape","dtype","stale","future","duplicate","out_of_order"])
def test_camera_contract(fault):
    detector=Perception();cfg=detector.config
    frame=np.zeros((cfg["image_height"],cfg["image_width"],3),np.uint8);stamp=time.time()
    if fault=="none":frame=None
    if fault=="shape":frame=frame[:50]
    if fault=="dtype":frame=frame.astype(float)
    if fault=="stale":stamp-=5
    if fault=="future":stamp+=5
    if fault in ("duplicate","out_of_order"):
        detector.last_timestamp=stamp+(1 if fault=="out_of_order" else 0)
    assert detector.process(frame,stamp).status=="CAMERA_FAILURE"


@pytest.fixture(scope="module")
def sim():
    instance=Simulation()
    yield instance
    instance.close()


def test_rendered_empty_grid_and_pose(sim):
    sim.load(np.zeros((6,7),np.uint8));obs=sim.snapshot()
    assert obs.status=="READY" and obs.cells==[EMPTY]*42
    assert obs.feeder==[RED]*21+[YELLOW]*21 and len(obs.tags)==8
    assert len(sim.detector.board_regions)==42
    for group in ("board","feeder"):
        assert np.linalg.norm(np.asarray(obs.poses[group])[:3,3]-sim.pose(group)[:3,3])<.002


def test_real_gravity_drop_and_fresh_verification(sim):
    sim.load(np.zeros((6,7),np.uint8));before=sim.snapshot();game=Coordinator(sim.config)
    request=game.begin(before,"human");sim.release(request["slot"],3)
    sim.step(1.2);completed=time.time();after=sim.snapshot()
    assert after.timestamp>completed and after.cells==sim.truth()[0] and after.feeder==sim.truth()[1]
    assert game.finish(after,completed)==(3,0)
    assert game.version==1


def test_unknown_hand_cannot_be_a_move(sim):
    sim.load(np.zeros((6,7),np.uint8));before=sim.snapshot();game=Coordinator(sim.config)
    game.begin(before,"human");sim.block();after=sim.snapshot()
    assert after.status=="OCCLUDED"
    with pytest.raises(ValueError):game.finish(after,before.timestamp)
    assert game.cells==[EMPTY]*42


def test_robot_inhibit_clears_stability(sim):
    sim.load(np.zeros((6,7),np.uint8));sim.snapshot();rgb,stamp=sim.frame()
    assert sim.detector.process(rgb,stamp,robot_executing=True).status=="ROBOT_EXECUTING"
    rgb,stamp=sim.frame()
    assert sim.detector.process(rgb,stamp).status=="MOVING"


def test_keyboard_game_feeds_engine_and_logs_both_players(sim,tmp_path):
    from run_perception import play

    class TwoTurns:
        turns = 0

        def show(self,obs=None,label=""):
            if label.startswith("1..7"):
                self.turns += 1
                if self.turns==3:raise KeyboardInterrupt
                return ord('4') if self.turns==1 else -1
            return -1

    with pytest.raises(KeyboardInterrupt):play(sim,TwoTurns(),RED,tmp_path)
    events = [json.loads(line) for line in (tmp_path/"engine_moves.jsonl").read_text().splitlines()]
    assert [(m["row"],m["column"],m["colour"]) for m in events]==[(0,3,RED),(1,3,YELLOW)]
    assert [m["game_version"] for m in events]==[1,2]
    assert len({m["operation_id"] for m in events})==2
    assert sim.truth()[0][3]==RED and sim.truth()[0][10]==YELLOW
