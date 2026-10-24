"""Rendered MuJoCo perception experiments and an inspectable HTML results gallery.

Run normally to step through BEFORE and AFTER views. Space/N advances; Esc exits.
Use --headless to run all cases and save PNGs, observations and measured scores.
Every PASS checks a specified outcome; an intentionally blocked scene is a PASS
only when perception actually blocks it. Ground truth is used for scoring only.
"""
import argparse
import html
import json
from pathlib import Path
import time
import cv2
import numpy as np
import chess
import mujoco
from perception_sim import Simulation, HERE, overlay, show_observation, tray_centers
from chess_game import Coordinator, occupancy, candidates
from chess_coords import square_to_pos, pos_to_square, HOME_POSE

CASTLE = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
EP = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
PROMOTE = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"
CAPTURE = "4k3/8/8/3p1p2/4P3/8/8/4K3 w - - 0 1"


# Data, not a separate class/file for every scenario. Keep expected responses
# independent of the image detector so these tests can catch detector mistakes.
CASES = [
    ("opening", chess.STARTING_FEN, None, "none", "READY", True),
    ("normal_move", chess.STARTING_FEN, "e2e4", "move", "READY", True),
    ("knight_move", chess.STARTING_FEN, "g1f3", "move", "READY", True),
    ("capture", CAPTURE, "e4d5", "move", "READY", True),
    ("capture_ambiguity", CAPTURE, "e4d5", "ambiguous", "READY", True),
    ("capture_storage_missing", CAPTURE, "e4d5", "missing_capture", "READY", False),
    ("last_white_storage_slot", chess.STARTING_FEN, None, "last_white", "READY", True),
    ("last_black_storage_slot", chess.STARTING_FEN, None, "last_black", "READY", True),
    ("maximum_promotion_storage", "4k3/8/8/8/8/8/8/4K3 w - - 0 1", None, "capacity", "READY", True),
    ("kingside_castle", CASTLE, "e1g1", "move", "READY", True),
    ("queenside_castle", CASTLE, "e1c1", "move", "READY", True),
    ("black_kingside_castle", CASTLE.replace(" w "," b "), "e8g8", "move", "READY", True),
    ("black_queenside_castle", CASTLE.replace(" w "," b "), "e8c8", "move", "READY", True),
    ("partial_castle", CASTLE, "e1g1", "partial", "READY", False),
    ("en_passant", EP, "e5d6", "move", "READY", True),
    ("incomplete_en_passant", EP, "e5d6", "partial", "READY", False),
    ("black_en_passant", "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1", "e4d3", "move", "READY", True),
    ("promotion_queen", PROMOTE, "a7a8q", "move", "READY", True),
    ("promotion_rook", PROMOTE, "a7a8r", "move", "READY", True),
    ("promotion_bishop", PROMOTE, "a7a8b", "move", "READY", True),
    ("promotion_knight", PROMOTE, "a7a8n", "move", "READY", True),
    ("capture_promotion", "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8q", "move", "READY", True),
    ("promotion_ambiguous", PROMOTE, "a7a8q", "ambiguous", "READY", True),
    ("promotion_spare_not_removed", PROMOTE, "a7a8q", "bad_promotion", "READY", False),
    ("touch_and_return", chess.STARTING_FEN, None, "touch", "READY", True),
    ("identity_swap_not_observable", chess.STARTING_FEN, None, "identity_swap", "READY", True),
    ("illegal_three_square_pawn", chess.STARTING_FEN, "e2e4", "illegal", "READY", False),
    ("failed_pickup", chess.STARTING_FEN, "e2e4", "failed", "READY", False),
    ("wrong_square", chess.STARTING_FEN, "e2e4", "wrong", "READY", False),
    ("accidental_second_move", chess.STARTING_FEN, "e2e4", "extra", "READY", False),
    ("offset_3mm", chess.STARTING_FEN, None, "offset_small", "READY", True),
    ("offset_12mm", chess.STARTING_FEN, None, "offset_large", "OCCLUDED", False),
    ("between_squares", chess.STARTING_FEN, None, "between", "OCCLUDED", False),
    ("toppled_piece", chess.STARTING_FEN, None, "toppled", "OCCLUDED", False),
    ("hand_on_board", chess.STARTING_FEN, None, "hand", "OCCLUDED", False),
    ("hand_on_capture_tray", chess.STARTING_FEN, None, "tray_hand", "OCCLUDED", False),
    ("object_on_capture_tray", chess.STARTING_FEN, None, "tray_object", "OCCLUDED", False),
    ("foreign_object", chess.STARTING_FEN, None, "foreign", "OCCLUDED", False),
    ("object_in_workspace_gap", chess.STARTING_FEN, None, "apron", "OCCLUDED", False),
    ("arm_blocks_board", chess.STARTING_FEN, None, "arm", "OCCLUDED", False),
    ("one_board_tag_hidden", chess.STARTING_FEN, None, "one_tag", "READY", True),
    ("one_corrupted_tag", chess.STARTING_FEN, None, "corrupt_tag", "READY", True),
    ("all_board_tags_hidden", chess.STARTING_FEN, None, "board_tags", "CALIBRATION_INVALID", False),
    ("rack_tags_hidden", chess.STARTING_FEN, None, "rack_tags", "CALIBRATION_INVALID", False),
    ("tray_tags_hidden", chess.STARTING_FEN, None, "tray_tags", "CALIBRATION_INVALID", False),
    ("board_shifted", chess.STARTING_FEN, None, "board_shift", "CALIBRATION_INVALID", False),
    ("camera_shifted", chess.STARTING_FEN, None, "camera_shift", "CALIBRATION_INVALID", False),
    ("moderate_exposure", chess.STARTING_FEN, None, "exposure", "READY", True),
    ("sensor_noise", chess.STARTING_FEN, None, "noise", "READY", True),
    ("severe_blur", chess.STARTING_FEN, None, "blur", "CALIBRATION_INVALID", False),
    ("missing_rack_piece", chess.STARTING_FEN, None, "inventory", "READY", False),
    ("camera_disconnected", chess.STARTING_FEN, None, "disconnect", "CAMERA_FAILURE", False),
    ("stale_frame", chess.STARTING_FEN, None, "stale", "CAMERA_FAILURE", False),
    ("out_of_order_frame", chess.STARTING_FEN, None, "out_of_order", "CAMERA_FAILURE", False),
    ("robot_in_progress", chess.STARTING_FEN, None, "executing", "ROBOT_EXECUTING", False),
    ("actual_arm_move", chess.STARTING_FEN, "e2e4", "robot_move", "READY", True),
]


def arrange(sim, board, uci, action, animate_viewer=None, observation=None):
    """Apply physical changes. Return the intended result separately from truth."""
    expected = board.copy()
    rack, tray = [1]*32, [0]*48
    effect = None
    kwargs = {}
    move = chess.Move.from_uci(uci) if uci else None
    if move:
        expected.push(move)
    moving = {"move", "ambiguous", "missing_capture", "bad_promotion", "extra", "robot_move"}
    if action in moving:
        mapping, rack, tray = sim.physical_move(board, move, animate=action=="robot_move", viewer=animate_viewer,observation=observation if action=="robot_move" else None)
        if action == "missing_capture":
            sim.put(sim.mapping[move.to_square],[0,0,-2])
        if action == "bad_promotion":
            # Restore the spare at the rack while the pawn still occupies a8.
            spare = mapping[move.to_square]
            sim.put(spare,[0,0,-2])
            pawn = sim.mapping[move.from_square]
            sim.put(pawn,square_to_pos("a8",.014))
            # Put another matching compact proxy back into its original slot.
            q = sim.model.jnt_qposadr[sim.model.body_jntadr[spare]]
            sim.data.qpos[q:q+7] = sim.initial_qpos[q:q+7]
        if action == "extra":
            sim.put(sim.mapping[chess.D2],square_to_pos("d4",.014))
    elif action == "touch":
        sim.put(sim.mapping[chess.E2],square_to_pos("e3",.014))
        sim.detector.process(sim.frame())
        sim.put(sim.mapping[chess.E2],square_to_pos("e2",.014))
    elif action == "identity_swap":
        sim.put(sim.mapping[chess.A2],square_to_pos("b2",.014))
        sim.put(sim.mapping[chess.B2],square_to_pos("a2",.014))
    elif action in ("last_white","last_black"):
        square,slot = (chess.E2,23) if action == "last_white" else (chess.E7,47)
        target = tray_centers()[slot].copy()
        target[2] = .024
        sim.put(sim.mapping[square],target)
        expected.remove_piece_at(square)
        tray[slot] = 1
    elif action == "capacity":
        # Worst-case physical inventory: 15 original non-kings plus 8 used
        # promotion replacements per colour. The two kings remain on the board.
        for color,row in (("w",0),("b",1)):
            originals = [b for b in sim.piece_ids if sim.model.body(b).name.startswith(color+"_")
                         and "_king_" not in sim.model.body(b).name]
            spares = [sim.model.body(f"spare_{color}_{i:02d}").id for i in range(8)]
            for i,body in enumerate(originals+spares):
                slot = row*24+i
                target = tray_centers()[slot].copy()
                body_name = sim.model.body(body).name
                from chess_coords import PIECE_REST_Z
                target[2] = .022 if body_name.startswith("spare_") else PIECE_REST_Z[body_name.split('_')[1]]
                sim.put(body,target)
                tray[slot] = 1
            rack[row*16:row*16+8] = [0]*8
    elif action in ("partial", "wrong", "illegal"):
        target = "e5" if action == "illegal" else "e3" if action == "wrong" else chess.square_name(move.to_square)
        kind = chess.piece_name(board.piece_at(move.from_square).piece_type)
        from chess_coords import PIECE_REST_Z
        sim.put(sim.mapping[move.from_square],square_to_pos(target,PIECE_REST_Z[kind]-.01))
    elif action.startswith("offset") or action == "between":
        p = square_to_pos("e2",.014)
        p[1] += .003 if action == "offset_small" else .012 if action == "offset_large" else .015
        sim.put(sim.mapping[chess.E2],p)
    elif action == "toppled":
        sim.put(sim.mapping[chess.E2], square_to_pos("e2",.009),[.70710678,0,.70710678,0])
    elif action in ("hand","foreign","apron"):
        body = sim.model.body("foreign" if action == "apron" else action).id
        sim.data.mocap_pos[sim.model.body_mocapid[body]] = [.38,.22,.045] if action == "apron" else [.25,0,.055]
    elif action in ("tray_hand","tray_object"):
        body = sim.model.body("hand" if action == "tray_hand" else "foreign").id
        sim.data.mocap_pos[sim.model.body_mocapid[body]] = [.25,.19,.05]
    elif action == "arm":
        sim.data.qpos[:6], sim.data.ctrl[:6] = HOME_POSE, HOME_POSE
    elif action.endswith("tag") or action.endswith("tags"):
        ids = [0] if action == "one_tag" else range(4) if action == "board_tags" else range(4,8) if action == "rack_tags" else range(8,12)
        if action == "corrupt_tag":
            sim.model.geom_pos[sim.model.geom("tag_0").id,0] += .025
        else:
            for i in ids:
                sim.model.geom_rgba[sim.model.geom(f"tag_{i}").id,3] = 0
    elif action == "board_shift":
        sim.model.body_pos[sim.model.body("chessboard").id,0] += .02
        sim.model.body_pos[sim.model.body("board_fiducials").id,0] += .02
    elif action == "camera_shift":
        sim.model.cam_pos[sim.model.camera("top_camera").id,0] += .025
    elif action == "exposure":
        effect = lambda im: np.clip(im.astype(float)*.83+7,0,255).astype(np.uint8)
    elif action == "noise":
        rng = np.random.default_rng(23)
        effect = lambda im: np.clip(im.astype(float)+rng.normal(0,2,im.shape),0,255).astype(np.uint8)
    elif action == "blur":
        effect = lambda im: cv2.GaussianBlur(im,(31,31),9)
    elif action == "inventory":
        sim.put("spare_w_00",[0,0,-2])
    elif action == "stale":
        kwargs = {"timestamp":time.time()-10}
    elif action == "out_of_order":
        kwargs = {"timestamp":sim.detector.last_timestamp-1}
    elif action == "executing":
        kwargs = {"robot_executing":True}
    return expected,rack,tray,effect,kwargs


def ground_truth(sim):
    result = [0]*64
    mujoco.mj_forward(sim.model,sim.data)
    for b in sim.piece_ids:
        p = sim.data.xpos[b]
        name = pos_to_square(p[0],p[1])
        if name and p[2] > 0:
            result[chess.parse_square(name)] = 1
    return result


def wait_for_next(sim, obs, message):
    show_observation(sim,obs,"Experiments",message+" | Space/N: next; Esc: exit")
    while True:
        key = cv2.waitKey(30) & 255
        if key in (32,ord('n'),13):
            return True
        if key == 27:
            return False


def run_case(sim, case, folder, interactive=False):
    name, fen, uci, action, status, should_verify = case
    board = chess.Board(fen)
    sim.load_position(board)
    before = sim.snapshot()
    baseline_ok, baseline_reason = sim.detector.establish_baseline(before,occupancy(board),0,rack=[1]*32,tray=[0]*48)
    if not baseline_ok:
        raise AssertionError(f"{name}: baseline invalid: {baseline_reason}")
    folder.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(folder/"before.png"),cv2.cvtColor(sim.frame(),cv2.COLOR_RGB2BGR))
    if interactive and not wait_for_next(sim,before,name+": BEFORE"):
        raise KeyboardInterrupt
    started = time.time()
    expected,rack,tray,effect,kwargs = arrange(sim,board,uci,action,observation=before)
    if action == "disconnect":
        after = sim.detector.process(None)
        final_image = sim.frame()
    else:
        after = sim.snapshot(effect=effect,**kwargs)
        final_image = sim.frame()
        if effect:
            final_image = effect(final_image)
    verified,reason = sim.detector.verify(after,occupancy(expected),version=0,after=started,rack=rack,tray=tray)
    options = candidates(board,after)
    candidate_ok = True
    if uci and should_verify:
        candidate_ok = uci in options
    if action == "ambiguous":
        candidate_ok = len(options) > 1
    if action in ("partial","illegal","extra","failed"):
        candidate_ok = not options
    if action == "wrong":
        candidate_ok = uci not in options  # e2e3 is legal, but is not the requested e2e4.
    truth = ground_truth(sim)
    known = [i for i,v in enumerate(after.occupancy) if v != 255]
    correct = sum(after.occupancy[i] == truth[i] for i in known)
    score_ok = after.status != "READY" or (len(known) == 64 and correct == 64)
    passed = after.status == status and verified == should_verify and candidate_ok and score_ok
    pose_error, center_errors = None, []
    if after.status == "READY":
        from ai_gambit_perception.perception import transform
        estimated = np.array(after.poses["board"])
        nominal = transform(sim.config["board_rotation"],sim.config["board_origin"])
        corners = np.array([[0,0,0,1],[.24,0,0,1],[0,.24,0,1],[.24,.24,0,1]])
        pose_error = float(np.max(np.linalg.norm((estimated-nominal) @ corners.T,axis=0))*1000)
        for body in sim.piece_ids:
            position = sim.data.xpos[body]
            square_name = pos_to_square(position[0],position[1])
            if square_name and position[2] > 0:
                square = chess.parse_square(square_name)
                local = [(square%8+.5)*.03+after.offsets[square][0],
                         (square//8+.5)*.03+after.offsets[square][1],0,1]
                predicted = estimated @ local
                center_errors.append(float(np.linalg.norm(predicted[:2]-position[:2])*1000))
    result = {"name":name,"passed":passed,"expected_status":status,"actual_status":after.status,
              "expected_verification":should_verify,"verified":verified,"reason":reason,
              "legal_candidates":options,"candidate_check":candidate_ok,"known_squares":len(known),
              "correct_known_squares":correct,"ground_truth":truth,"observation":after.as_dict(),
              "max_board_corner_error_mm":pose_error,
              "mean_approximate_piece_center_error_mm":float(np.mean(center_errors)) if center_errors else None}
    if action == "identity_swap":
        result["limitation"] = "Swapped identities cannot be detected from binary occupancy alone. This is an explicit observability limit."
    cv2.imwrite(str(folder/"after.png"),cv2.cvtColor(final_image,cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(folder/"occupancy.png"),cv2.cvtColor(overlay(after),cv2.COLOR_RGB2BGR))
    (folder/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"{'PASS' if passed else 'FAIL'} {name}: {after.status}; verify={verified}; {correct}/{len(known)} visible squares",flush=True)
    if interactive and not wait_for_next(sim,after,name+": AFTER - "+reason):
        raise KeyboardInterrupt
    return result


def write_gallery(results, folder):
    cards = []
    for r in results:
        name = r["name"]
        pose_error = r.get("max_board_corner_error_mm")
        center_error = r.get("mean_approximate_piece_center_error_mm")
        geometry = ""
        if pose_error is not None:
            geometry = f'<p>Maximum board-corner error: {pose_error:.2f} mm.'
            if center_error is not None:
                geometry += f' Mean approximate visual piece-centre error: {center_error:.2f} mm.'
            geometry += ' Simulation measurements; not hardware accuracy.</p>'
        limitation = '<p>'+html.escape(r.get("limitation",""))+'</p>' if r.get("limitation") else ''
        images = ''.join(f'<figure><img src="{name}/{file}.png"><figcaption>{label}</figcaption></figure>'
                         for file,label in (("before","Before"),("after","After: camera"),("occupancy","After: perception")))
        cards.append(f'<article><h2>{"PASS" if r["passed"] else "FAIL"} — {name}</h2>'
                     f'<p>Expected {r["expected_status"]}; observed {r["actual_status"]}. '
                     f'Verification {r["verified"]}. {r["correct_known_squares"]}/{r["known_squares"]} known squares correct.</p>'
                     f'<p>{html.escape(r["reason"])}<br>Legal candidates: {html.escape(", ".join(r["legal_candidates"]) or "none")}</p>'
                     f'{geometry}{limitation}<div>{images}</div><a href="{name}/result.json">Full observation and ground truth</a></article>')
    passed = sum(r["passed"] for r in results)
    page = f'''<!doctype html><meta charset="utf-8"><title>Ai-Gambit perception experiments</title>
    <style>body{{font:16px system-ui;background:#111923;color:#eee;margin:30px}}article{{background:#202c3b;padding:20px;margin:20px 0;border-radius:12px}}article div{{display:flex}}figure{{margin:6px;width:33%}}img{{width:100%}}a{{color:#8dd4ff}}h2{{font-size:20px}}</style>
    <h1>MuJoCo RGB perception: {passed}/{len(results)} expected outcomes</h1>
    <p>Green = occupied, grey = empty, orange = unknown. Blocking an uncertain scene is an expected success.
    Ground truth scores evaluate image output; simulator coordinates never enter the detector.
    Occupancy cannot identify swapped pieces, prove capture identity, or choose a promotion type.</p>{''.join(cards)}'''
    (folder/"index.html").write_text(page,encoding="utf-8")
    (folder/"summary.json").write_text(json.dumps(results,indent=2),encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--case",choices=[c[0] for c in CASES],nargs="+")
    parser.add_argument("--output",type=Path,default=HERE/"perception_results")
    args = parser.parse_args()
    sim, results = Simulation(), []
    try:
        for case in CASES:
            if not args.case or case[0] in args.case:
                results.append(run_case(sim,case,args.output/case[0],not args.headless))
    except KeyboardInterrupt:
        print("Experiments stopped.")
    finally:
        sim.close()
        cv2.destroyAllWindows()
        write_gallery(results,args.output)
    print("Open:",args.output/"index.html")
    if any(not r["passed"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
