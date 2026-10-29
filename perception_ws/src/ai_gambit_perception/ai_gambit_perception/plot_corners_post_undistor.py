"""Mark the chessboard's 4 corners by hand on an undistorted camera snapshot.

Takes one snapshot from the overhead camera, removes the lens's barrel
distortion (same calibration and settings as test_perception.py), and shows
it. Click 4 points in any order; after the 4th an outline joins them, and the
points are saved to board_corners.json (with the snapshot as
board_corners_snapshot.png) next to this script.

Run:  python plot_corners_post_undistor.py [--camera 1] [--image saved.png]

Mouse / keys (the window can be resized and closed with its X button):
  left click   add a point          right click / U   undo the last point
  R            start again          Esc / Q           quit
Clicks map to the snapshot's own pixels whatever the window size.
"""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from test_perception import SETUP, open_camera, undistort_maps

WINDOW = "Undistorted snapshot - click the 4 board corners"
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "board_corners.json"
SNAPSHOT = HERE / "board_corners_snapshot.png"


def undistorted_snapshot(args):
    """One still, undistorted frame: the median of a few frames after the
    camera's exposure settles (less sensor noise than a single frame)."""
    camera = open_camera(args.camera, args.width, args.height)
    try:
        frames, start = [], time.time()
        while time.time() - start < 2.5 or len(frames) < 9:
            ok, frame = camera.read()
            if ok:
                frames.append(frame)
            if time.time() - start > 10:
                break
    finally:
        camera.release()
    if not frames:
        raise SystemExit(f"Camera {args.camera} gave no frames")
    still = np.median(np.array(frames[-9:]), axis=0).astype(np.uint8)
    maps = undistort_maps(args.setup, (still.shape[1], still.shape[0]))
    return cv2.remap(still, *maps, cv2.INTER_LINEAR)


def in_image_order(points):
    """The 4 points as top-left, top-right, bottom-right, bottom-left (image
    pixels, y down), however they were clicked."""
    pts = np.asarray(points, float)
    centre = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0]))]  # Clockwise on screen.
    start = int(np.argmin(pts.sum(axis=1)))  # Top-left: smallest x + y.
    return np.roll(pts, -start, axis=0)


def draw(image, points):
    view = image.copy()
    if len(points) == 4:
        cv2.polylines(view, [in_image_order(points).round().astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(points, 1):
        cv2.circle(view, (round(x), round(y)), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(view, str(i), (round(x) + 9, round(y) - 9), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2, cv2.LINE_AA)
    text = (f"click point {len(points) + 1} of 4   (right click/U: undo)" if len(points) < 4
            else f"saved to {OUTPUT.name}   (R: start again, Esc: quit)")
    cv2.rectangle(view, (0, 0), (view.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(view, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .8, (40, 220, 255), 2, cv2.LINE_AA)
    return view


def save(points, image, args):
    cv2.imwrite(str(SNAPSHOT), image)
    ordered = in_image_order(points)
    OUTPUT.write_text(json.dumps({
        "snapshot": SNAPSHOT.name,
        "image_size": [image.shape[1], image.shape[0]],
        "undistorted_with": str(args.setup),
        "clicked": [[round(x, 1), round(y, 1)] for x, y in points],
        "corners": {name: [round(float(x), 1), round(float(y), 1)]
                    for name, (x, y) in zip(("top_left", "top_right", "bottom_right", "bottom_left"), ordered)},
    }, indent=2))
    print(f"Saved {OUTPUT}")
    for name, (x, y) in zip(("top-left", "top-right", "bottom-right", "bottom-left"), ordered):
        print(f"  {name:12s} ({x:.1f}, {y:.1f}) px")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=1, help="Camera index (1 = the USB overhead camera here)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--setup", default=str(SETUP), help="real_setup.json holding the lens calibration")
    parser.add_argument("--image", help="Use this already-undistorted image instead of taking a snapshot")
    args = parser.parse_args()

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise SystemExit(f"Cannot read {args.image}")
    else:
        print("Taking an undistorted snapshot...")
        image = undistorted_snapshot(args)
    points = []

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((float(x), float(y)))
            if len(points) == 4:
                save(points, image, args)
        elif event == cv2.EVENT_RBUTTONDOWN and points and len(points) < 4:
            points.pop()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)  # Resizable; clicks still arrive in image pixels.
    cv2.resizeWindow(WINDOW, image.shape[1], image.shape[0])
    cv2.setMouseCallback(WINDOW, click)
    try:
        while True:
            cv2.imshow(WINDOW, draw(image, points))
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("u") and points and len(points) < 4:
                points.pop()
            if key == ord("r"):
                points.clear()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
