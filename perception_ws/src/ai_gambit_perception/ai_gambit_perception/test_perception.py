"""Live view of the overhead camera: raw and lens-corrected side by side,
with AprilTag detections on both.

The correction uses the lens calibration (camera_matrix, distortion) saved in
so101_simulation/real_setup.json by `real_setup.py intrinsics`; it is only
valid at the resolution it was measured at (1280x720 here). Every captured
pixel is kept, so the corrected image has curved black borders.

Run:  python test_perception.py [--camera 1] [--setup path/to/real_setup.json]
The window can be resized and closed with its X button (or Esc / Q).
S saves a snapshot of both views next to this file.
"""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np

WINDOW = "Live camera - raw | undistorted"
SETUP = Path(__file__).resolve().parents[4] / "so101_simulation" / "real_setup.json"


def open_camera(index, width, height):
    # Media Foundation is the backend that delivers frames from this camera on Windows.
    camera = cv2.VideoCapture(index, cv2.CAP_MSMF if hasattr(cv2, "CAP_MSMF") else cv2.CAP_ANY)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not camera.isOpened():
        raise SystemExit(f"Camera {index} could not be opened (is another program using it?)")
    return camera


def undistort_maps(setup_path, size):
    """Pixel maps that straighten the barrel distortion, computed once.
    alpha=1 keeps every captured pixel (black borders where nothing was seen)."""
    setup = json.loads(Path(setup_path).read_text())
    if setup.get("camera_matrix") is None:
        raise SystemExit(f"{setup_path} has no lens calibration; run real_setup.py intrinsics first")
    K, distortion = np.array(setup["camera_matrix"]), np.array(setup["distortion"])
    calibrated = setup.get("camera", {})
    if (calibrated.get("width"), calibrated.get("height")) != size:
        raise SystemExit(f"Lens calibration was made at {calibrated.get('width')}x{calibrated.get('height')}, "
                         f"but the camera gives {size[0]}x{size[1]}")
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, distortion, size, 1, size)
    return cv2.initUndistortRectifyMap(K, distortion, None, new_K, size, cv2.CV_16SC2)


def draw_tags(frame, detector, title):
    corners, ids, _ = detector.detectMarkers(frame)
    seen = [] if ids is None else sorted(int(i) for i in ids.flat)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, f"{title}   tags seen: {seen or 'none'}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (40, 220, 255), 2, cv2.LINE_AA)
    return seen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=1, help="Camera index (1 = the USB overhead camera here)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--setup", default=str(SETUP), help="real_setup.json holding the lens calibration")
    args = parser.parse_args()

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    camera = open_camera(args.camera, args.width, args.height)
    maps = None
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)  # Resizable.
    cv2.resizeWindow(WINDOW, 1600, 450)
    shots = 0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                time.sleep(.05)
                continue
            if maps is None:  # Built from the first frame's actual size.
                maps = undistort_maps(args.setup, (frame.shape[1], frame.shape[0]))
            corrected = cv2.remap(frame, *maps, cv2.INTER_LINEAR)
            draw_tags(frame, detector, "raw")
            draw_tags(corrected, detector, "undistorted")
            view = np.hstack([frame, corrected])
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            # The X button hides the window; stop then rather than reopening it.
            if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("s"):
                shots += 1
                path = Path(__file__).with_name(f"snapshot_{shots}.png")
                cv2.imwrite(str(path), view)
                print("saved", path)
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
