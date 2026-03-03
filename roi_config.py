# ============================================================
#  roi_config.py  –  ROI definitions + interactive setup tool
# ============================================================

import cv2
import numpy as np

# ── Default ROI (edit these to match your camera view) ──────
# Format: list of (x, y) pixel points forming a polygon
TRACK_ROI = [
    (0, 500),   # top-left near rail entry
    (0, 360),   # top-right just before gate
    (1580, 500),   # bottom-right near crossing exit
    (1420, 1400)    # bottom-left near near rail
]

# ── Crossing Y line (horizontal line a train must cross) ────
CROSSING_Y_PERCENT = 0.55   # 55% down the frame height


# ============================================================
#  Interactive ROI calibration tool
#  Run this file directly:  python roi_config.py
# ============================================================

_clicked_points = []

def _mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _clicked_points.append((x, y))
        img = param["frame"].copy()
        for pt in _clicked_points:
            cv2.circle(img, pt, 6, (0, 255, 0), -1)
        if len(_clicked_points) > 1:
            cv2.polylines(img, [np.array(_clicked_points, np.int32)], False, (0, 255, 0), 2)
        cv2.putText(img, f"Points: {len(_clicked_points)}/4  -  press ENTER when done",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(param["win"], img)


def define_roi_interactively(video_source="railway_video.mp4", roi_name="TRACK"):
    """
    Opens the first frame of the video and lets you click
    4 corner points to define an ROI polygon.
    Returns list of (x, y) tuples.
    """
    global _clicked_points
    _clicked_points = []

    cap = cv2.VideoCapture(video_source)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"[ERROR] Could not open video: {video_source}")
        return TRACK_ROI

    win = f"Define {roi_name} ROI - Click 4 corners then press ENTER"
    param = {"frame": frame, "win": win}

    display = frame.copy()
    cv2.putText(display, "Click 4 corners of the zone - then press ENTER",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)   # create window FIRST
    cv2.imshow(win, display)
    cv2.waitKey(1)                             # let the window render before binding
    cv2.setMouseCallback(win, _mouse_callback, param)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 13 and len(_clicked_points) >= 3:   # ENTER
            break
        if key == 27:                                   # ESC → cancel
            _clicked_points = []
            break

    cv2.destroyAllWindows()

    if len(_clicked_points) >= 3:
        print(f"[INFO] {roi_name} ROI defined: {_clicked_points}")
        return _clicked_points
    else:
        print(f"[WARN] ROI not defined. Using default {roi_name}_ROI.")
        return TRACK_ROI


# ── Run directly to calibrate ────────────────────────────────
if __name__ == "__main__":
    print("=== ROI Calibration Tool ===")
    pts = define_roi_interactively("railway_video.mp4", "TRACK")
    print("\nCopy this into roi_config.py:")
    print(f"TRACK_ROI = {pts}")