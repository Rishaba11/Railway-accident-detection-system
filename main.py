# ============================================================
#  main.py  –  Railway Accident Prevention System
# ============================================================

import cv2
import numpy as np
import time
import os
import threading
from collections import deque

from ultralytics import YOLO
from roi_config import TRACK_ROI, CROSSING_Y_PERCENT, define_roi_interactively

try:
    import pygame
    pygame.mixer.init()
    PYGAME_OK = True
except Exception:
    PYGAME_OK = False
    print("[WARN] pygame not found — audio alerts disabled.")


# ============================================================
#  USER SETTINGS
# ============================================================

VIDEO_SOURCE     = "railway_video1.mp4"
MODEL_PATH       = "yolov8n.pt"
CONF_THRESHOLD   = 0.40
IOU_THRESHOLD    = 0.45
ALERT_COOLDOWN   = 3
SAVE_ALERTS      = True
ALERT_DIR        = "alerts"
SOUND_FILE       = "alert.wav"

HIGH_RISK_TTC    = 5.0
MEDIUM_RISK_TTC  = 10.0

# ── Speed settings ──────────────────────────────────────────
# The real speed of this train. Used ONLY to derive the scale factor.
KNOWN_TRAIN_SPEED_KMH = 90.0

# How many frames to collect before locking the scale.
# During this time speed is shown live (not frozen).
AUTO_CALIB_FRAMES = 20

# EMA weight. 0.25 = responsive but smooth.
SPEED_EMA = 0.25

# Clamp absurd single-frame spikes
MAX_BELIEVABLE_SPEED_KMH = 101.0

# Min pixel movement to count as real motion
MIN_DISP_PX = 1.5

# ── Class IDs (COCO) ────────────────────────────────────────
CLS_PERSON     = 0
CLS_CAR        = 2
CLS_MOTORCYCLE = 3
CLS_BUS        = 5
CLS_TRAIN      = 6
CLS_TRUCK      = 7

VEHICLE_CLASSES  = {CLS_CAR, CLS_MOTORCYCLE, CLS_BUS, CLS_TRUCK}
OBSTACLE_CLASSES = {CLS_PERSON} | VEHICLE_CLASSES
ALL_MONITORED    = OBSTACLE_CLASSES | {CLS_TRAIN}


# ============================================================
#  SOUND / ALERT
# ============================================================

_last_alert_time = 0.0

def play_alert_sound():
    def _play():
        if PYGAME_OK and os.path.exists(SOUND_FILE):
            pygame.mixer.music.load(SOUND_FILE)
            pygame.mixer.music.play()
            time.sleep(2)
        else:
            try:
                import winsound
                winsound.Beep(1200, 600)
            except Exception:
                pass
    threading.Thread(target=_play, daemon=True).start()

def trigger_alert(frame, risk_level, reason):
    global _last_alert_time
    now = time.time()
    if now - _last_alert_time < ALERT_COOLDOWN:
        return
    _last_alert_time = now
    print(f"\n{'='*55}")
    print(f"  🚨  ALERT  |  {risk_level}  |  {reason}")
    print(f"  🕐  {time.strftime('%Y-%m-%d  %H:%M:%S')}")
    print(f"{'='*55}\n")
    play_alert_sound()
    if SAVE_ALERTS and frame is not None:
        os.makedirs(ALERT_DIR, exist_ok=True)
        ts   = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(ALERT_DIR, f"{risk_level.replace(' ','_')}_{ts}.jpg")
        cv2.imwrite(path, frame)


# ============================================================
#  ROI UTILITIES
# ============================================================

def make_polygon(points):
    return np.array(points, dtype=np.int32)

def inside_roi(point, polygon_np):
    return cv2.pointPolygonTest(
        polygon_np, (float(point[0]), float(point[1])), False) >= 0


# ============================================================
#  DRAWING
# ============================================================

COL_SAFE    = (  0, 220,   0)
COL_LOW     = (  0, 220, 220)
COL_MEDIUM  = (  0, 165, 255)
COL_HIGH    = (  0,   0, 255)
COL_TRAIN   = (255,  80,  80)
COL_PERSON  = (  0, 255, 255)
COL_VEHICLE = (255, 165,   0)
COL_WHITE   = (255, 255, 255)
COL_BLACK   = (  0,   0,   0)

RISK_COLORS = {
    "SAFE"        : COL_SAFE,
    "LOW RISK"    : COL_LOW,
    "MEDIUM RISK" : COL_MEDIUM,
    "HIGH RISK"   : COL_HIGH,
}

def get_object_color(class_name, in_zone):
    if in_zone:                return COL_HIGH
    if class_name == "train":  return COL_TRAIN
    if class_name == "person": return COL_PERSON
    return COL_VEHICLE

def draw_roi_zone(frame, polygon_np, label, danger=False):
    color   = COL_HIGH if danger else COL_SAFE
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon_np], color)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.polylines(frame, [polygon_np], True, color, 2)
    cv2.putText(frame, label,
                (polygon_np[0][0], polygon_np[0][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

def draw_box(frame, x1, y1, x2, y2, cx, cy, label, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, (cx, cy), 5, COL_WHITE, -1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_WHITE, 1)

def draw_hud(frame, risk_level, ttc, train_speed_kmh,
             train_direction, alert_flag, fps, frame_count, calib_locked):
    h, w = frame.shape[:2]
    risk_color = RISK_COLORS.get(risk_level, COL_WHITE)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), COL_BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"STATUS: {risk_level}", (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, risk_color, 3)
    cv2.putText(frame, f"FPS: {fps:.1f}   Frame: {frame_count}",
                (w - 280, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_WHITE, 2)

    lock_col = COL_SAFE if calib_locked else (0, 200, 255)
    lock_txt = "LOCKED" if calib_locked else "CALIBRATING"

    items = [
        (f"Train Speed  : {train_speed_kmh:>6.1f} km/h", COL_TRAIN),
        (f"TTC           : {ttc:>6.2f} s",
            COL_HIGH   if ttc < HIGH_RISK_TTC   else
            COL_MEDIUM if ttc < MEDIUM_RISK_TTC else COL_SAFE),
        (f"Direction     : {train_direction}", COL_WHITE),
        (f"Scale calib   : {lock_txt}", lock_col),
    ]
    y = 80
    for txt, col in items:
        cv2.putText(frame, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, col, 2)
        y += 28

    indicators = [
        ("PERSON/VEHICLE ON TRACK", alert_flag),
        ("TRAIN APPROACHING",       train_direction == "APPROACHING"),
    ]
    y = 80
    for lbl, active in indicators:
        cv2.putText(frame, ("● " + lbl), (w - 370, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    COL_HIGH if active else COL_SAFE, 2)
        y += 30

    if risk_level == "HIGH RISK":
        flash = frame.copy()
        cv2.rectangle(flash, (0, 0), (w, h), COL_HIGH, -1)
        cv2.addWeighted(flash, 0.10, frame, 0.90, 0, frame)
        cv2.putText(frame, "⚠ DANGER ⚠", (w // 2 - 140, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, COL_HIGH, 3)

    cv2.putText(frame,
                "[Q] Quit   [P] Pause   [R] Redraw ROI   [S] Reset calib",
                (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)


# ============================================================
#  TRAIN TRACKER
# ============================================================

class TrainTracker:
    """
    Speed estimation that is live from frame 1 and never freezes.

    How it works
    ------------
    Every frame we compute raw_px_per_sec = pixel_displacement / real_dt.

    We maintain a rolling window (deque) of the last WINDOW raw values.
    The reported speed at any frame is:

        speed_kmh = median(window) * scale

    where `scale` starts as a rough bootstrap guess and locks after
    AUTO_CALIB_FRAMES samples.

    Because we use the rolling-window MEDIAN (not EMA from a frozen
    seed), the number fluctuates naturally every frame — it reflects
    real pixel motion — and is never stuck at a constant value.

    Bootstrap (first AUTO_CALIB_FRAMES frames)
    -------------------------------------------
    We don't know the true scale yet, so we use a running estimate:
        scale_est = KNOWN_TRAIN_SPEED_KMH / median(calib_buf_so_far)
    This means from frame 2 onward the display shows a live, fluctuating
    speed that converges toward the true value rather than being frozen.

    After AUTO_CALIB_FRAMES frames the scale is locked permanently.
    """

    WINDOW = 8   # rolling median window size

    def __init__(self, frame_h):
        self.frame_h = frame_h
        self.states  = {}

    def _new_state(self):
        return {
            "prev_cx"    : None,
            "prev_cy"    : None,
            "prev_t"     : None,
            "calib_buf"  : [],          # raw px/s during bootstrap
            "scale"      : None,        # None = not locked yet
            "window"     : deque(maxlen=self.WINDOW),  # recent raw px/s
            "speed_kmh"  : 0.0,
            "prev_dist"  : None,
            "direction"  : "UNKNOWN",
            "ttc"        : 999.0,
        }

    def reset_calibration(self):
        for s in self.states.values():
            s["calib_buf"] = []
            s["scale"]     = None
            s["window"].clear()
            s["speed_kmh"] = 0.0

    def update(self, track_id, cx, cy, crossing_y, t_now):
        if track_id not in self.states:
            self.states[track_id] = self._new_state()
        s = self.states[track_id]

        if s["prev_cx"] is not None:
            dx   = cx - s["prev_cx"]
            dy   = cy - s["prev_cy"]
            disp = np.sqrt(dx * dx + dy * dy)
            dt   = max(t_now - s["prev_t"], 1e-4)

            # Skip if track was lost and re-acquired (huge dt = warp jump)
            if dt > 2.0:
                s["prev_cx"] = cx
                s["prev_cy"] = cy
                s["prev_t"]  = t_now
                return s

            if disp >= MIN_DISP_PX:
                raw_pps = disp / dt   # raw pixels per second

                # Clamp impossible single-frame spikes before they
                # enter either the window or the calib buffer
                # (use a loose pixel-space limit: >5000 px/s at 30fps
                #  = >166px/frame which is physically impossible)
                if raw_pps < 5000:
                    s["window"].append(raw_pps)
                    s["calib_buf"].append(raw_pps)

                # ── Compute current scale estimate ────────────
                if s["scale"] is None:
                    # Bootstrap: lock after enough samples
                    if len(s["calib_buf"]) >= AUTO_CALIB_FRAMES:
                        median_raw = float(np.median(s["calib_buf"]))
                        if median_raw > 0.1:
                            s["scale"] = KNOWN_TRAIN_SPEED_KMH / median_raw
                            print(f"[CALIB] Track {track_id} locked: "
                                  f"scale={s['scale']:.5f}")
                    # Running scale estimate from whatever we have so far
                    # — this is what makes it live during bootstrap
                    if s["calib_buf"]:
                        scale_now = (KNOWN_TRAIN_SPEED_KMH
                                     / float(np.median(s["calib_buf"])))
                    else:
                        scale_now = None
                else:
                    scale_now = s["scale"]

                # ── Speed from rolling median × scale ─────────
                if scale_now and s["window"]:
                    median_pps    = float(np.median(s["window"]))
                    raw_kmh       = median_pps * scale_now
                    if raw_kmh <= MAX_BELIEVABLE_SPEED_KMH:
                        # Light EMA just to remove last-pixel jitter
                        s["speed_kmh"] = (SPEED_EMA * raw_kmh
                                          + (1 - SPEED_EMA) * s["speed_kmh"])

        s["prev_cx"] = cx
        s["prev_cy"] = cy
        s["prev_t"]  = t_now

        # ── Direction ─────────────────────────────────────────
        dist_px = abs(cy - crossing_y)
        if s["prev_dist"] is not None:
            if dist_px < s["prev_dist"] - 1:
                s["direction"] = "APPROACHING"
            elif dist_px > s["prev_dist"] + 1:
                s["direction"] = "MOVING AWAY"
        s["prev_dist"] = dist_px

        # ── TTC ───────────────────────────────────────────────
        scale = s["scale"] or (
            KNOWN_TRAIN_SPEED_KMH / float(np.median(s["calib_buf"]))
            if s["calib_buf"] else None
        )
        if scale and s["speed_kmh"] > 0.5:
            px_per_sec = s["speed_kmh"] / scale
            s["ttc"]   = dist_px / px_per_sec if px_per_sec > 0 else 999.0
        else:
            s["ttc"] = 999.0

        return s

    def most_dangerous(self):
        if not self.states:
            return None
        return min(self.states.values(), key=lambda s: s["ttc"])

    def any_locked(self):
        return any(s["scale"] is not None for s in self.states.values())

    def cleanup(self, active_ids):
        for tid in list(self.states.keys()):
            if tid not in active_ids:
                del self.states[tid]


# ============================================================
#  RISK ASSESSOR
# ============================================================

def assess_risk(alert_flag, train_state):
    if train_state is None:
        return ("LOW RISK", "Obstacle on track, no train detected") if alert_flag \
               else ("SAFE", "No hazards detected")

    direction   = train_state["direction"]
    ttc         = train_state["ttc"]
    approaching = (direction == "APPROACHING")

    if alert_flag and approaching and ttc < HIGH_RISK_TTC:
        return "HIGH RISK",   f"Obstacle on track + Train approaching in {ttc:.1f}s"
    if approaching and ttc < HIGH_RISK_TTC:
        return "HIGH RISK",   f"Train approaching fast — TTC {ttc:.1f}s"
    if alert_flag and approaching and ttc < MEDIUM_RISK_TTC:
        return "MEDIUM RISK", f"Obstacle on track + Train approaching in {ttc:.1f}s"
    if approaching and ttc < MEDIUM_RISK_TTC:
        return "MEDIUM RISK", f"Train approaching — TTC {ttc:.1f}s"
    if alert_flag:
        return "LOW RISK",    "Obstacle on track"
    return "SAFE", "All clear"


# ============================================================
#  ROI REDRAW
# ============================================================

def redraw_roi(cap, frame_w, frame_h):
    saved_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, saved_pos - 1))
    ret, snap = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)
    if not ret:
        return None, None

    WIN   = "Redraw TRACK ROI — Click corners, ENTER to confirm | ESC to cancel"
    state = {"points": [], "snap": snap}

    def _cb(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        state["points"].append((x, y))
        img = state["snap"].copy()
        for pt in state["points"]:
            cv2.circle(img, pt, 7, (0, 255, 0), -1)
        if len(state["points"]) > 1:
            cv2.polylines(img, [np.array(state["points"], np.int32)],
                          False, (0, 255, 0), 2)
        n = len(state["points"])
        cv2.putText(img,
                    f"Points: {n}  — ENTER to confirm" if n >= 3
                    else f"Points: {n}  (need >=3)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(WIN, img)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    display = snap.copy()
    cv2.putText(display, "Click >=3 corners — ENTER to confirm | ESC to cancel",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.imshow(WIN, display)
    cv2.waitKey(1)
    cv2.setMouseCallback(WIN, _cb)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 13 and len(state["points"]) >= 3:
            break
        if key == 27:
            state["points"].clear()
            break

    cv2.destroyWindow(WIN)
    cv2.waitKey(1)

    pts = state["points"]
    if len(pts) >= 3:
        return pts, make_polygon(pts)
    return None, None


# ============================================================
#  MAIN LOOP
# ============================================================

def main():
    os.makedirs(ALERT_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)
    cap   = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {VIDEO_SOURCE}")
        return

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ret, first_frame = cap.read()
    if not ret:
        return

    frame_h, frame_w = first_frame.shape[:2]
    crossing_y       = int(CROSSING_Y_PERCENT * frame_h)

    current_roi_pts = list(TRACK_ROI)
    track_poly      = make_polygon(current_roi_pts)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    train_tracker = TrainTracker(frame_h)
    paused        = False
    frame_count   = 0
    frame         = first_frame.copy()
    prev_time     = time.time()

    WIN = "🚂 Railway Accident Prevention System"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  break
        if key == ord('p'):  paused = not paused
        if key == ord('s'):
            train_tracker.reset_calibration()
            print("[INFO] Speed calibration reset.")
        if key == ord('r'):
            new_pts, new_poly = redraw_roi(cap, frame_w, frame_h)
            if new_pts:
                current_roi_pts = new_pts
                track_poly      = new_poly
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

        if paused:
            cv2.imshow(WIN, frame)
            continue

        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        results = model.track(
            frame, persist=True,
            conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False
        )

        alert_flag       = False
        active_train_ids = set()

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in ALL_MONITORED or box.id is None:
                    continue

                track_id        = int(box.id[0])
                conf            = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy          = (x1 + x2) // 2, (y1 + y2) // 2
                class_name      = model.names[cls_id]
                in_track        = inside_roi((cx, cy), track_poly)

                if cls_id == CLS_TRAIN:
                    active_train_ids.add(track_id)
                    train_tracker.update(track_id, cx, cy, crossing_y, now)
                    spd   = train_tracker.states[track_id]["speed_kmh"]
                    label = f"TRAIN ID:{track_id}  {spd:.1f} km/h"
                    draw_box(frame, x1, y1, x2, y2, cx, cy, label, COL_TRAIN)
                    continue

                if cls_id in OBSTACLE_CLASSES and in_track:
                    alert_flag = True

                color = get_object_color(class_name, in_track)
                label = f"{class_name} ID:{track_id} {conf:.2f}"
                draw_box(frame, x1, y1, x2, y2, cx, cy, label, color)

        train_tracker.cleanup(active_train_ids)

        danger_train    = train_tracker.most_dangerous()
        ttc             = danger_train["ttc"]       if danger_train else 999.0
        train_speed_kmh = danger_train["speed_kmh"] if danger_train else 0.0
        train_direction = danger_train["direction"] if danger_train else "UNKNOWN"
        calib_locked    = train_tracker.any_locked()

        risk_level, reason = assess_risk(alert_flag, danger_train)
        if risk_level in ("HIGH RISK", "MEDIUM RISK"):
            trigger_alert(frame, risk_level, reason)

        draw_roi_zone(frame, track_poly, "TRACK ZONE", danger=alert_flag)
        draw_hud(frame, risk_level, ttc, train_speed_kmh,
                 train_direction, alert_flag, fps, frame_count, calib_locked)

        cv2.imshow(WIN, frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()