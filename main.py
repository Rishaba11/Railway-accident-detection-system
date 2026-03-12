# ============================================================
#  main.py  –  Railway Accident Prevention System (Modified)
# ============================================================

import cv2
import numpy as np
import time
import os
import threading

# ── YOLOv8 ──────────────────────────────────────────────────
from ultralytics import YOLO

# ── ROI config ──────────────────────────────────────────────
# Removed GATE_ROI
from roi_config import TRACK_ROI, CROSSING_Y_PERCENT

# ── Optional sound (pygame) ─────────────────────────────────
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

VIDEO_SOURCE      = "railway_video.mp4" 
MODEL_PATH        = "yolov8n.pt"         
CONF_THRESHOLD    = 0.40
IOU_THRESHOLD     = 0.45
PIXELS_PER_METER  = 10                    
ALERT_COOLDOWN    = 3                     
SPEED_ALPHA       = 0.25                  
SAVE_ALERTS       = True
ALERT_DIR         = "alerts"
SOUND_FILE        = "alert.wav"           

HIGH_RISK_TTC    = 5.0
MEDIUM_RISK_TTC  = 10.0

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
#  HELPER: SOUND ALERT
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
#  HELPER: ROI UTILITIES
# ============================================================

def make_polygon(points):
    return np.array(points, dtype=np.int32)

def inside_roi(point, polygon_np):
    return cv2.pointPolygonTest(polygon_np, (float(point[0]), float(point[1])), False) >= 0


# ============================================================
#  HELPER: DRAWING
# ============================================================

COL_SAFE     = (0,  220,  0)
COL_LOW      = (0,  220, 220)
COL_MEDIUM   = (0,  165, 255)
COL_HIGH     = (0,   0,  255)
COL_TRAIN    = (255, 80,  80)
COL_PERSON   = (0,  255, 255)
COL_VEHICLE  = (255,165,   0)
COL_WHITE    = (255,255, 255)
COL_BLACK    = (  0,   0,   0)

RISK_COLORS = {
    "SAFE"         : COL_SAFE,
    "LOW RISK"     : COL_LOW,
    "MEDIUM RISK" : COL_MEDIUM,
    "HIGH RISK"   : COL_HIGH,
}

def get_object_color(class_name, in_zone):
    if in_zone:
        return COL_HIGH
    if class_name == "train":
        return COL_TRAIN
    if class_name == "person":
        return COL_PERSON
    return COL_VEHICLE


def draw_roi_zone(frame, polygon_np, label, danger=False):
    color   = COL_HIGH if danger else COL_SAFE
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon_np], color)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.polylines(frame, [polygon_np], True, color, 2)
    cv2.putText(frame, label, (polygon_np[0][0], polygon_np[0][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_box(frame, x1, y1, x2, y2, cx, cy, label, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, (cx, cy), 5, COL_WHITE, -1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_WHITE, 1)


def draw_hud(frame, risk_level, ttc, train_speed_kmh,
             train_direction, alert_flag, 
             fps, frame_count):
    h, w = frame.shape[:2]
    risk_color = RISK_COLORS.get(risk_level, COL_WHITE)

    bar_h = 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), COL_BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"STATUS: {risk_level}", (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, risk_color, 3)

    fps_txt = f"FPS: {fps:.1f}   Frame: {frame_count}"
    cv2.putText(frame, fps_txt, (w - 280, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_WHITE, 2)

    panel_x = 12
    items = [
        (f"Train Speed : {train_speed_kmh:>6.1f} km/h", COL_TRAIN),
        (f"TTC          : {ttc:>6.2f} s",
            COL_HIGH if ttc < HIGH_RISK_TTC else
            COL_MEDIUM if ttc < MEDIUM_RISK_TTC else COL_SAFE),
        (f"Direction    : {train_direction}", COL_WHITE),
    ]
    y = 90
    for txt, col in items:
        cv2.putText(frame, txt, (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
        y += 32

    indicators = [
        ("PERSON/VEHICLE ON TRACK", alert_flag),
        ("TRAIN APPROACHING",       train_direction == "APPROACHING"),
    ]
    y = 90
    for label, active in indicators:
        dot_col = COL_HIGH if active else COL_SAFE
        dot_txt = "●"
        cv2.putText(frame, dot_txt + " " + label,
                    (w - 360, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, dot_col, 2)
        y += 30

    if risk_level == "HIGH RISK":
        flash = frame.copy()
        cv2.rectangle(flash, (0, 0), (w, h), COL_HIGH, -1)
        cv2.addWeighted(flash, 0.10, frame, 0.90, 0, frame)
        cv2.putText(frame, "⚠ DANGER ⚠", (w // 2 - 140, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, COL_HIGH, 3)


# ============================================================
#  TRAIN STATE TRACKER
# ============================================================

class TrainTracker:
    def __init__(self):
        self.states = {}

    def update(self, track_id, cx, cy, crossing_y, fps):
        if track_id not in self.states:
            self.states[track_id] = {
                "prev_cx"        : None,
                "prev_cy"        : None,
                "prev_dist"      : None,
                "speed_px_s"     : 0.0,
                "speed_kmh"      : 0.0,
                "direction"      : "UNKNOWN",
                "ttc"            : 999.0,
            }

        s = self.states[track_id]

        if s["prev_cx"] is not None:
            dx  = cx - s["prev_cx"]
            dy  = cy - s["prev_cy"]
            raw_px_s = np.sqrt(dx*dx + dy*dy) * fps
            s["speed_px_s"] = (SPEED_ALPHA * raw_px_s
                               + (1 - SPEED_ALPHA) * s["speed_px_s"])
            speed_mps        = s["speed_px_s"] / PIXELS_PER_METER
            s["speed_kmh"]   = speed_mps * 3.6

        s["prev_cx"] = cx
        s["prev_cy"] = cy
        dist_px = abs(cy - crossing_y)

        if s["prev_dist"] is not None:
            if dist_px < s["prev_dist"] - 1:
                s["direction"] = "APPROACHING"
            elif dist_px > s["prev_dist"] + 1:
                s["direction"] = "MOVING AWAY"

        s["prev_dist"] = dist_px

        if s["speed_px_s"] > 0.5:
            s["ttc"] = dist_px / s["speed_px_s"]
        else:
            s["ttc"] = 999.0

        return s

    def most_dangerous(self):
        if not self.states:
            return None
        return min(self.states.values(), key=lambda s: s["ttc"])

    def cleanup(self, active_ids):
        for tid in list(self.states.keys()):
            if tid not in active_ids:
                del self.states[tid]


# ============================================================
#  RISK ASSESSOR
# ============================================================

def assess_risk(alert_flag, train_state):
    """ Simplified: Removed gate_danger check """
    if train_state is None:
        if alert_flag:
            return "LOW RISK", "Obstacle on track, no train detected"
        return "SAFE", "No hazards detected"

    direction = train_state["direction"]
    ttc       = train_state["ttc"]
    approaching = (direction == "APPROACHING")

    if alert_flag and approaching and ttc < HIGH_RISK_TTC:
        return "HIGH RISK", f"Obstacle on track + Train approaching in {ttc:.1f}s"

    if approaching and ttc < HIGH_RISK_TTC:
        return "HIGH RISK", f"Train approaching fast — TTC {ttc:.1f}s"

    if alert_flag and approaching and ttc < MEDIUM_RISK_TTC:
        return "MEDIUM RISK", f"Obstacle on track + Train approaching in {ttc:.1f}s"

    if approaching and ttc < MEDIUM_RISK_TTC:
        return "MEDIUM RISK", f"Train approaching — TTC {ttc:.1f}s"

    if alert_flag:
        return "LOW RISK", "Obstacle on track"

    return "SAFE", "All clear"


# ============================================================
#  MAIN LOOP
# ============================================================

def main():
    os.makedirs(ALERT_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        return

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ret, first_frame = cap.read()
    if not ret: return

    frame_h, frame_w = first_frame.shape[:2]
    crossing_y = int(CROSSING_Y_PERCENT * frame_h)
    track_poly = make_polygon(TRACK_ROI)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    train_tracker = TrainTracker()

    paused      = False
    frame_count = 0
    prev_time   = time.time()

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('p'): paused = not paused
        
        if paused:
            cv2.imshow("🚂 Railway Accident Prevention System", frame if 'frame' in dir() else first_frame)
            continue

        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        results = model.track(frame, persist=True, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

        alert_flag   = False
        active_train_ids = set()

        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in ALL_MONITORED or box.id is None: continue

                track_id   = int(box.id[0])
                conf       = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                class_name = model.names[cls_id]

                in_track = inside_roi((cx, cy), track_poly)

                if cls_id == CLS_TRAIN:
                    active_train_ids.add(track_id)
                    train_tracker.update(track_id, cx, cy, crossing_y, actual_fps)
                    label = f"TRAIN ID:{track_id} v={train_tracker.states[track_id]['speed_kmh']:.1f}km/h"
                    draw_box(frame, x1, y1, x2, y2, cx, cy, label, COL_TRAIN)
                    continue

                if cls_id in OBSTACLE_CLASSES:
                    if in_track:
                        alert_flag = True

                    color = get_object_color(class_name, in_track)
                    label = f"{class_name} ID:{track_id} {conf:.2f}"
                    draw_box(frame, x1, y1, x2, y2, cx, cy, label, color)

        train_tracker.cleanup(active_train_ids)
        danger_train = train_tracker.most_dangerous()

        ttc             = danger_train["ttc"]       if danger_train else 999.0
        train_speed_kmh = danger_train["speed_kmh"] if danger_train else 0.0
        train_direction = danger_train["direction"] if danger_train else "UNKNOWN"

        risk_level, reason = assess_risk(alert_flag, danger_train)

        if risk_level in ("HIGH RISK", "MEDIUM RISK"):
            trigger_alert(frame, risk_level, reason)

        draw_roi_zone(frame, track_poly, "TRACK ZONE", danger=alert_flag)

        #cv2.line(frame, (0, crossing_y), (frame_w, crossing_y), (255, 255, 0), 1)
        
        draw_hud(frame, risk_level, ttc, train_speed_kmh, train_direction, alert_flag, fps, frame_count)

        cv2.imshow("🚂 Railway Accident Prevention System", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()