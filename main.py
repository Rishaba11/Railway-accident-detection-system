import os
import threading
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

from roi_config import (
    CROSSING_Y_PERCENT,
    TRACK_ROI,
    edit_crossing_line_on_frame,
    edit_roi_on_frame,
)

try:
    import pygame

    pygame.mixer.init()
    PYGAME_OK = True
except Exception:
    PYGAME_OK = False
    print("[WARN] pygame not found - audio alerts disabled.")


# ============================================================
#  USER SETTINGS
# ============================================================

VIDEO_SOURCE = "railway_video1.mp4"
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45
ALERT_COOLDOWN = 3
SAVE_ALERTS = True
ALERT_DIR = "alerts"
SOUND_FILE = "alert.wav"
ENABLE_SOUND = True
TRAIN_ONLY_VIEW = False
SESSION_LABEL = "Default session"

HIGH_RISK_TTC = 5.0
MEDIUM_RISK_TTC = 10.0

KNOWN_TRAIN_SPEED_KMH = 90.0
AUTO_CALIB_FRAMES = 20
SPEED_EMA = 0.25
MAX_BELIEVABLE_SPEED_KMH = 101.0
MIN_DISP_PX = 1.5

CLS_PERSON = 0
CLS_CAR = 2
CLS_MOTORCYCLE = 3
CLS_BUS = 5
CLS_TRAIN = 6
CLS_TRUCK = 7

VEHICLE_CLASSES = {CLS_CAR, CLS_MOTORCYCLE, CLS_BUS, CLS_TRUCK}
OBSTACLE_CLASSES = {CLS_PERSON} | VEHICLE_CLASSES
ALL_MONITORED = OBSTACLE_CLASSES | {CLS_TRAIN}


# ============================================================
#  SOUND / ALERT
# ============================================================

_last_alert_time = 0.0


def play_alert_sound():
    if not ENABLE_SOUND:
        return

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
    print(f"\n{'=' * 55}")
    print(f"  ALERT  |  {risk_level}  |  {reason}")
    print(f"  TIME   |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 55}\n")

    play_alert_sound()

    if SAVE_ALERTS and frame is not None:
        os.makedirs(ALERT_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(ALERT_DIR, f"{risk_level.replace(' ', '_')}_{ts}.jpg")
        cv2.imwrite(path, frame)


# ============================================================
#  ROI UTILITIES
# ============================================================


def make_polygon(points):
    return np.array(points, dtype=np.int32)


def inside_roi(point, polygon_np):
    return cv2.pointPolygonTest(
        polygon_np,
        (float(point[0]), float(point[1])),
        False,
    ) >= 0


def wrap_text(text, max_chars=40):
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


# ============================================================
#  DRAWING
# ============================================================

COL_SAFE = (86, 209, 120)
COL_LOW = (95, 208, 226)
COL_MEDIUM = (70, 180, 255)
COL_HIGH = (83, 107, 255)
COL_TRAIN = (95, 120, 255)
COL_PERSON = (114, 232, 255)
COL_VEHICLE = (123, 198, 242)
COL_WHITE = (255, 255, 255)
COL_BLACK = (0, 0, 0)
COL_PANEL = (18, 34, 52)
COL_PANEL_ALT = (28, 48, 71)
COL_PANEL_BORDER = (91, 118, 148)
COL_TEXT_SOFT = (197, 214, 232)
COL_TEXT_MUTED = (150, 169, 190)
COL_TEAL = (215, 227, 106)
COL_GOLD = (123, 198, 242)

RISK_COLORS = {
    "SAFE": COL_SAFE,
    "LOW RISK": COL_LOW,
    "MEDIUM RISK": COL_MEDIUM,
    "HIGH RISK": COL_HIGH,
}


def get_object_color(class_name, in_zone):
    if in_zone:
        return COL_HIGH
    if class_name == "train":
        return COL_TRAIN
    if class_name == "person":
        return COL_PERSON
    return COL_VEHICLE


def apply_panel(frame, x1, y1, x2, y2, fill=COL_PANEL, alpha=0.72, border=COL_PANEL_BORDER):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1)


def draw_badge(frame, x, y, text, color, fill=COL_PANEL_ALT):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.55, 2)
    x2 = x + tw + 24
    y2 = y + 30
    apply_panel(frame, x, y, x2, y2, fill=fill, alpha=0.88, border=color)
    cv2.putText(frame, text, (x + 12, y + 20), font, 0.55, color, 2)
    return x2


def draw_metric(frame, x, y, label, value, value_color=COL_WHITE):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, label, (x, y), font, 0.48, COL_TEXT_MUTED, 1)
    cv2.putText(frame, value, (x + 138, y), font, 0.58, value_color, 2)


def draw_indicator(frame, x, y, label, active, active_color):
    color = active_color if active else COL_TEXT_MUTED
    cv2.circle(frame, (x, y - 5), 6, color, -1)
    cv2.putText(frame, label, (x + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2)


def format_ttc(ttc):
    if ttc >= 998:
        return "No train"
    return f"{ttc:.2f} s"


def draw_crossing_line(frame, crossing_y):
    height, width = frame.shape[:2]
    label_y = max(24, min(height - 12, crossing_y - 10))
    cv2.line(frame, (0, crossing_y), (width, crossing_y), COL_GOLD, 2)
    cv2.putText(
        frame,
        "CROSSING LINE",
        (16, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        COL_GOLD,
        2,
    )


def draw_roi_zone(frame, polygon_np, label, danger=False):
    color = COL_HIGH if danger else COL_SAFE
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon_np], color)
    cv2.addWeighted(overlay, 0.14, frame, 0.86, 0, frame)
    cv2.polylines(frame, [polygon_np], True, color, 2)

    min_x = int(np.min(polygon_np[:, 0]))
    min_y = int(np.min(polygon_np[:, 1]))
    cv2.putText(
        frame,
        label,
        (min_x + 6, max(24, min_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def draw_box(frame, x1, y1, x2, y2, cx, cy, label, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, (cx, cy), 5, COL_WHITE, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.52, 1)
    label_bottom = max(y1, th + 12)
    cv2.rectangle(frame, (x1, label_bottom - th - 10), (x1 + tw + 10, label_bottom), color, -1)
    cv2.putText(frame, label, (x1 + 5, label_bottom - 5), font, 0.52, COL_WHITE, 1)


def draw_hud(
    frame,
    risk_level,
    reason,
    ttc,
    train_speed_kmh,
    train_direction,
    alert_flag,
    fps,
    frame_count,
    calib_locked,
    crossing_y,
    source_label,
    paused=False,
):
    height, width = frame.shape[:2]
    risk_color = RISK_COLORS.get(risk_level, COL_WHITE)

    if risk_level == "HIGH RISK":
        flash = frame.copy()
        cv2.rectangle(flash, (0, 0), (width, height), COL_HIGH, -1)
        cv2.addWeighted(flash, 0.08, frame, 0.92, 0, frame)

    apply_panel(frame, 16, 16, width - 16, 88, alpha=0.78)
    cv2.putText(frame, "RAPS MONITOR", (30, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.95, COL_WHITE, 2)
    cv2.putText(frame, source_label, (30, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.56, COL_TEXT_SOFT, 1)

    fps_text = f"FPS {fps:.1f} | Frame {frame_count}"
    fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)[0]
    cv2.putText(
        frame,
        fps_text,
        (width - fps_size[0] - 28, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        COL_WHITE,
        2,
    )

    badge_right = draw_badge(frame, 30, 96, risk_level, risk_color)
    if train_direction == "APPROACHING":
        badge_right = draw_badge(frame, badge_right + 10, 96, "STOP CROSSING", COL_GOLD)
        badge_right = draw_badge(frame, badge_right + 10, 96, "CLOSE DOORS", COL_GOLD)
    if paused:
        draw_badge(frame, badge_right + 10, 96, "PAUSED", COL_GOLD)

    apply_panel(frame, 16, 136, 386, 360, alpha=0.78)
    cv2.putText(frame, "Risk Summary", (30, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_WHITE, 2)

    reason_y = 194
    for line in wrap_text(reason, max_chars=34)[:3]:
        cv2.putText(frame, line, (30, reason_y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, COL_TEXT_SOFT, 1)
        reason_y += 24

    ttc_color = (
        COL_HIGH
        if ttc < HIGH_RISK_TTC
        else COL_MEDIUM
        if ttc < MEDIUM_RISK_TTC
        else COL_SAFE
    )
    lock_color = COL_SAFE if calib_locked else COL_GOLD

    draw_metric(frame, 30, 256, "Train speed", f"{train_speed_kmh:5.1f} km/h", COL_TRAIN)
    draw_metric(frame, 30, 284, "Time to cross", format_ttc(ttc), ttc_color)
    draw_metric(frame, 30, 312, "Direction", train_direction, COL_WHITE)
    draw_metric(frame, 30, 340, "Calibration", "Locked" if calib_locked else "Learning", lock_color)

    apply_panel(frame, width - 350, 136, width - 16, 360, alpha=0.78)
    cv2.putText(frame, "Live Signals", (width - 336, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_WHITE, 2)
    draw_indicator(frame, width - 334, 206, "Track occupied", alert_flag, COL_HIGH)
    draw_indicator(
        frame,
        width - 334,
        236,
        "Train approaching",
        train_direction == "APPROACHING",
        COL_MEDIUM,
    )
    draw_indicator(frame, width - 334, 266, "Snapshot capture", SAVE_ALERTS, COL_GOLD)
    draw_indicator(frame, width - 334, 296, "Audio alert", ENABLE_SOUND, COL_TEAL)

    crossing_text = f"Crossing line: {crossing_y / float(height):.0%} of frame height"
    cv2.putText(
        frame,
        crossing_text,
        (width - 336, 348),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        COL_TEXT_SOFT,
        1,
    )

    apply_panel(frame, 16, height - 54, width - 16, height - 16, alpha=0.80)
    help_text = "[Q] quit   [P] pause   [R] redraw ROI   [C] crossing line   [S] reset calibration   [V] train-only view"
    cv2.putText(
        frame,
        help_text,
        (30, height - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        COL_TEXT_SOFT,
        1,
    )

    if risk_level == "HIGH RISK":
        danger_text = "DANGER"
        (tw, _), _ = cv2.getTextSize(danger_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.putText(
            frame,
            danger_text,
            (width // 2 - tw // 2, height - 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            COL_HIGH,
            3,
        )


# ============================================================
#  TRAIN TRACKER
# ============================================================


class TrainTracker:
    WINDOW = 8

    def __init__(self, frame_h):
        self.frame_h = frame_h
        self.states = {}

    def _new_state(self):
        return {
            "prev_cx": None,
            "prev_cy": None,
            "prev_t": None,
            "calib_buf": [],
            "scale": None,
            "window": deque(maxlen=self.WINDOW),
            "speed_kmh": 0.0,
            "prev_dist": None,
            "direction": "UNKNOWN",
            "ttc": 999.0,
        }

    def reset_calibration(self):
        for state in self.states.values():
            state["calib_buf"] = []
            state["scale"] = None
            state["window"].clear()
            state["speed_kmh"] = 0.0

    def update(self, track_id, cx, cy, crossing_y, t_now):
        if track_id not in self.states:
            self.states[track_id] = self._new_state()
        state = self.states[track_id]

        if state["prev_cx"] is not None:
            dx = cx - state["prev_cx"]
            dy = cy - state["prev_cy"]
            disp = np.sqrt(dx * dx + dy * dy)
            dt = max(t_now - state["prev_t"], 1e-4)

            if dt > 2.0:
                state["prev_cx"] = cx
                state["prev_cy"] = cy
                state["prev_t"] = t_now
                return state

            if disp >= MIN_DISP_PX:
                raw_pps = disp / dt

                if raw_pps < 5000:
                    state["window"].append(raw_pps)
                    state["calib_buf"].append(raw_pps)

                if state["scale"] is None:
                    if len(state["calib_buf"]) >= AUTO_CALIB_FRAMES:
                        median_raw = float(np.median(state["calib_buf"]))
                        if median_raw > 0.1:
                            state["scale"] = KNOWN_TRAIN_SPEED_KMH / median_raw
                            print(f"[CALIB] Track {track_id} locked: scale={state['scale']:.5f}")

                    if state["calib_buf"]:
                        scale_now = KNOWN_TRAIN_SPEED_KMH / float(np.median(state["calib_buf"]))
                    else:
                        scale_now = None
                else:
                    scale_now = state["scale"]

                if scale_now and state["window"]:
                    median_pps = float(np.median(state["window"]))
                    raw_kmh = median_pps * scale_now
                    if raw_kmh <= MAX_BELIEVABLE_SPEED_KMH:
                        state["speed_kmh"] = (
                            SPEED_EMA * raw_kmh + (1 - SPEED_EMA) * state["speed_kmh"]
                        )

        state["prev_cx"] = cx
        state["prev_cy"] = cy
        state["prev_t"] = t_now

        dist_px = abs(cy - crossing_y)
        if state["prev_dist"] is not None:
            if dist_px < state["prev_dist"] - 1:
                state["direction"] = "APPROACHING"
            elif dist_px > state["prev_dist"] + 1:
                state["direction"] = "MOVING AWAY"
        state["prev_dist"] = dist_px

        scale = state["scale"] or (
            KNOWN_TRAIN_SPEED_KMH / float(np.median(state["calib_buf"]))
            if state["calib_buf"]
            else None
        )
        if scale and state["speed_kmh"] > 0.5:
            px_per_sec = state["speed_kmh"] / scale
            state["ttc"] = dist_px / px_per_sec if px_per_sec > 0 else 999.0
        else:
            state["ttc"] = 999.0

        return state

    def most_dangerous(self):
        if not self.states:
            return None
        return min(self.states.values(), key=lambda state: state["ttc"])

    def any_locked(self):
        return any(state["scale"] is not None for state in self.states.values())

    def cleanup(self, active_ids):
        for track_id in list(self.states.keys()):
            if track_id not in active_ids:
                del self.states[track_id]


# ============================================================
#  RISK ASSESSMENT
# ============================================================


def assess_risk(alert_flag, train_state):
    approach_instruction = "STOP CROSSING / CLOSE DOORS"

    if train_state is None:
        if alert_flag:
            return "LOW RISK", "Obstacle on track, no train detected."
        return "SAFE", "No hazards detected."

    direction = train_state["direction"]
    ttc = train_state["ttc"]
    approaching = direction == "APPROACHING"

    if alert_flag and approaching and ttc < HIGH_RISK_TTC:
        return "HIGH RISK", f"{approach_instruction} - obstacle on track, train arriving in {ttc:.1f}s."
    if approaching and ttc < HIGH_RISK_TTC:
        return "HIGH RISK", f"{approach_instruction} - train arriving in {ttc:.1f}s."
    if alert_flag and approaching and ttc < MEDIUM_RISK_TTC:
        return "MEDIUM RISK", f"{approach_instruction} - obstacle on track, train arriving in {ttc:.1f}s."
    if approaching and ttc < MEDIUM_RISK_TTC:
        return "MEDIUM RISK", f"{approach_instruction} - train arriving in {ttc:.1f}s."
    if approaching:
        return "LOW RISK", f"Train detected. {approach_instruction}."
    if alert_flag:
        return "LOW RISK", "Obstacle detected inside the protected track zone."
    return "SAFE", "All clear."


# ============================================================
#  INTERACTIVE GEOMETRY TOOLS
# ============================================================


def render_geometry_editor(frame, title, footer, points=None, guide_points=None, crossing_y=None):
    display = frame.copy()

    if guide_points and len(guide_points) >= 3:
        guide_np = make_polygon(guide_points)
        guide_overlay = display.copy()
        cv2.fillPoly(guide_overlay, [guide_np], COL_PANEL_ALT)
        cv2.addWeighted(guide_overlay, 0.12, display, 0.88, 0, display)
        cv2.polylines(display, [guide_np], True, COL_TEXT_MUTED, 1)

    if points:
        pts_np = make_polygon(points)
        if len(points) >= 3:
            active_overlay = display.copy()
            cv2.fillPoly(active_overlay, [pts_np], COL_SAFE)
            cv2.addWeighted(active_overlay, 0.18, display, 0.82, 0, display)
            cv2.polylines(display, [pts_np], True, COL_TEAL, 2)
        elif len(points) > 1:
            cv2.polylines(display, [pts_np], False, COL_TEAL, 2)
        for point in points:
            cv2.circle(display, point, 5, COL_GOLD, -1)

    if crossing_y is not None:
        draw_crossing_line(display, crossing_y)

    height, width = display.shape[:2]
    apply_panel(display, 14, 14, width - 14, 92, alpha=0.82)
    cv2.putText(display, title, (28, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.82, COL_WHITE, 2)
    cv2.putText(display, footer, (28, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.54, COL_TEXT_SOFT, 1)
    return display


def redraw_roi(reference_frame, current_roi_pts, crossing_y):
    points = edit_roi_on_frame(
        reference_frame,
        roi_name="TRACK",
        initial_points=current_roi_pts,
        crossing_y_percent=crossing_y / float(reference_frame.shape[0]),
    )
    if points and len(points) >= 3:
        return points, make_polygon(points)
    return None, None


def redraw_crossing_line(reference_frame, current_crossing_y, current_roi_pts):
    crossing_percent = edit_crossing_line_on_frame(
        reference_frame,
        roi_points=current_roi_pts,
        initial_percent=current_crossing_y / float(reference_frame.shape[0]),
    )
    if crossing_percent is None:
        return None
    return int(round(crossing_percent * reference_frame.shape[0]))


# ============================================================
#  MAIN LOOP
# ============================================================


def main():
    os.makedirs(ALERT_DIR, exist_ok=True)

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {VIDEO_SOURCE}")
        return

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        print("[ERROR] Could not read the first frame.")
        return

    frame_h, _frame_w = first_frame.shape[:2]
    crossing_y = int(CROSSING_Y_PERCENT * frame_h)
    current_roi_pts = list(TRACK_ROI)
    track_poly = make_polygon(current_roi_pts)
    if not isinstance(VIDEO_SOURCE, int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    train_tracker = TrainTracker(frame_h)
    paused = False
    train_only_view = bool(TRAIN_ONLY_VIEW)
    frame_count = 0
    fps = 0.0
    prev_time = time.time()
    reference_frame = first_frame.copy()
    frame = first_frame.copy()

    window_name = "RAPS Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            paused = not paused
            if paused:
                paused_frame = frame.copy()
                if not train_only_view:
                    draw_badge(paused_frame, 30, 96, "PAUSED", COL_GOLD)
                frame = paused_frame
            print("[INFO] Paused." if paused else "[INFO] Resumed.")
        if key == ord("s"):
            train_tracker.reset_calibration()
            print("[INFO] Speed calibration reset.")
        if key == ord("r"):
            new_pts, new_poly = redraw_roi(reference_frame, current_roi_pts, crossing_y)
            if new_pts:
                current_roi_pts = new_pts
                track_poly = new_poly
                print(f"[INFO] ROI updated with {len(current_roi_pts)} points.")
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if key == ord("c"):
            new_crossing_y = redraw_crossing_line(reference_frame, crossing_y, current_roi_pts)
            if new_crossing_y is not None:
                crossing_y = new_crossing_y
                print(f"[INFO] Crossing line moved to {crossing_y / float(frame_h):.1%} of frame.")
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if key == ord("v"):
            train_only_view = not train_only_view
            print(
                "[INFO] Train-only visual mode enabled."
                if train_only_view
                else "[INFO] Full overlay view enabled."
            )

        if paused:
            cv2.imshow(window_name, frame)
            continue

        ret, raw_frame = cap.read()
        if not ret:
            break

        reference_frame = raw_frame.copy()
        frame_count += 1

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        results = model.track(
            raw_frame,
            persist=True,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )

        display = raw_frame.copy()
        alert_flag = False
        active_train_ids = set()

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in ALL_MONITORED or box.id is None:
                    continue

                track_id = int(box.id[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                class_name = model.names[cls_id]
                in_track = inside_roi((cx, cy), track_poly)

                if cls_id == CLS_TRAIN:
                    active_train_ids.add(track_id)
                    train_state = train_tracker.update(track_id, cx, cy, crossing_y, now)
                    speed_label = f"TRAIN ID:{track_id}  {train_state['speed_kmh']:.1f} km/h"
                    draw_box(display, x1, y1, x2, y2, cx, cy, speed_label, COL_TRAIN)
                    continue

                if cls_id in OBSTACLE_CLASSES and in_track:
                    alert_flag = True

                color = get_object_color(class_name, in_track)
                label = f"{class_name} ID:{track_id} {conf:.2f}"
                draw_box(display, x1, y1, x2, y2, cx, cy, label, color)

        train_tracker.cleanup(active_train_ids)

        danger_train = train_tracker.most_dangerous()
        ttc = danger_train["ttc"] if danger_train else 999.0
        train_speed_kmh = danger_train["speed_kmh"] if danger_train else 0.0
        train_direction = danger_train["direction"] if danger_train else "UNKNOWN"
        calib_locked = train_tracker.any_locked()

        risk_level, reason = assess_risk(alert_flag, danger_train)

        if not train_only_view:
            draw_roi_zone(display, track_poly, "TRACK ZONE", danger=alert_flag)
            draw_crossing_line(display, crossing_y)
            draw_hud(
                display,
                risk_level,
                reason,
                ttc,
                train_speed_kmh,
                train_direction,
                alert_flag,
                fps,
                frame_count,
                calib_locked,
                crossing_y,
                SESSION_LABEL,
                paused=False,
            )

        if risk_level in ("HIGH RISK", "MEDIUM RISK"):
            trigger_alert(display, risk_level, reason)

        frame = display
        cv2.imshow(window_name, frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
