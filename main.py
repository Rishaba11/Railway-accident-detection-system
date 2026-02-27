import cv2
import numpy as np
from ultralytics import YOLO
from roi_config import TRACK_ROI

# ================== SETTINGS ==================
model = YOLO("yolov8n.pt")  # use yolov8s if GPU

FPS = 30
CROSSING_Y = 360

# Train tracking state per ID
train_states = {}

# ================== VIDEO ==================
cap = cv2.VideoCapture("railway_video.mp4")

def is_inside_roi(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    alert_flag = False
    ttc = 999
    train_direction = "UNKNOWN"
    train_speed_display = 0

    # 🔥 YOLO TRACKING (IMPORTANT CHANGE)
    results = model.track(frame, persist=True, verbose=False)

    # Draw ROI
    cv2.polylines(frame, [np.array(TRACK_ROI)], True, (0, 255, 255), 2)

    for r in results:
        if r.boxes is None:
            continue

        boxes = r.boxes

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]

            if box.id is None:
                continue

            track_id = int(box.id[0])

            if class_name in ["person", "car", "truck", "bus", "motorcycle", "train"]:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # ROI check
                if is_inside_roi((cx, cy), TRACK_ROI):
                    color = (0, 0, 255)
                    alert_flag = True
                else:
                    color = (0, 255, 0)

                # Draw box + ID
                label = f"{class_name} ID:{track_id} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # ================= TRAIN TRACKING =================
                if class_name == "train":

                    if track_id not in train_states:
                        train_states[track_id] = {
                            "prev_center": None,
                            "prev_distance": None,
                            "speed": 0,
                            "direction": "UNKNOWN"
                        }

                    state = train_states[track_id]

                    # ---- speed ----
                    if state["prev_center"] is not None:
                        dx = cx - state["prev_center"][0]
                        dy = cy - state["prev_center"][1]
                        pixel_distance = np.sqrt(dx * dx + dy * dy)
                        state["speed"] = pixel_distance * FPS

                    state["prev_center"] = (cx, cy)

                    # ---- distance ----
                    distance_to_crossing = abs(cy - CROSSING_Y)

                    # ---- direction ----
                    if state["prev_distance"] is not None:
                        if distance_to_crossing < state["prev_distance"]:
                            state["direction"] = "APPROACHING"
                        else:
                            state["direction"] = "MOVING AWAY"

                    state["prev_distance"] = distance_to_crossing

                    # ---- TTC ----
                    if state["speed"] > 0:
                        ttc = distance_to_crossing / state["speed"]
                    else:
                        ttc = 999

                    train_direction = state["direction"]
                    train_speed_display = state["speed"]

    # -------- Display Speed --------
    cv2.putText(frame,
                f"Train Speed(px/s): {train_speed_display:.1f}",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2)

    # -------- Display TTC --------
    cv2.putText(frame,
                f"TTC: {ttc:.2f}s",
                (50, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2)

    # -------- Display Direction --------
    cv2.putText(frame,
                f"Train: {train_direction}",
                (50, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2)

    # ================= RISK ASSESSMENT =================
    risk_level = "SAFE"
    risk_color = (0, 255, 0)

    if alert_flag:
        if train_direction == "APPROACHING" and ttc < 5:
            risk_level = "HIGH RISK"
            risk_color = (0, 0, 255)
        elif train_direction == "APPROACHING" and 5 <= ttc < 10:
            risk_level = "MEDIUM RISK"
            risk_color = (0, 165, 255)
        else:
            risk_level = "LOW RISK"
            risk_color = (0, 255, 255)

    cv2.putText(frame,
                f"Status: {risk_level}",
                (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                risk_color,
                3)

    cv2.imshow("Railway Safety System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()