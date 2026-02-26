import cv2
import numpy as np
from ultralytics import YOLO
from roi_config import TRACK_ROI

# ================== SETTINGS ==================
model = YOLO("yolov8n.pt")  # fast model for CPU (use yolov8s if GPU)

FPS = 30
CROSSING_Y = 360  # tuned for your video

# ============ TRAIN TRACKING VARIABLES ============
prev_train_center = None
train_speed = 0

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

    # -------- YOLO Detection --------
    results = model(frame, verbose=False)

    # -------- Draw ROI --------
    cv2.polylines(frame, [np.array(TRACK_ROI)], True, (0, 255, 255), 2)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]

            if class_name in ["person", "car", "truck", "bus", "motorcycle", "train"]:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # center point
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # ROI check
                if is_inside_roi((cx, cy), TRACK_ROI):
                    color = (0, 0, 255)
                    alert_flag = True
                else:
                    color = (0, 255, 0)

                # draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # ================= TRAIN TRACKING =================
                if class_name == "train":
                    train_cx, train_cy = cx, cy

                    if prev_train_center is not None:
                        dx = train_cx - prev_train_center[0]
                        dy = train_cy - prev_train_center[1]
                        pixel_distance = np.sqrt(dx * dx + dy * dy)
                        train_speed = pixel_distance * FPS

                    prev_train_center = (train_cx, train_cy)

                    # distance to crossing
                    distance_to_crossing = abs(train_cy - CROSSING_Y)

                    if train_speed > 0:
                        ttc = distance_to_crossing / train_speed
                    else:
                        ttc = 999

    # -------- Display Speed --------
    cv2.putText(frame,
                f"Train Speed(px/s): {train_speed:.1f}",
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

    # ================= SMART ALERT =================
    if alert_flag and train_speed > 5 and ttc < 5:
        cv2.putText(frame,
                    "HIGH RISK: TRAIN APPROACHING!",
                    (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3)
    elif alert_flag:
        cv2.putText(frame,
                    "WARNING: OBJECT ON TRACK",
                    (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 165, 255),
                    3)

    # -------- Show --------
    cv2.imshow("Railway Safety System", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
