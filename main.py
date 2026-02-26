import cv2
import numpy as np
from ultralytics import YOLO
from roi_config import TRACK_ROI

# Load YOLOv8 model (auto-downloads)
model = YOLO("yolov8s.pt")

# Open video
cap = cv2.VideoCapture("railway_video.mp4")

def is_inside_roi(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, verbose=False)

    # Draw ROI
    cv2.polylines(frame, [np.array(TRACK_ROI)], True, (0, 255, 255), 2)

    alert_flag = False

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Get class name
            class_name = model.names[cls_id]

            # We care about these objects
            if class_name in ["person", "car", "truck", "bus", "motorcycle", "train"]:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Center point
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Check if inside track
                if is_inside_roi((cx, cy), TRACK_ROI):
                    color = (0, 0, 255)
                    alert_flag = True
                else:
                    color = (0, 255, 0)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show alert
    if alert_flag:
        cv2.putText(frame, "WARNING: OBJECT ON TRACK!",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3)

    cv2.imshow("Railway Safety System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()