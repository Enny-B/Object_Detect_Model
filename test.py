# detect_video.py
import cv2
import os
import math
from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")
class_names = ["wired_mouse", "wireless_mouse"]

cap = cv2.VideoCapture("detect.mp4")
if not cap.isOpened():
    print("Current working directory:", os.getcwd())
    print("Video file exists:", os.path.exists("detect.mp4"))
    print("Error: Cannot open video.")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        print("Cannot grab frame.")
        break

    results = model(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            cls = int(box.cls[0])
            label = class_names[cls]

            print(f"Detected: {label} | Confidence: {confidence:.2f}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Video Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
