import os
import math
import cv2
from flask import Flask, request, render_template
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv8 model
model_path = "runs/detect/train2/weights/best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at: {model_path}")
model = YOLO(model_path)

# Define your class names
class_names = ["wired_mouse", "wireless_mouse"]

@app.route("/", methods=["GET", "POST"])
def index():
    user_img = None
    detection_info = []

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return "file not uploaded", 400

        # Ensure save directory exists
        image_dir = os.path.join("static", "images")
        os.makedirs(image_dir, exist_ok=True)

        # Save uploaded image
        filepath = os.path.join(image_dir, file.filename)
        file.save(filepath)

        # Run YOLO detection
        results = model(filepath)
        result = results[0]

        # Draw annotations
        annotated_img = result.plot()
        output_filename = "detected-mouse.jpg"
        output_path = os.path.join(image_dir, output_filename)
        success = cv2.imwrite(output_path, annotated_img)

        if not success:
            return "Failed to save image", 500

        # Log and collect detected labels
        if result.boxes:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = class_names[cls_id]
                print(f"Detected: {label} | Confidence: {conf:.2f}")
                detection_info.append((label, f"{conf:.2f}"))
        else:
            print("No objects detected.")

        user_img = f"images/{output_filename}"

    return render_template("index.html", user_img=user_img, detections=detection_info)


if __name__ == "__main__":
    app.run(debug=True)
