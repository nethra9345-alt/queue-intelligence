import cv2
import json
import time
from ultralytics import YOLO
from queue_logic import QueueMonitor

# Load model
model = YOLO("models/best.pt")

# Queue monitor
queue_monitor = QueueMonitor()

def generate_frames():

    cap = cv2.VideoCapture("uploads/queue_video.mp4")

    if not cap.isOpened():
        print("Error opening video")
        return

    prev_time = time.time()
    start_time = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            imgsz=1280,
            conf=0.25,
            iou=0.6,
            verbose=False
        )

        confidence = 0

        if (
            results
            and results[0].boxes is not None
            and results[0].boxes.conf is not None
        ):

            confs = results[0].boxes.conf.cpu().numpy()

            if len(confs) > 0:

                confidence = float(confs.mean() * 100)

        output_frame, data = queue_monitor.process(
            frame,
            results
        )

        # -------------------------
        # Detection Confidence
        # -------------------------
        data["confidence"] = round(confidence, 1)

        # -------------------------
        # Live FPS
        # -------------------------
        current = time.time()
        fps = 1 / (current - prev_time)
        prev_time = current

        data["fps"] = round(fps, 1)

        # -------------------------
        # Throughput
        # -------------------------
        elapsed = (current - start_time) / 60

        if elapsed > 0:
            throughput = queue_monitor.people_served / elapsed
        else:
            throughput = 0

        data["throughput"] = round(throughput, 2)

        # -------------------------
        # Save Final JSON
        # -------------------------
        with open("output/queue_data.json", "w") as file:
            json.dump(data, file, indent=4)

        ret, buffer = cv2.imencode(".jpg", output_frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

    cap.release()