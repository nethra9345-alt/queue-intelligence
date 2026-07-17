import cv2
from ultralytics import YOLO
from queue_logic import QueueMonitor

# Load YOLO model
model = YOLO("models/yolov8n.pt")

# Initialize Queue Monitor
queue_monitor = QueueMonitor()

# Open video
cap = cv2.VideoCapture("uploads/queue_video.mp4")

if not cap.isOpened():
    print("Error opening video")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # YOLO + ByteTrack
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        imgsz=1280,
        conf=0.10,
        iou=0.5,
        verbose=False
    )

    # Process Queue Logic
    output_frame, queue_count = queue_monitor.process(
        frame,
        results
    )

    cv2.imshow("Queue Intelligence", output_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()