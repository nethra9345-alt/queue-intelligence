import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("models/yolov8n.pt")

# Open video
cap = cv2.VideoCapture("uploads/queue_video.mp4")

if not cap.isOpened():
    print("Error opening video")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Track people using ByteTrack
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],      # Detect only 'person'
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow("Queue Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()