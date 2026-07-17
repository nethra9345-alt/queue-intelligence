import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("models/yolov8n.pt")

# Video path
VIDEO_PATH = "uploads/queue_video.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error opening video")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO
    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("Queue Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()