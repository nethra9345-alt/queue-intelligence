import cv2
import numpy as np

VIDEO_PATH = "uploads/queue_video.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

ret, frame = cap.read()

if not ret:
    print("Cannot open video")
    exit()

points = []

def mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(points)

cv2.namedWindow("Select ROI")
cv2.setMouseCallback("Select ROI", mouse)

while True:

    img = frame.copy()

    # Draw points
    for p in points:
        cv2.circle(img, p, 5, (0,0,255), -1)

    # Draw polygon
    if len(points) > 1:
        cv2.polylines(
            img,
            [np.array(points)],
            False,
            (0,255,0),
            2
        )

    cv2.putText(
        img,
        "Left Click = Add Point | S = Save | Z = Undo | Q = Quit",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.imshow("Select ROI", img)

    key = cv2.waitKey(1)

    if key == ord('z'):
        if len(points):
            points.pop()

    elif key == ord('s'):
        np.save("roi_points.npy", np.array(points))
        print("ROI Saved!")
        break

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()