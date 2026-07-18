import cv2
import numpy as np
import time
import json


class QueueMonitor:

    def __init__(self):

        # Load Polygon ROI
        self.polygon = np.load("roi_points.npy")

        # Person entry time
        self.entry_times = {}

        # IDs currently inside queue
        self.current_ids = set()

        # Analytics
        self.people_served = 0
        self.total_wait = 0

        # Grace period for exit detection
        self.exit_times = {}
        self.exit_delay = 10  # seconds

    def process(self, frame, results):

        queue_count = 0
        annotated = frame.copy()
        current_frame_ids = set()
        current_waits = {}
        longest_wait_ids = []

        # -----------------------------
        # Draw Queue Polygon
        # -----------------------------
        cv2.polylines(
            annotated,
            [self.polygon.astype(np.int32)],
            True,
            (0, 255, 0),
            3
        )

        cv2.putText(
            annotated,
            "QUEUE AREA",
            tuple(self.polygon[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # -----------------------------
        # Process Detections
        # -----------------------------
        if (
            results
            and results[0].boxes is not None
            and results[0].boxes.id is not None
        ):

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, ids):

                xmin, ymin, xmax, ymax = map(int, box)

                # Use center point
                cx = (xmin + xmax) // 2
                cy = (ymin + ymax) // 2

                inside = cv2.pointPolygonTest(
                    self.polygon.astype(np.int32),
                    (cx, cy),
                    False
                )

                if inside >= 0:

                    track_id = int(track_id)

                    current_frame_ids.add(track_id)

                    # Cancel exit timer if person reappears
                    if track_id in self.exit_times:
                        del self.exit_times[track_id]

                    # First time entering queue
                    if track_id not in self.entry_times:
                        self.entry_times[track_id] = time.time()

                    queue_count += 1

                    wait = int(time.time() - self.entry_times[track_id])
                    current_waits[track_id] = wait

                    # Bounding Box
                    if track_id in longest_wait_ids:

                        color = (0, 0, 255)      # Red

                    else:

                        color = (0, 255, 0)      # Green

                    cv2.rectangle(
                        annotated,
                        (xmin, ymin),
                        (xmax, ymax),
                        color,
                        2
                    )

                    # ID
                    cv2.putText(
                        annotated,
                        f"ID:{track_id}",
                        (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

                    # Wait Time
                    cv2.putText(
                        annotated,
                        f"{wait}s",
                        (xmin, ymax + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

        if current_waits:

            avg_wait = sum(current_waits.values()) / len(current_waits)

            longest_wait = max(current_waits.values())

            longest_wait_ids = [

                pid

                for pid, wait in current_waits.items()

                if wait == longest_wait

            ]

        else:

            avg_wait = 0

            longest_wait = 0

            longest_wait_ids = []

        # -----------------------------
        # Detect Exits (Grace Period)
        # -----------------------------
        current_time = time.time()

        left_people = self.current_ids - current_frame_ids

        # Start exit timer
        for pid in left_people:

            if pid not in self.exit_times:
                self.exit_times[pid] = current_time

        # Remove after delay
        for pid in list(self.exit_times.keys()):

            if current_time - self.exit_times[pid] >= self.exit_delay:

                if pid in self.entry_times:

                    wait = current_time - self.entry_times[pid]

                    self.total_wait += wait

                    self.people_served += 1

                    del self.entry_times[pid]

                del self.exit_times[pid]

        self.current_ids = current_frame_ids

        # -----------------------------
        # Queue Status
        # -----------------------------
        if queue_count <= 5:
            status = "Low"

        elif queue_count <= 10:
            status = "Moderate"

        else:
            status = "High"

        # -----------------------------
        # Recommendation
        # -----------------------------
        if queue_count > 10:
            recommendation = "Open Counter 2"
        else:
            recommendation = "Queue Normal"

        # -----------------------------
        # JSON Output
        # -----------------------------
        queue_data = {

            "queue_count": queue_count,
            "people_served": self.people_served,
            "average_wait": round(avg_wait, 1),
            "longest_wait": round(longest_wait, 1),
            "longest_wait_ids": longest_wait_ids,
            "status": status,
            "recommendation": recommendation

        }

        # -----------------------------
        # Dashboard Text
        # -----------------------------
        cv2.putText(
            annotated,
            f"Queue Count : {queue_count}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            3
        )

        cv2.putText(
            annotated,
            f"People Served : {self.people_served}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            annotated,
            f"Avg Wait : {avg_wait:.1f}s",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            annotated,
            f"Status : {status}",
            (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            f"Recommendation : {recommendation}",
            (30, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        return annotated, queue_data