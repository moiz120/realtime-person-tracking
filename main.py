"""
Standalone CLI runner — quick test of detection without the dashboard or API.
Run with: python main.py
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import cv2
from detector import PersonTracker

def main():
    tracker = PersonTracker(conf_threshold=0.5)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Running... press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        annotated_frame, current_count, unique_count = tracker.process_frame(frame)

        cv2.putText(annotated_frame, f"Current: {current_count} | Unique: {unique_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Person Detection (CLI mode) - Press 'q' to quit", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()