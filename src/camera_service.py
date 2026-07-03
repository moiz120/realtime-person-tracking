import cv2
import threading
import time
from detector import PersonTracker

class CameraService:
    def __init__(self):
        self.tracker = PersonTracker()
        self.cap = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock()

        # shared state, protected by self.lock
        self.latest_frame = None       # JPEG bytes of latest annotated frame
        self.current_count = 0
        self.unique_count = 0
        self.fps = 0.0

    def start(self):
        if self.running:
            return False  # already running
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            return False
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        return True

    def _loop(self):
        prev_time = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            annotated, current, unique = self.tracker.process_frame(frame)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time

            # encode as JPEG so the API can serve it over HTTP
            ok, buffer = cv2.imencode(".jpg", annotated)

            with self.lock:
                if ok:
                    self.latest_frame = buffer.tobytes()
                self.current_count = current
                self.unique_count = unique
                self.fps = fps

    def get_status(self):
        with self.lock:
            return {
                "running": self.running,
                "current_count": self.current_count,
                "unique_count": self.unique_count,
                "fps": round(self.fps, 1)
            }

    def get_frame(self):
        with self.lock:
            return self.latest_frame

    def get_events(self):
        return self.tracker.events

    def reset(self):
        self.tracker.reset()