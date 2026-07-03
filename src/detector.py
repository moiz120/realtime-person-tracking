from ultralytics import YOLO

class PersonTracker:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.seen_ids = set()  # every unique track ID ever seen (lifetime unique count)

    def process_frame(self, frame):
        """
        Runs detection + tracking on a single frame.
        Returns: annotated frame, current frame count, lifetime unique count
        """
        # persist=True tells the tracker to remember IDs across frames (not just this call)
        # tracker="bytetrack.yaml" uses ByteTrack, bundled with ultralytics
        results = self.model.track(
            frame,
            classes=[0],              # person only
            conf=self.conf_threshold,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        annotated_frame = results[0].plot()

        current_count = 0
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.int().tolist()
            current_count = len(ids)
            for track_id in ids:
                self.seen_ids.add(track_id)

        lifetime_unique_count = len(self.seen_ids)

        return annotated_frame, current_count, lifetime_unique_count

    def reset(self):
        """Clears tracking history — useful for a 'reset session' button."""
        self.seen_ids.clear()
        self.model = YOLO(self.model.model_name if hasattr(self.model, "model_name") else "yolov8n.pt")