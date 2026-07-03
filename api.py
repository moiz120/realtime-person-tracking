import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import numpy as np
import cv2

from camera_service import CameraService
from detector import PersonTracker

app = FastAPI(
    title="Person Detection API",
    description="Real-time person detection & tracking service",
    version="1.0.0"
)
from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

camera_service = CameraService()
# separate stateless tracker for one-off image uploads (no ID tracking needed)
image_detector = PersonTracker()


class StatusResponse(BaseModel):
    running: bool
    current_count: int
    unique_count: int
    fps: float


class DetectResponse(BaseModel):
    person_count: int
    boxes: list


@app.get("/health")
def health_check():
    """Simple liveness check — used by monitoring tools/load balancers."""
    return {"status": "ok"}


@app.post("/stream/start")
def start_stream():
    success = camera_service.start()
    if not success:
        raise HTTPException(status_code=400, detail="Camera already running or unavailable.")
    return {"message": "Camera stream started."}


@app.post("/stream/stop")
def stop_stream():
    camera_service.stop()
    return {"message": "Camera stream stopped."}


@app.get("/status", response_model=StatusResponse)
def get_status():
    """Current live counts, FPS, and whether the stream is active."""
    return camera_service.get_status()


@app.get("/frame")
def get_frame():
    """Returns the latest annotated frame as a JPEG image."""
    frame_bytes = camera_service.get_frame()
    if frame_bytes is None:
        raise HTTPException(status_code=404, detail="No frame available. Is the stream running?")
    return Response(content=frame_bytes, media_type="image/jpeg")


@app.get("/events")
def get_events():
    """All logged entry events (track_id + timestamp) for this session."""
    return {"events": camera_service.get_events()}


@app.post("/stream/reset")
def reset_stream():
    camera_service.reset()
    return {"message": "Session reset."}


@app.post("/detect", response_model=DetectResponse)
async def detect_image(file: UploadFile = File(...)):
    """
    Stateless detection on a single uploaded image.
    Doesn't require the camera stream — useful for testing or batch processing.
    """
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    results = image_detector.model(frame, classes=[0], conf=0.5, verbose=False)
    boxes = results[0].boxes.xyxy.tolist() if results[0].boxes is not None else []

    return DetectResponse(person_count=len(boxes), boxes=boxes)