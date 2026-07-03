\# Real-Time Multi-Person Detection \& Tracking System



A real-time person detection and tracking system that evolved from a basic HOG-based OpenCV detector into a full YOLOv8 + ByteTrack multi-object tracking pipeline, with a live dashboard and a REST API layer.



\## Features



\- \*\*Real-time detection \& tracking\*\* — YOLOv8 for person detection, ByteTrack for consistent multi-person tracking across frames

\- \*\*Live dashboard\*\* — Streamlit app with a custom dark indigo theme, showing the live camera feed with tracked bounding boxes

\- \*\*Crowd alerts\*\* — configurable threshold alerts when person count exceeds a set limit

\- \*\*Event logging\*\* — persistent log of detection events

\- \*\*CSV export\*\* — download tracking/session data for analysis

\- \*\*REST API\*\* — FastAPI layer exposing detection and tracking functionality for external consumption



\## Tech Stack



\- \*\*Detection/Tracking:\*\* YOLOv8 (Ultralytics), ByteTrack

\- \*\*Computer Vision:\*\* OpenCV

\- \*\*Dashboard:\*\* Streamlit

\- \*\*API:\*\* FastAPI

\- \*\*Language:\*\* Python



\## Project Structure



```

person-detector/

├── app.py                     # Streamlit dashboard entry point

├── api.py                     # FastAPI REST API entry point

├── main.py                    # Core detection/tracking runner

├── src/

│   ├── detector.py            # YOLOv8 detection logic

│   └── camera\_service.py      # Camera feed handling

├── .streamlit/

│   └── config.toml            # Dashboard theme config

└── requirements.txt

```



\## Setup



```bash

git clone https://github.com/moiz120/realtime-person-tracking.git

cd realtime-person-tracking

python -m venv venv

venv\\Scripts\\activate        # Windows

pip install -r requirements.txt

```



YOLOv8 will auto-download the pretrained weights (`yolov8n.pt`) on first run via the `ultralytics` package.



\## Usage



\*\*Run the dashboard:\*\*

```bash

streamlit run app.py

```



\*\*Run the REST API:\*\*

```bash

uvicorn api:app --reload

```



\*\*Run detection/tracking standalone:\*\*

```bash

python main.py

```



\## Notes



This project started as a simple HOG + SVM person detector using OpenCV and was rebuilt around YOLOv8 and ByteTrack for better accuracy and stable multi-person identity tracking across frames.



