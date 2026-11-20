# AI-Enabled Smart Almirah — Object Detection & Inventory Prototype

A Flask and OpenCV computer-vision prototype that runs YOLO object detection on uploaded images or webcam frames, estimates object color, and records detection events in a persistent SQLite inventory.

The application pipeline is implemented. The exact object categories it can recognize depend on the YOLO weights supplied at runtime.

## Features

- **YOLO inference:** Runs a supplied Ultralytics YOLO model on an uploaded image or the latest webcam frame.
- **Detection results:** Returns class label, confidence score, bounding box, estimated color, and measured inference time.
- **Webcam capture:** Opens and reads from a local camera using OpenCV.
- **Live preview:** Exposes the latest camera frames through an MJPEG endpoint.
- **Flask REST API:** Provides detection, camera, configuration, statistics, inventory, and detection-history endpoints.
- **SQLite persistence:** Stores inventory aggregates and detection history across application restarts.
- **Color estimate:** Uses a lightweight median-pixel/HSV heuristic over the detected crop. It is not a separately trained color classifier.
- **Dashboard:** Supports image uploads, camera controls, confidence settings, detection results, and inventory removal.
- **Optional voice feedback:** Uses Windows PowerShell and .NET System.Speech when enabled on Windows.

## Important: model weights determine the classes

The repository does not include YOLO weights or a training dataset. The default model path is **best.pt**, supplied by the user through the project folder or MODEL_PATH.

The model tested locally, **yolov8n.pt**, is the general-purpose COCO-pretrained YOLOv8n model. It can recognize COCO object categories such as people and bottles, but it does not provide dedicated garment classes such as shirt, jeans, dress, or jacket.

A clothing-trained model is required for garment-specific recognition. Do not rename yolov8n.pt to best.pt and describe it as a custom clothing model. The application loads the classes contained in the selected weights file; it does not train or fine-tune a model itself.

For a general-object smoke test on Windows PowerShell:

~~~powershell
$env:MODEL_PATH = "yolov8n.pt"
python app.py
~~~

For a clothing-trained model:

~~~powershell
$env:MODEL_PATH = "best.pt"
python app.py
~~~

Large .pt files are excluded from Git through .gitignore. Keep model weights locally or distribute them through a dedicated model-artifact location.

## Project structure

~~~text
smart-almirah-yolo/
├── app.py
├── requirements.txt
├── README.md
├── sample_output.txt
├── instance/                 # Created automatically; SQLite database
├── best.pt                   # Optional clothing-trained weights; not committed
├── static/
│   └── index.html
├── tests/
│   └── test_app.py
└── .github/
    └── workflows/
        └── tests.yml
~~~

## Setup

Python 3.10 or newer is recommended.

1. Clone the repository and enter the project directory.

~~~bash
git clone https://github.com/Madhu0568/smart-almirah-yolo.git
cd smart-almirah-yolo
~~~

2. Create and activate a virtual environment.

Windows PowerShell:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
~~~

macOS/Linux:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
~~~

3. Install dependencies.

~~~bash
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

4. Provide YOLO weights. For a general-object smoke test, set MODEL_PATH to a local yolov8n.pt. For clothing detection, provide a clothing-trained model such as best.pt.

Windows PowerShell:

~~~powershell
$env:MODEL_PATH = "yolov8n.pt"
~~~

macOS/Linux:

~~~bash
export MODEL_PATH="yolov8n.pt"
~~~

5. Start the application.

~~~bash
python app.py
~~~

Open http://127.0.0.1:5001. The SQLite database is created automatically at instance/smart_almirah.db by default.

The dashboard and non-model API endpoints can start without weights. Detection endpoints return HTTP 503 until a valid model file is available.

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| MODEL_PATH | best.pt | Path to the YOLO weights file |
| DATABASE_PATH | instance/smart_almirah.db | SQLite database path |
| CONFIDENCE_THRESHOLD | 0.65 | Minimum detection confidence (0 to 1) |
| CAMERA_INDEX | 0 | Default OpenCV camera index |
| TARGET_FPS | 15 | Target camera frame-read and preview rate |
| VOICE_ENABLED | true | Enable Windows voice announcements |
| HOST | 127.0.0.1 | Flask bind address |
| PORT | 5001 | Flask port |
| FLASK_DEBUG | false | Enable Flask debug mode |

TARGET_FPS limits the camera reader/preview loop. It is not an inference frame-skip setting. The current detector runs when an image is uploaded or when the user requests detection of the latest camera frame; continuous automatic inference is not implemented.

Color is estimated from pixels inside each detected bounding box. Lighting, shadows, background pixels, and multi-color garments can affect the estimate.

Windows voice feedback uses PowerShell with .NET System.Speech. On non-Windows systems the application continues without voice announcements.

## API reference

| Method | Endpoint | Description |
|---|---|---|
| GET | / | Dashboard |
| GET | /api/health | Service, model-file, database, and voice status |
| POST | /api/detect | Detect in multipart image field image, or use the latest frame when the camera is running |
| GET | /api/inventory | List saved items; optional category and color query parameters |
| GET | /api/inventory/summary | Counts grouped by category and color |
| DELETE | /api/inventory/{item_id} | Remove one inventory entry |
| POST | /api/camera/start | Open the configured webcam; optional JSON camera_index |
| POST | /api/camera/stop | Stop webcam capture and release the device |
| GET | /api/camera/status | Camera state and uptime |
| GET | /api/camera/feed | MJPEG camera stream |
| GET | /api/config | Read confidence and voice settings |
| PUT | /api/config | Update confidence_threshold and/or voice_enabled |
| GET | /api/stats | Detection totals and measured averages |
| GET | /api/detection-log | Recent detection events; optional limit (maximum 500) |

### Detect an uploaded image

~~~bash
curl -X POST -F "image=@sample.jpg" http://127.0.0.1:5001/api/detect
~~~

Windows PowerShell:

~~~powershell
curl.exe -X POST -F "image=@sample.jpg" http://127.0.0.1:5001/api/detect
~~~

### Detect a live camera frame

Start the webcam:

~~~bash
curl -X POST http://127.0.0.1:5001/api/camera/start
~~~

Request detection on the latest captured frame:

~~~bash
curl -X POST http://127.0.0.1:5001/api/detect
~~~

Stop the webcam:

~~~bash
curl -X POST http://127.0.0.1:5001/api/camera/stop
~~~

### Change settings

~~~powershell
curl.exe -X PUT http://127.0.0.1:5001/api/config -H "Content-Type: application/json" -d '{"confidence_threshold": 0.70, "voice_enabled": true}'
~~~

## Testing and verification

Run the automated API and SQLite tests with:

~~~bash
python -m unittest discover -s tests -v
~~~

GitHub Actions runs the unit tests on pushes and pull requests. These tests cover API behavior and database aggregation; they do not require a webcam or model weights and do not validate clothing-detection accuracy.

Manual Windows smoke testing has confirmed that the Flask dashboard starts, image upload reaches the inference endpoint, and the general COCO-pretrained yolov8n.pt model can return a bottle detection. At a very low confidence threshold, a false-positive vase prediction was also observed. This is a functional smoke test, not an accuracy benchmark.

The webcam capture and MJPEG routes are implemented. Camera availability, complete start/detect/stop behavior, and Windows voice output must be verified on the target machine. No long-duration stress-test or memory-leak result is claimed.

## Current limitations and next steps

- Garment-specific recognition requires a custom clothing dataset and clothing-trained YOLO weights; those artifacts are not included in this repository.
- The application currently loads external weights; it does not contain a model-training or fine-tuning pipeline.
- Image-size tuning and inference frame skipping are not exposed as settings in the current implementation.
- Color estimation is approximate.
- Repeatedly detecting the same physical garment increments the inventory event count. The application does not yet track unique garments across frames.
- Detection accuracy and inference speed depend on the selected model, input resolution, hardware, lighting, and camera quality. Benchmark results should be reported only after repeatable measurements.
