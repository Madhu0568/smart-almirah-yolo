# AI-Enabled Smart Almirah

A Flask-based computer-vision application for detecting clothing with a YOLO model, estimating object colors, and maintaining a persistent wardrobe inventory in SQLite.

## What it does

- Runs YOLO inference on an uploaded image or the latest frame from a live webcam.
- Returns each detected class, confidence score, estimated color, bounding box, and measured frame inference time.
- Aggregates detections into a wardrobe inventory grouped by category and estimated color.
- Persists inventory and detection history in SQLite so records survive application restarts.
- Provides camera start/stop controls and an MJPEG camera preview.
- Includes a browser dashboard for image upload, live-camera detection, inventory management, and confidence settings.
- Supports optional Windows voice announcements through PowerShell and System.Speech.

## Important: use clothing-trained weights

The application expects a YOLO weights file named **best.pt** in the project root by default. The weights must be trained or fine-tuned to recognize the clothing classes you want to detect.

The generic Ultralytics YOLOv8n COCO weights are not a clothing detector for classes such as shirts, jeans, jackets, or dresses. Do not rename generic weights to best.pt and assume they will recognize clothing. Use your own apparel-trained weights or set MODEL_PATH to a compatible model file.

Large model files are intentionally excluded from Git. Keep the trained weights on your local machine, or use an approved model-artifact download process.

## Project structure

~~~text
smart-almirah-yolo/
├── app.py
├── requirements.txt
├── README.md
├── sample_output.txt
├── instance/                 # Created automatically; SQLite database
├── best.pt                   # Your clothing-trained weights; not committed
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
.\\.venv\\Scripts\\Activate.ps1
~~~

macOS/Linux:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
~~~

3. Install the packages.

~~~bash
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

4. Put your clothing-trained YOLO weights in the project folder as best.pt, or configure MODEL_PATH to the path of your weights.

Windows PowerShell:

~~~powershell
$env:MODEL_PATH = "best.pt"
~~~

macOS/Linux:

~~~bash
export MODEL_PATH="best.pt"
~~~

5. Start the application.

~~~bash
python app.py
~~~

Open http://127.0.0.1:5001 in your browser. The SQLite database is created automatically at instance/smart_almirah.db by default.

The dashboard and non-model API endpoints can start without weights, but detection endpoints return HTTP 503 until a valid model file is available.

## Configuration

The following environment variables are supported:

| Variable | Default | Description |
|---|---|---|
| MODEL_PATH | best.pt | Path to clothing-trained YOLO weights |
| DATABASE_PATH | instance/smart_almirah.db | SQLite database path |
| CONFIDENCE_THRESHOLD | 0.65 | Minimum detection confidence (0 to 1) |
| CAMERA_INDEX | 0 | Default OpenCV camera index |
| TARGET_FPS | 15 | Target frame-read/display rate |
| VOICE_ENABLED | true | Enable Windows voice announcements |
| HOST | 127.0.0.1 | Flask bind address |
| PORT | 5001 | Flask port |
| FLASK_DEBUG | false | Enable Flask debug mode |

Color is estimated from the median pixel color inside each detected bounding box. It is a lightweight heuristic, not a separately trained color-classification model; lighting, shadows, and mixed-color garments can affect the estimate.

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
| POST | /api/camera/stop | Stop webcam capture |
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

Then request detection on the latest captured frame:

~~~bash
curl -X POST http://127.0.0.1:5001/api/detect
~~~

Stop the webcam when finished:

~~~bash
curl -X POST http://127.0.0.1:5001/api/camera/stop
~~~

### Change settings

~~~bash
curl -X PUT http://127.0.0.1:5001/api/config \\
  -H "Content-Type: application/json" \\
  -d '{"confidence_threshold": 0.70, "voice_enabled": true}'
~~~

## Testing

The API and SQLite tests use Python's built-in unittest framework:

~~~bash
python -m unittest discover -s tests -v
~~~

The GitHub Actions workflow runs these tests on pushes and pull requests. The automated tests do not require a webcam or model weights. To validate actual clothing recognition, run the application locally with your own trained weights and test images.

## Performance reporting

Inference time is measured around the actual YOLO predict call for each frame. Results depend on the model, input resolution, hardware, and execution provider. No fixed latency or speedup is claimed in this repository; benchmark your own setup before reporting performance figures.

## Limitations

- Correct clothing recognition depends on the class labels and quality of the supplied trained weights.
- Color estimation is approximate.
- Webcam support depends on local device availability and operating-system camera permissions.
- The inventory count records detection events. Repeatedly detecting the same garment can increase the count; it is not a unique physical-garment tracking system.
