import colorsys
from contextlib import contextmanager
import os
import sqlite3
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, Response, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder="static")
app.config.update(
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,
    MODEL_PATH=os.getenv("MODEL_PATH", "best.pt"),
    DATABASE_PATH=os.getenv("DATABASE_PATH", "instance/smart_almirah.db"),
    CONFIDENCE_THRESHOLD=float(os.getenv("CONFIDENCE_THRESHOLD", "0.65")),
    INFERENCE_IMAGE_SIZE=int(os.getenv("INFERENCE_IMAGE_SIZE", "640")),
    TARGET_FPS=max(1, int(os.getenv("TARGET_FPS", "15"))),
    VOICE_ENABLED=os.getenv("VOICE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
)

_model = None
_model_path_loaded = None
_model_lock = threading.Lock()

_camera_lock = threading.Lock()
_frame_lock = threading.Lock()
_camera_running = threading.Event()
_camera_capture = None
_camera_thread = None
_camera_started_at = None
_latest_frame = None


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def database_path():
    path = Path(app.config["DATABASE_PATH"]).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


@contextmanager
def get_db():
    connection = sqlite3.connect(str(database_path()), timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db():
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with get_db() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS inventory (
                item_id TEXT PRIMARY KEY,
                category TEXT NOT NULL COLLATE NOCASE,
                color TEXT NOT NULL COLLATE NOCASE,
                count INTEGER NOT NULL DEFAULT 0,
                confidence REAL NOT NULL DEFAULT 0,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                UNIQUE(category, color)
            );

            CREATE TABLE IF NOT EXISTS detection_log (
                detection_id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                color TEXT NOT NULL,
                confidence REAL NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                inference_time_ms REAL NOT NULL,
                detected_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_detection_log_detected_at
                ON detection_log(detected_at);
            """
        )


def resolve_model_path():
    path = Path(app.config["MODEL_PATH"]).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def get_model():
    """Load model lazily so the API and tests can start without weights installed."""
    global _model, _model_path_loaded
    path = resolve_model_path()
    path_key = str(path.resolve())

    with _model_lock:
        if _model is not None and _model_path_loaded == path_key:
            return _model, None

        if not path.is_file():
            return None, (
                "Model weights were not found. Add a clothing-trained YOLO .pt file "
                "to the project (default: best.pt) or set MODEL_PATH to its location."
            )

        try:
            from ultralytics import YOLO
            _model = YOLO(str(path))
            _model_path_loaded = path_key
            return _model, None
        except Exception as exc:
            app.logger.exception("Could not load YOLO weights")
            _model = None
            _model_path_loaded = None
            return None, f"Could not load the YOLO model: {exc}"


def estimate_color(crop):
    """Estimate an object's dominant color from the median BGR pixel values."""
    if crop is None or getattr(crop, "size", 0) == 0:
        return "unknown"

    import numpy as np

    b, g, r = [int(value) for value in np.median(crop.reshape(-1, 3), axis=0)]
    hue, saturation, value = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)

    if saturation < 0.18:
        if value < 0.18:
            return "black"
        if value > 0.84:
            return "white"
        return "gray"

    if hue < 0.04 or hue >= 0.96:
        return "red"
    if hue < 0.10:
        return "brown" if value < 0.55 else "orange"
    if hue < 0.17:
        return "yellow"
    if hue < 0.43:
        return "green"
    if hue < 0.53:
        return "cyan"
    if hue < 0.72:
        return "navy" if value < 0.38 else "blue"
    if hue < 0.86:
        return "purple"
    return "pink"


def record_detections(detections):
    """Persist detections and aggregate wardrobe inventory in SQLite."""
    with get_db() as db:
        for detection in detections:
            box = detection["bounding_box"]
            category = detection["category"]
            color = detection["color"]
            detected_at = detection["detected_at"]
            confidence = detection["confidence"]

            existing = db.execute(
                "SELECT item_id FROM inventory WHERE category = ? AND color = ?",
                (category, color),
            ).fetchone()

            if existing:
                db.execute(
                    """
                    UPDATE inventory
                    SET count = count + 1,
                        confidence = MAX(confidence, ?),
                        last_seen = ?
                    WHERE item_id = ?
                    """,
                    (confidence, detected_at, existing["item_id"]),
                )
            else:
                db.execute(
                    """
                    INSERT INTO inventory
                    (item_id, category, color, count, confidence, first_seen, last_seen)
                    VALUES (?, ?, ?, 1, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        category,
                        color,
                        confidence,
                        detected_at,
                        detected_at,
                    ),
                )

            db.execute(
                """
                INSERT INTO detection_log
                (detection_id, category, color, confidence, x, y, width, height,
                 inference_time_ms, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    detection["item_id"],
                    category,
                    color,
                    confidence,
                    box["x"],
                    box["y"],
                    box["width"],
                    box["height"],
                    detection["inference_time_ms"],
                    detected_at,
                ),
            )


def announce_detections(detections):
    """Use Windows System.Speech when enabled; safely do nothing on other platforms."""
    if not app.config["VOICE_ENABLED"] or os.name != "nt" or not detections:
        return

    labels = sorted({f'{item["color"]} {item["category"]}' for item in detections})
    phrase = "Detected " + ", ".join(labels)
    phrase = phrase.replace("'", "''").replace("\r", " ").replace("\n", " ")
    command = (
        "Add-Type -AssemblyName System.Speech; "
        "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$speaker.Speak('{phrase}'); $speaker.Dispose()"
    )

    try:
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
    except OSError:
        app.logger.warning("Windows voice feedback could not be started")


def camera_reader(capture):
    global _latest_frame
    failures = 0
    try:
        while _camera_running.is_set():
            success, frame = capture.read()
            if not success:
                failures += 1
                if failures >= 30:
                    app.logger.error("Camera stopped returning frames")
                    _camera_running.clear()
                    break
                time.sleep(0.05)
                continue

            failures = 0
            with _frame_lock:
                _latest_frame = frame

            time.sleep(1 / app.config["TARGET_FPS"])
    finally:
        capture.release()


def stop_camera_resources():
    """Stop the reader and release the device immediately.

    Releasing the capture before joining is important because a blocking
    VideoCapture.read() may otherwise keep the camera device open.
    """
    global _camera_capture, _camera_thread, _camera_started_at, _latest_frame

    _camera_running.clear()

    with _camera_lock:
        capture = _camera_capture
        thread = _camera_thread
        _camera_capture = None
        _camera_thread = None
        _camera_started_at = None

    if capture is not None:
        try:
            capture.release()
        except Exception:
            app.logger.exception("Could not release the camera device cleanly")

    if thread and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=5)
        if thread.is_alive():
            app.logger.warning("Camera reader thread did not stop within 5 seconds")

    with _frame_lock:
        _latest_frame = None


@app.errorhandler(413)
def request_entity_too_large(_error):
    return jsonify({"error": "Image is too large. Maximum upload size is 10 MB."}), 413


@app.route("/")
def dashboard():
    return app.send_static_file("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    model_path = resolve_model_path()
    return jsonify({
        "status": "ok",
        "model_path": app.config["MODEL_PATH"],
        "model_weights_found": model_path.is_file(),
        "model_loaded": _model is not None,
        "camera_active": _camera_running.is_set(),
        "database_path": str(database_path()),
        "voice_feedback_enabled": app.config["VOICE_ENABLED"],
        "voice_feedback_supported": os.name == "nt",
    })


@app.route("/api/detect", methods=["POST"])
def run_detection():
    """Detect items in an uploaded image or the latest live-camera frame."""
    uploaded = request.files.get("image")
    camera_source = False

    if uploaded is not None and uploaded.filename:
        try:
            import cv2
            import numpy as np
        except ImportError:
            return jsonify({"error": "Install OpenCV and NumPy to process uploaded images."}), 503

        file_bytes = uploaded.read()
        if not file_bytes:
            return jsonify({"error": "The uploaded image is empty."}), 400
        image_array = np.frombuffer(file_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "The uploaded file is not a supported image."}), 400
    else:
        if not _camera_running.is_set():
            return jsonify({
                "error": "Choose an image or start the camera before running detection."
            }), 400
        with _frame_lock:
            frame = None if _latest_frame is None else _latest_frame.copy()
        if frame is None:
            return jsonify({"error": "The camera has not produced a frame yet. Try again."}), 503
        camera_source = True

    loaded_model, model_error = get_model()
    if model_error:
        return jsonify({"error": model_error}), 503

    threshold = float(app.config["CONFIDENCE_THRESHOLD"])
    start = time.perf_counter()
    try:
        results = loaded_model.predict(
            source=frame,
            conf=threshold,
            imgsz=app.config["INFERENCE_IMAGE_SIZE"],
            verbose=False,
        )
    except Exception:
        app.logger.exception("YOLO inference failed")
        return jsonify({"error": "YOLO inference failed. Check the model file and input image."}), 500

    inference_time_ms = round((time.perf_counter() - start) * 1000, 2)
    timestamp = utc_now()
    result = results[0]
    detections = []
    boxes = result.boxes

    if boxes is not None:
        names = loaded_model.names
        for box in boxes:
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            x1 = max(0, min(x1, frame.shape[1]))
            x2 = max(0, min(x2, frame.shape[1]))
            y1 = max(0, min(y1, frame.shape[0]))
            y2 = max(0, min(y2, frame.shape[0]))

            class_id = int(box.cls[0].item())
            if isinstance(names, dict):
                category = names.get(class_id, str(class_id))
            else:
                category = names[class_id]

            confidence = float(box.conf[0].item())
            color = estimate_color(frame[y1:y2, x1:x2])
            detections.append({
                "item_id": str(uuid.uuid4()),
                "category": str(category).strip(),
                "color": color,
                "confidence": round(confidence, 4),
                "bounding_box": {
                    "x": x1,
                    "y": y1,
                    "width": max(0, x2 - x1),
                    "height": max(0, y2 - y1),
                },
                "inference_time_ms": inference_time_ms,
                "detected_at": timestamp,
            })

    if detections:
        try:
            record_detections(detections)
            announce_detections(detections)
        except sqlite3.Error:
            app.logger.exception("Could not save detections to SQLite")
            return jsonify({"error": "Detection succeeded, but saving to SQLite failed."}), 500

    return jsonify({
        "detections": detections,
        "count": len(detections),
        "inference_time_ms": inference_time_ms,
        "model": app.config["MODEL_PATH"],
        "source": "camera" if camera_source else "upload",
    })


@app.route("/api/inventory", methods=["GET"])
def get_inventory():
    conditions = []
    values = []

    category = request.args.get("category")
    color = request.args.get("color")
    if category:
        conditions.append("category = ?")
        values.append(category)
    if color:
        conditions.append("color = ?")
        values.append(color)

    query = "SELECT * FROM inventory"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY count DESC, last_seen DESC"

    with get_db() as db:
        items = [dict(row) for row in db.execute(query, values).fetchall()]

    return jsonify({
        "total_items": len(items),
        "total_count": sum(item["count"] for item in items),
        "items": items,
    })


@app.route("/api/inventory/summary", methods=["GET"])
def inventory_summary():
    with get_db() as db:
        total = db.execute(
            "SELECT COUNT(*) AS unique_items, COALESCE(SUM(count), 0) AS total_count FROM inventory"
        ).fetchone()
        categories = db.execute(
            "SELECT category, SUM(count) AS count FROM inventory GROUP BY category ORDER BY count DESC"
        ).fetchall()
        colors = db.execute(
            "SELECT color, SUM(count) AS count FROM inventory GROUP BY color ORDER BY count DESC"
        ).fetchall()

    return jsonify({
        "total_unique_items": total["unique_items"],
        "total_count": total["total_count"],
        "by_category": {row["category"]: row["count"] for row in categories},
        "by_color": {row["color"]: row["count"] for row in colors},
    })


@app.route("/api/inventory/<item_id>", methods=["DELETE"])
def remove_item(item_id):
    with get_db() as db:
        cursor = db.execute("DELETE FROM inventory WHERE item_id = ?", (item_id,))
        deleted = cursor.rowcount

    if not deleted:
        return jsonify({"error": "Item not found"}), 404
    return jsonify({"message": "Inventory item removed", "item_id": item_id})


@app.route("/api/camera/start", methods=["POST"])
def start_camera():
    global _camera_capture, _camera_thread, _camera_started_at

    if _camera_running.is_set():
        return jsonify({"message": "Camera is already running", "active": True,
                        "feed_url": "/api/camera/feed"})

    payload = request.get_json(silent=True)
    payload = payload if isinstance(payload, dict) else {}
    try:
        camera_index = int(payload.get("camera_index", os.getenv("CAMERA_INDEX", "0")))
        if camera_index < 0:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "camera_index must be a non-negative integer."}), 400

    try:
        import cv2
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        capture = cv2.VideoCapture(camera_index, backend)
    except Exception:
        app.logger.exception("OpenCV camera initialization failed")
        return jsonify({"error": "OpenCV could not initialize the camera."}), 503

    if not capture.isOpened():
        capture.release()
        return jsonify({
            "error": "Could not open the camera. Check the camera index and operating-system permissions."
        }), 503

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with _camera_lock:
        _camera_capture = capture
        _camera_started_at = utc_now()
        _camera_running.set()
        _camera_thread = threading.Thread(
            target=camera_reader,
            args=(capture,),
            name="smart-almirah-camera",
            daemon=True,
        )
        _camera_thread.start()

    return jsonify({
        "message": "Camera started",
        "active": True,
        "feed_url": "/api/camera/feed",
        "started_at": _camera_started_at,
    })


@app.route("/api/camera/stop", methods=["POST"])
def stop_camera():
    was_active = _camera_running.is_set()
    stop_camera_resources()
    return jsonify({
        "message": "Camera stopped" if was_active else "Camera was already stopped",
        "active": False,
    })


@app.route("/api/camera/status", methods=["GET"])
def camera_status():
    active = _camera_running.is_set()
    uptime_seconds = None
    if active and _camera_started_at:
        started = datetime.fromisoformat(_camera_started_at)
        uptime_seconds = max(0, int((datetime.now(timezone.utc) - started).total_seconds()))

    return jsonify({
        "active": active,
        "feed_url": "/api/camera/feed" if active else None,
        "started_at": _camera_started_at if active else None,
        "uptime_seconds": uptime_seconds,
    })


@app.route("/api/camera/feed", methods=["GET"])
def camera_feed():
    if not _camera_running.is_set():
        return jsonify({"error": "Start the camera before requesting the video feed."}), 409

    def generate_frames():
        import cv2
        while _camera_running.is_set():
            with _frame_lock:
                frame = None if _latest_frame is None else _latest_frame.copy()

            if frame is None:
                time.sleep(0.05)
                continue

            success, encoded = cv2.imencode(".jpg", frame)
            if success:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    encoded.tobytes() +
                    b"\r\n"
                )
            time.sleep(1 / app.config["TARGET_FPS"])

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify({
        "confidence_threshold": app.config["CONFIDENCE_THRESHOLD"],
        "image_size": app.config["INFERENCE_IMAGE_SIZE"],
        "target_fps": app.config["TARGET_FPS"],
        "model": app.config["MODEL_PATH"],
        "voice_enabled": app.config["VOICE_ENABLED"],
    })


@app.route("/api/config", methods=["PUT"])
def update_config():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    if "confidence_threshold" in data:
        value = data["confidence_threshold"]
        if isinstance(value, bool):
            return jsonify({"error": "confidence_threshold must be a number from 0 to 1."}), 400
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            return jsonify({"error": "confidence_threshold must be a number from 0 to 1."}), 400
        if not 0 <= threshold <= 1:
            return jsonify({"error": "confidence_threshold must be between 0 and 1."}), 400
        app.config["CONFIDENCE_THRESHOLD"] = threshold

    if "image_size" in data:
        value = data["image_size"]
        if isinstance(value, bool):
            return jsonify({"error": "image_size must be an integer from 320 to 1280 in steps of 32."}), 400
        try:
            image_size = int(value)
        except (TypeError, ValueError):
            return jsonify({"error": "image_size must be an integer from 320 to 1280 in steps of 32."}), 400
        if image_size < 320 or image_size > 1280 or image_size % 32 != 0:
            return jsonify({"error": "image_size must be between 320 and 1280 and divisible by 32."}), 400
        app.config["INFERENCE_IMAGE_SIZE"] = image_size

    if "voice_enabled" in data:
        if not isinstance(data["voice_enabled"], bool):
            return jsonify({"error": "voice_enabled must be true or false."}), 400
        app.config["VOICE_ENABLED"] = data["voice_enabled"]

    return jsonify({"message": "Configuration updated", "config": get_config_data()})


def get_config_data():
    return {
        "confidence_threshold": app.config["CONFIDENCE_THRESHOLD"],
        "image_size": app.config["INFERENCE_IMAGE_SIZE"],
        "target_fps": app.config["TARGET_FPS"],
        "model": app.config["MODEL_PATH"],
        "voice_enabled": app.config["VOICE_ENABLED"],
    }


@app.route("/api/detection-log", methods=["GET"])
def get_detection_log():
    limit = request.args.get("limit", default=50, type=int)
    limit = max(1, min(limit or 50, 500))

    with get_db() as db:
        total = db.execute("SELECT COUNT(*) FROM detection_log").fetchone()[0]
        rows = db.execute(
            """
            SELECT detection_id AS item_id, category, color, confidence,
                   x, y, width, height, inference_time_ms, detected_at
            FROM detection_log
            ORDER BY detected_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    entries = []
    for row in rows:
        item = dict(row)
        item["bounding_box"] = {
            "x": item.pop("x"),
            "y": item.pop("y"),
            "width": item.pop("width"),
            "height": item.pop("height"),
        }
        entries.append(item)

    return jsonify({"total": total, "entries": entries})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    with get_db() as db:
        stats = db.execute(
            """
            SELECT COUNT(*) AS total_detections,
                   COALESCE(AVG(confidence), 0) AS average_confidence,
                   COALESCE(AVG(inference_time_ms), 0) AS average_inference_ms
            FROM detection_log
            """
        ).fetchone()
        unique_items = db.execute("SELECT COUNT(*) FROM inventory").fetchone()[0]

    return jsonify({
        "total_detections": stats["total_detections"],
        "unique_items": unique_items,
        "average_confidence": round(stats["average_confidence"], 4),
        "average_inference_ms": round(stats["average_inference_ms"], 2),
        "model": app.config["MODEL_PATH"],
        "fps_target": app.config["TARGET_FPS"],
    })


init_db()


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5001"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)
