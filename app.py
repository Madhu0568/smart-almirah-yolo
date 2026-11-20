from datetime import datetime, timezone
import time
import uuid

from flask import Flask, jsonify, request
from ultralytics import YOLO

app = Flask(__name__)

# Use a clothing-trained model file (for example, best.pt) via MODEL_PATH.
# The default yolov8n.pt is COCO-trained and does not contain clothing classes.
import os
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
CONFIDENCE_THRESHOLD = 0.65

model = None
model_load_error = None
try:
    model = YOLO(MODEL_PATH)
except Exception as exc:
    model_load_error = str(exc)

# Temporary in-memory storage. This will be replaced with SQLite in the next step.
inventory = {}
detection_log = []
camera_status = {"active": False, "feed_url": None, "started_at": None}
detection_config = {
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "model": MODEL_PATH,
    "frame_batch_size": 1,
    "target_fps": 15,
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def get_model():
    if model is None:
        return None, (
            "YOLO model could not be loaded. Set MODEL_PATH to a valid "
            "clothing-trained .pt weights file and install the requirements."
        )
    return model, None


def extract_color(crop):
    """Return a simple dominant-color estimate from an object crop."""
    import cv2
    import numpy as np

    if crop is None or crop.size == 0:
        return "unknown"

    # Median BGR is less sensitive to small highlights than a single pixel.
    b, g, r = [int(v) for v in np.median(crop.reshape(-1, 3), axis=0)]
    brightness = (r + g + b) / 3

    if brightness < 45:
        return "black"
    if brightness > 220 and max(r, g, b) - min(r, g, b) < 25:
        return "white"
    if max(r, g, b) - min(r, g, b) < 22:
        return "gray"
    if r > g * 1.25 and r > b * 1.25:
        return "red"
    if g > r * 1.2 and g > b * 1.15:
        return "green"
    if b > r * 1.2 and b > g * 1.1:
        return "blue"
    if r > 150 and g > 100 and b < 110:
        return "beige" if brightness > 150 else "brown"
    return "unknown"


@app.route("/api/detect", methods=["POST"])
def run_detection():
    """Run real YOLO inference on an uploaded image."""
    loaded_model, error = get_model()
    if error:
        return jsonify({"error": error}), 503

    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify({
            "error": "Upload an image using multipart/form-data with field name 'image'."
        }), 400

    import cv2
    import numpy as np

    file_bytes = uploaded.read()
    if not file_bytes:
        return jsonify({"error": "The uploaded image is empty."}), 400

    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "The uploaded file is not a supported image."}), 400

    threshold = float(detection_config["confidence_threshold"])
    if not 0.0 <= threshold <= 1.0:
        return jsonify({"error": "Confidence threshold must be between 0 and 1."}), 400

    start = time.perf_counter()
    try:
        results = loaded_model.predict(
            source=frame,
            conf=threshold,
            verbose=False,
        )
    except Exception as exc:
        app.logger.exception("YOLO inference failed")
        return jsonify({"error": f"Inference failed: {exc}"}), 500
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    detections = []
    timestamp = utc_now()
    result = results[0]

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            category = loaded_model.names.get(class_id, str(class_id))
            crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            color = extract_color(crop)

            detection = {
                "item_id": str(uuid.uuid4())[:8],
                "category": str(category),
                "color": color,
                "confidence": round(confidence, 4),
                "bounding_box": {
                    "x": x1,
                    "y": y1,
                    "width": max(0, x2 - x1),
                    "height": max(0, y2 - y1),
                },
                "detected_at": timestamp,
                "inference_time_ms": elapsed_ms,
            }
            detections.append(detection)

            item_key = f"{color}_{category}".lower()
            if item_key in inventory:
                inventory[item_key]["count"] += 1
                inventory[item_key]["last_seen"] = timestamp
                inventory[item_key]["confidence"] = max(
                    inventory[item_key]["confidence"], confidence
                )
            else:
                inventory[item_key] = {
                    "item_id": detection["item_id"],
                    "category": str(category),
                    "color": color,
                    "count": 1,
                    "confidence": round(confidence, 4),
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                }

            detection_log.append(detection)

    return jsonify({
        "detections": detections,
        "count": len(detections),
        "inference_time_ms": elapsed_ms,
        "model": MODEL_PATH,
    })


@app.route("/api/inventory", methods=["GET"])
def get_inventory():
    category = request.args.get("category")
    color = request.args.get("color")
    items = list(inventory.values())

    if category:
        items = [item for item in items if item["category"].lower() == category.lower()]
    if color:
        items = [item for item in items if item["color"].lower() == color.lower()]

    return jsonify({
        "total_items": len(items),
        "total_count": sum(item["count"] for item in items),
        "items": items,
    })


@app.route("/api/inventory/summary", methods=["GET"])
def inventory_summary():
    by_category = {}
    by_color = {}
    for item in inventory.values():
        by_category[item["category"]] = by_category.get(item["category"], 0) + item["count"]
        by_color[item["color"]] = by_color.get(item["color"], 0) + item["count"]

    return jsonify({
        "total_unique_items": len(inventory),
        "total_count": sum(item["count"] for item in inventory.values()),
        "by_category": dict(sorted(by_category.items(), key=lambda pair: pair[1], reverse=True)),
        "by_color": dict(sorted(by_color.items(), key=lambda pair: pair[1], reverse=True)),
    })


@app.route("/api/inventory/<item_id>", methods=["DELETE"])
def remove_item(item_id):
    for key, item in list(inventory.items()):
        if item["item_id"] == item_id:
            removed = inventory.pop(key)
            return jsonify({"message": "Item removed", "item": removed})
    return jsonify({"error": "Item not found"}), 404


@app.route("/api/camera/start", methods=["POST"])
def start_camera():
    return jsonify({
        "error": "Live camera capture is not implemented yet. Use POST /api/detect with an image upload."
    }), 501


@app.route("/api/camera/stop", methods=["POST"])
def stop_camera():
    return jsonify({
        "error": "Live camera capture is not implemented yet."
    }), 501


@app.route("/api/camera/status", methods=["GET"])
def get_camera_status():
    return jsonify({**camera_status, "uptime": None})


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify(detection_config)


@app.route("/api/config", methods=["PUT"])
def update_config():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    if "confidence_threshold" in data:
        try:
            threshold = float(data["confidence_threshold"])
        except (TypeError, ValueError):
            return jsonify({"error": "confidence_threshold must be a number."}), 400
        if not 0.0 <= threshold <= 1.0:
            return jsonify({"error": "confidence_threshold must be between 0 and 1."}), 400
        detection_config["confidence_threshold"] = threshold

    return jsonify({"message": "Configuration updated", "config": detection_config})


@app.route("/api/detection-log", methods=["GET"])
def get_detection_log():
    limit = request.args.get("limit", 50, type=int)
    limit = max(1, min(limit, 500))
    return jsonify({"total": len(detection_log), "entries": detection_log[-limit:]})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    if not detection_log:
        return jsonify({
            "total_detections": 0,
            "unique_items": len(inventory),
            "average_confidence": 0,
            "average_inference_ms": 0,
            "model": MODEL_PATH,
            "fps_target": detection_config["target_fps"],
        })

    return jsonify({
        "total_detections": len(detection_log),
        "unique_items": len(inventory),
        "average_confidence": round(
            sum(item["confidence"] for item in detection_log) / len(detection_log), 3
        ),
        "average_inference_ms": round(
            sum(item["inference_time_ms"] for item in detection_log) / len(detection_log), 2
        ),
        "model": MODEL_PATH,
        "fps_target": detection_config["target_fps"],
    })


@app.route("/")
def dashboard():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
