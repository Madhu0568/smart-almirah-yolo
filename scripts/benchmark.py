#!/usr/bin/env python3
"""Benchmark YOLO inference on a folder of images and save JSON metrics."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure YOLO inference latency on test images.")
    parser.add_argument("--weights", required=True, help="YOLO model weights.")
    parser.add_argument("--source", required=True, help="Image file or folder of images.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="benchmark-results.json")
    args = parser.parse_args()

    if args.imgsz < 320 or args.imgsz > 1280 or args.imgsz % 32 != 0:
        parser.error("--imgsz must be between 320 and 1280 and divisible by 32.")
    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1.")
    if args.warmup < 0 or args.limit < 1:
        parser.error("--warmup cannot be negative and --limit must be positive.")

    source = Path(args.source).expanduser().resolve()
    if source.is_file():
        images = [source]
    elif source.is_dir():
        images = sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        parser.error(f"Source does not exist: {source}")

    images = images[:args.limit]
    if not images:
        parser.error(f"No supported images found in: {source}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        parser.error(f"Install project requirements first: {exc}")

    model = YOLO(args.weights)
    first_kwargs = {"source": str(images[0]), "imgsz": args.imgsz, "conf": args.conf, "verbose": False}
    if args.device:
        first_kwargs["device"] = args.device

    for _ in range(args.warmup):
        model.predict(**first_kwargs)

    latencies = []
    total_detections = 0
    for image_path in images:
        kwargs = {"source": str(image_path), "imgsz": args.imgsz, "conf": args.conf, "verbose": False}
        if args.device:
            kwargs["device"] = args.device
        started = time.perf_counter()
        results = model.predict(**kwargs)
        latencies.append((time.perf_counter() - started) * 1000)
        boxes = results[0].boxes
        total_detections += 0 if boxes is None else len(boxes)

    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "weights": args.weights,
        "source": str(source),
        "images_processed": len(latencies),
        "total_detections": total_detections,
        "imgsz": args.imgsz,
        "confidence_threshold": args.conf,
        "warmup_runs": args.warmup,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2),
            "median": round(statistics.median(latencies), 2),
            "p95": round(percentile(latencies, 95), 2),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
        },
        "throughput_images_per_second": round(1000 / statistics.mean(latencies), 3),
        "note": "Measure includes model preprocessing and inference; excludes model loading and image file enumeration.",
    }

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved benchmark report to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
