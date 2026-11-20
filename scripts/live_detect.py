#!/usr/bin/env python3
"""Run YOLO inference on a webcam/video with configurable frame skipping.

Examples:
    python scripts/live_detect.py --weights yolov8n.pt --source 0
    python scripts/live_detect.py --weights best.pt --source 0 --imgsz 640 --conf 0.30 --frame-skip 3

Press Q in the preview window to stop.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def main() -> int:
    parser = argparse.ArgumentParser(description="Live YOLO inference with frame skipping.")
    parser.add_argument("--weights", default="best.pt", help="YOLO weights path.")
    parser.add_argument("--source", default="0", help="Camera index or video file path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size, 320–1280.")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold, 0–1.")
    parser.add_argument("--frame-skip", type=int, default=2, help="Run inference on every Nth frame (default: 2).")
    args = parser.parse_args()

    if args.imgsz < 320 or args.imgsz > 1280 or args.imgsz % 32 != 0:
        parser.error("--imgsz must be between 320 and 1280 and divisible by 32.")
    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1.")
    if args.frame_skip < 1:
        parser.error("--frame-skip must be at least 1.")

    weights = Path(args.weights).expanduser()
    if not weights.is_file() and args.weights.endswith(".pt"):
        parser.error(f"Model weights not found: {weights}")

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        parser.error(f"Install project requirements first: {exc}")

    model = YOLO(str(weights) if weights.is_file() else args.weights)
    source = parse_source(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        parser.error(f"Could not open video source: {args.source}")

    frame_index = 0
    last_annotated = None
    last_inference_ms = 0.0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_index += 1
            if (frame_index - 1) % args.frame_skip == 0:
                started = time.perf_counter()
                prediction = model.predict(
                    source=frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    verbose=False,
                )[0]
                last_inference_ms = (time.perf_counter() - started) * 1000
                last_annotated = prediction.plot()

            display = last_annotated if last_annotated is not None else frame
            cv2.putText(
                display,
                f"Frame: {frame_index} | Skip: {args.frame_skip} | Inference: {last_inference_ms:.1f} ms",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (40, 220, 160),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Smart Almirah - YOLO Live Detection", display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
