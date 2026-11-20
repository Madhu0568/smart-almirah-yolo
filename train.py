#!/usr/bin/env python3
"""Train a YOLOv8 model on a user-provided YOLO-format clothing dataset.

Example:
    python train.py --data data/clothing.yaml --model yolov8n.pt --epochs 100 --imgsz 640

The resulting best.pt is copied to the requested output path (default: best.pt).
Dataset images, labels, and trained weights should not be committed to Git.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any


def load_dataset_config(data_yaml: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load and structurally validate a Ultralytics dataset YAML file."""
    yaml_path = Path(data_yaml).expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {yaml_path}")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install dependencies with pip install -r requirements.txt.") from exc

    try:
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Dataset YAML is invalid: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError("Dataset YAML must contain a mapping/object.")

    missing = [key for key in ("train", "val") if not config.get(key)]
    if missing:
        raise ValueError("Dataset YAML is missing required key(s): " + ", ".join(missing))

    names = config.get("names")
    nc = config.get("nc")
    if names is None and nc is None:
        raise ValueError("Dataset YAML must define class names using 'names' (recommended) or 'nc'.")

    if names is not None:
        if isinstance(names, list):
            class_count = len(names)
        elif isinstance(names, dict):
            class_count = len(names)
            expected_ids = {str(i) for i in range(class_count)}
            actual_ids = {str(key) for key in names.keys()}
            if actual_ids != expected_ids:
                raise ValueError("Class IDs in 'names' must be consecutive and start at 0.")
        else:
            raise ValueError("'names' must be a list or a mapping of consecutive class IDs to names.")
        if class_count == 0:
            raise ValueError("At least one class name must be provided.")
        if nc is not None and int(nc) != class_count:
            raise ValueError(f"'nc' is {nc}, but 'names' contains {class_count} classes.")

    return yaml_path, config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on a labelled clothing dataset.")
    parser.add_argument("--data", required=True, help="Path to a YOLO dataset YAML file.")
    parser.add_argument("--model", default="yolov8n.pt", help="Starting weights or model name (default: yolov8n.pt).")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100).")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size (default: 640).")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--workers", type=int, default=4, help="Data-loader workers (default: 4).")
    parser.add_argument("--device", default=None, help="Training device, e.g. cpu, 0, or 0,1.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience (default: 20).")
    parser.add_argument("--project", default="runs/clothing", help="Training output directory.")
    parser.add_argument("--name", default="exp", help="Training run name.")
    parser.add_argument("--output", default="best.pt", help="Where to copy the best trained weights.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.epochs < 1 or args.batch < 1 or args.workers < 0 or args.patience < 0:
        print("epochs and batch must be positive; workers and patience cannot be negative.", file=sys.stderr)
        return 2
    if args.imgsz < 320 or args.imgsz > 1280 or args.imgsz % 32 != 0:
        print("imgsz must be between 320 and 1280 and divisible by 32.", file=sys.stderr)
        return 2

    try:
        data_path, _ = load_dataset_config(args.data)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Dataset configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics is required. Install dependencies with pip install -r requirements.txt.", file=sys.stderr)
        return 2

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "patience": args.patience,
        "project": args.project,
        "name": args.name,
    }
    if args.device:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)
    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        print(f"Training finished, but best weights were not found at {best_weights}.", file=sys.stderr)
        return 1

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, output_path)
    print(f"Training complete. Best weights copied to: {output_path}")
    print("Set MODEL_PATH to this file before starting app.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
