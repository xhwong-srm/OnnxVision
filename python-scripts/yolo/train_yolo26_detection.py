"""Fine-tune an Ultralytics YOLO26 object detector on a YOLO-format dataset."""

from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any


MODELS = ("n", "s", "m", "l", "x")


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Path to the YOLO data.yaml")
    parser.add_argument("--model", choices=MODELS, default="n", help="YOLO26 model scale")
    parser.add_argument("--weights", type=Path, help="Custom .pt weights instead of yolo26<model>.pt")
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate (lr0)")
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=20, help="0 disables early stopping")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic", action=BooleanOptionalAction, default=True,
        help="Request deterministic Ultralytics/PyTorch algorithms (default: enabled)",
    )
    parser.add_argument(
        "--run-test", action=BooleanOptionalAction, default=True,
        help="Evaluate the test split after training when data.yaml defines one",
    )
    parser.add_argument("--resume", action="store_true", help="Resume the supplied --weights checkpoint")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "runs" / "yolo26n",
        help="Run directory; Ultralytics artifacts are written directly here",
    )
    parser.add_argument("--device", help='For example "0", "0,1", "cpu", or "mps"')
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required and is installed with Ultralytics") from error
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read dataset YAML {path}: {error}") from error
    if not isinstance(document, dict) or "train" not in document or "val" not in document:
        raise ValueError(f"Dataset YAML must define train and val: {path}")
    return document


def class_names(document: dict[str, Any]) -> list[str]:
    names = document.get("names")
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        try:
            ordered = sorted((int(key), str(value)) for key, value in names.items())
        except (TypeError, ValueError) as error:
            raise ValueError("Dataset names mapping must use integer class IDs") from error
        if [key for key, _ in ordered] != list(range(len(ordered))):
            raise ValueError("Dataset class IDs must be contiguous and zero-based")
        return [value for _, value in ordered]
    raise ValueError("Dataset YAML must define names as a list or mapping")


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch == 0 or args.batch < -1:
        raise ValueError("--epochs must be positive; --batch must be -1 or greater than 0")
    if args.lr <= 0 or args.workers < -1 or args.patience < 0 or args.checkpoint_interval < 1:
        raise ValueError("Invalid learning rate, workers, patience, or checkpoint interval")
    if args.resolution < 32 or args.resolution % 32:
        raise ValueError("--resolution must be at least 32 and divisible by 32")
    if args.resume and args.weights is None:
        raise ValueError("--resume requires --weights pointing to a previous last.pt")

    data = args.data.resolve()
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data}")
    dataset = load_yaml(data)
    classes = class_names(dataset)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    workers = args.workers if args.workers >= 0 else max(0, min(8, (os.cpu_count() or 1) // 2))
    weights = args.weights.resolve() if args.weights else Path(f"yolo26{args.model}.pt")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Ultralytics is not installed. Run: uv add ultralytics") from error

    parameters = {
        "command": [sys.executable, *sys.argv],
        "data": str(data),
        "model": args.model,
        "weights": str(weights),
        "resolution": args.resolution,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "workers": workers,
        "patience": args.patience,
        "checkpoint_interval": args.checkpoint_interval,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "run_test": args.run_test,
        "resume": args.resume,
        "requested_device": args.device,
        "output": str(output),
    }
    metadata_path = output / "metadata.json"
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": f"YOLO26{args.model}",
        "classes": classes,
        "dataset": {"yaml": str(data)},
        "parameters": parameters,
        "status": "training",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    model = YOLO(str(weights))
    options: dict[str, Any] = {
        "data": str(data),
        "epochs": args.epochs,
        "imgsz": args.resolution,
        "batch": args.batch,
        "lr0": args.lr,
        "workers": workers,
        "patience": args.patience,
        "save_period": args.checkpoint_interval,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "project": str(output.parent),
        "name": output.name,
        "exist_ok": True,
        "resume": args.resume,
    }
    if args.device is not None:
        options["device"] = args.device
    print(
        f"training YOLO26{args.model}; classes={classes}; data={data}; "
        f"batch={args.batch}; output={output}"
    )
    model.train(**options)

    best = output / "weights" / "best.pt"
    last = output / "weights" / "last.pt"
    selected = best if best.is_file() else last
    test_metrics = None
    if args.run_test and "test" in dataset and selected.is_file():
        validation_options: dict[str, Any] = {
            "data": str(data), "split": "test", "imgsz": args.resolution, "workers": workers
        }
        if args.device is not None:
            validation_options["device"] = args.device
        test_metrics = YOLO(str(selected)).val(**validation_options).results_dict

    metadata.update(
        {
            "status": "completed",
            "best_checkpoint": str(best) if best.is_file() else None,
            "last_checkpoint": str(last) if last.is_file() else None,
            "test_metrics": test_metrics,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
