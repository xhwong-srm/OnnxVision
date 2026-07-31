"""Fine-tune a LibreYOLO RTMDet object detector on a YOLO-format dataset."""

from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any


MODELS = ("t", "s", "m", "l", "x")


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Path to the YOLO data.yaml")
    parser.add_argument("--model", choices=MODELS, default="t", help="RTMDet model scale")
    parser.add_argument("--weights", type=Path, help="Custom LibreYOLO .pt checkpoint")
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.004, help="Initial learning rate")
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=20, help="0 disables early stopping")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic", action=BooleanOptionalAction, default=True,
        help="Request deterministic PyTorch algorithms (default: enabled)",
    )
    parser.add_argument(
        "--pretrained", action=BooleanOptionalAction, default=True,
        help="Start from LibreYOLO's pretrained RTMDet checkpoint (default: enabled)",
    )
    parser.add_argument(
        "--run-test", action=BooleanOptionalAction, default=True,
        help="Evaluate the test split after training when data.yaml defines one",
    )
    parser.add_argument("--resume", action="store_true", help="Resume the supplied --weights checkpoint")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "runs" / "rtmdet-t",
        help="Run directory",
    )
    parser.add_argument("--device", default="auto", help='For example "0", "cpu", "mps", or "auto"')
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required and is installed with LibreYOLO") from error
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


def configure_determinism(enabled: bool) -> None:
    if not enabled:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch < 1:
        raise ValueError("--epochs and --batch must be positive")
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
    output.mkdir(parents=True, exist_ok=True)
    workers = args.workers if args.workers >= 0 else max(0, min(8, (os.cpu_count() or 1) // 2))
    configure_determinism(args.deterministic)

    try:
        from libreyolo import LibreRTMDet, LibreYOLO
    except ImportError as error:
        raise RuntimeError("LibreYOLO is not installed. Run: uv add libreyolo") from error

    if args.weights:
        checkpoint = args.weights.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        model = LibreYOLO(str(checkpoint), device=args.device, task="detect")
        weights = str(checkpoint)
    elif args.pretrained:
        weights = f"LibreRTMDet{args.model}.pt"
        model = LibreYOLO(weights, device=args.device, task="detect")
    else:
        weights = None
        model = LibreRTMDet(
            model_path=None, size=args.model, nb_classes=len(classes),
            device=args.device, task="detect",
        )

    parameters = {
        "command": [sys.executable, *sys.argv],
        "data": str(data),
        "model": args.model,
        "weights": weights,
        "pretrained": args.pretrained,
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
        "experimental_training_acknowledged": True,
    }
    metadata_path = output / "metadata.json"
    metadata = {
        "model": f"LibreRTMDet{args.model}",
        "classes": classes,
        "dataset": {"yaml": str(data)},
        "parameters": parameters,
        "status": "training",
        "warning": "LibreYOLO 1.4.0 marks RTMDet detection training as experimental.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(
        f"training LibreRTMDet{args.model}; classes={classes}; data={data}; "
        f"batch={args.batch}; output={output}"
    )
    results = model.train(
        data=str(data),
        allow_experimental=True,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.resolution,
        lr0=args.lr,
        device=args.device,
        workers=workers,
        seed=args.seed,
        project=str(output.parent),
        name=output.name,
        exist_ok=True,
        pretrained=args.pretrained,
        resume=args.resume,
        patience=args.patience,
        save_period=args.checkpoint_interval,
    )

    best_value = results.get("best_checkpoint")
    last_value = results.get("last_checkpoint")
    best = Path(best_value).resolve() if best_value else output / "weights" / "best.pt"
    last = Path(last_value).resolve() if last_value else output / "weights" / "last.pt"
    selected = best if best.is_file() else last
    test_metrics = None
    if args.run_test and "test" in dataset and selected.is_file():
        test_model = LibreYOLO(str(selected), device=args.device, task="detect")
        test_metrics = test_model.val(
            data=str(data), split="test", imgsz=args.resolution,
            batch=args.batch, workers=workers, device=args.device,
        )

    metadata.update(
        {
            "status": "completed",
            "best_checkpoint": str(best) if best.is_file() else None,
            "last_checkpoint": str(last) if last.is_file() else None,
            "training_results": results,
            "test_metrics": test_metrics,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
