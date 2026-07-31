"""Fine-tune an Apache-licensed RF-DETR detector from an RF-DETR dataset."""

from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any


MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="RF-DETR dataset root containing train/ and valid/ splits",
    )
    parser.add_argument("--model", choices=MODEL_CLASSES, default="nano")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps; effective batch is batch multiplied by this value",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early-stopping patience in epochs; 0 disables early stopping",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-test",
        action=BooleanOptionalAction,
        default=True,
        help="Evaluate the untouched test split after training when it is present",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "rfdetr-nano",
        help="Directory in which RF-DETR checkpoints, logs, and metadata are saved",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Training device, for example "cuda", "cuda:0", or "cpu"',
    )
    return parser.parse_args()


def dataset_metadata(dataset: Path, include_test: bool) -> tuple[list[str], dict[str, int]]:
    """Read metadata needed to configure and describe the training run."""
    split_directories = ("train", "valid", "test") if include_test else ("train", "valid")
    documents = {
        split: json.loads((dataset / split / "_annotations.coco.json").read_text(encoding="utf-8"))
        for split in split_directories
        if split != "test" or (dataset / split / "_annotations.coco.json").is_file()
    }
    categories = documents["train"]["categories"]
    classes = [
        str(category["name"])
        for category in sorted(categories, key=lambda category: int(category["id"]))
    ]
    split_counts = {
        ("val" if split == "valid" else split): len(document["images"])
        for split, document in documents.items()
    }
    return classes, split_counts


def rfdetr_model(model_name: str, num_classes: int, resolution: int | None):
    try:
        from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
    except ImportError as error:
        raise RuntimeError(
            "RF-DETR is not installed. Run: uv add 'rfdetr[train]'"
        ) from error
    models = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }
    options: dict[str, Any] = {"num_classes": num_classes}
    if resolution is not None:
        options["resolution"] = resolution
    return models[model_name](**options)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch < 1 or args.grad_accum_steps < 1:
        raise ValueError("--epochs, --batch, and --grad-accum-steps must be at least 1")
    if args.lr <= 0 or args.lr_encoder <= 0:
        raise ValueError("--lr and --lr-encoder must be greater than 0")
    if args.workers < -1 or args.patience < 0 or args.checkpoint_interval < 1:
        raise ValueError("Invalid --workers, --patience, or --checkpoint-interval value")
    if args.resolution is not None and args.resolution < 32:
        raise ValueError("--resolution must be at least 32")

    dataset = args.data.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    workers = args.workers if args.workers >= 0 else max(0, min(8, (os.cpu_count() or 1) // 2))

    parameters = {
        "command": [sys.executable, *sys.argv],
        "data": str(dataset),
        "model": args.model,
        "resolution": args.resolution,
        "epochs": args.epochs,
        "batch": args.batch,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch": args.batch * args.grad_accum_steps,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "workers": workers,
        "patience": args.patience,
        "checkpoint_interval": args.checkpoint_interval,
        "seed": args.seed,
        "run_test": args.run_test,
        "requested_device": args.device,
        "output": str(output),
    }

    classes, split_counts = dataset_metadata(dataset, args.run_test)
    run_test = args.run_test and "test" in split_counts
    metadata_path = output / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model": f"RF-DETR {args.model}",
                "classes": classes,
                "dataset": {"source": str(dataset), "split_image_counts": split_counts},
                "parameters": parameters,
                "status": "training",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    model = rfdetr_model(args.model, len(classes), args.resolution)
    training_options: dict[str, Any] = {
        "dataset_dir": str(dataset),
        "output_dir": str(output),
        "epochs": args.epochs,
        "batch_size": args.batch,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "num_workers": workers,
        "seed": args.seed,
        "checkpoint_interval": args.checkpoint_interval,
        "early_stopping": args.patience > 0,
        "early_stopping_patience": args.patience,
        "run_test": run_test,
        "class_names": classes,
    }
    if args.device is not None:
        training_options["device"] = args.device
    print(
        f"training RF-DETR {args.model}; classes={classes}; "
        f"train={split_counts['train']}; val={split_counts['val']}; "
        f"effective_batch={parameters['effective_batch']}; output={output}"
    )
    model.train(**training_options)

    metadata_path.write_text(
        json.dumps(
            {
                "model": f"RF-DETR {args.model}",
                "classes": classes,
                "dataset": {"source": str(dataset), "split_image_counts": split_counts},
                "parameters": parameters,
                "status": "completed",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
