"""Fine-tune an Apache-licensed RF-DETR detector from a Dataset Builder COCO export.

The Dataset Builder exports ``images/<split>`` and
``annotations/instances_<split>.json``.  RF-DETR expects each split to contain
an ``_annotations.coco.json`` file beside its images.  This script validates
the former, creates the latter as a temporary hard-linked layout, and leaves
the original dataset untouched.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any


MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}
SPLITS = ("train", "val", "test")


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help=(
            "Dataset Builder COCO export containing images/<split>/ and "
            "annotations/instances_<split>.json"
        ),
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


def read_coco(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read COCO annotations {path}: {error}") from error
    for key in ("images", "annotations", "categories"):
        if not isinstance(document.get(key), list):
            raise ValueError(f"COCO annotations {path} must contain a list named '{key}'")
    return document


def resolve_image_path(dataset: Path, annotation_path: Path, file_name: str) -> Path:
    relative_path = Path(file_name)
    candidates = (
        dataset / relative_path,
        annotation_path.parent / relative_path,
        dataset / "images" / relative_path,
    )
    image_path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if image_path is None:
        raise FileNotFoundError(f"COCO image was not found for {annotation_path}: {file_name}")
    return image_path


def categories_by_id(document: dict[str, Any], annotation_path: Path) -> dict[int, str]:
    categories: dict[int, str] = {}
    for category in document["categories"]:
        try:
            category_id = int(category["id"])
            name = str(category["name"]).strip()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid category in {annotation_path}: {category!r}") from error
        if not name or category_id in categories:
            raise ValueError(f"Invalid or duplicate category in {annotation_path}: {category!r}")
        categories[category_id] = name
    if not categories:
        raise ValueError(f"No categories found in {annotation_path}")
    return categories


def prepare_rfdetr_dataset(dataset: Path, staging_root: Path, run_test: bool) -> tuple[list[str], dict[str, int]]:
    """Make RF-DETR's split layout with hard links, falling back to file copies."""
    expected_splits = ("train", "val")
    if run_test and (dataset / "annotations" / "instances_test.json").is_file():
        expected_splits += ("test",)
    documents: dict[str, tuple[Path, dict[str, Any], dict[int, str]]] = {}
    canonical_categories: dict[int, str] | None = None

    for split in expected_splits:
        annotation_path = dataset / "annotations" / f"instances_{split}.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"Missing required {split} annotations: {annotation_path}. "
                "Export non-empty train and val splits from Dataset Builder."
            )
        document = read_coco(annotation_path)
        categories = categories_by_id(document, annotation_path)
        if canonical_categories is None:
            canonical_categories = categories
        elif categories != canonical_categories:
            raise ValueError(
                f"Categories in {annotation_path} differ from the training split: "
                f"{categories} != {canonical_categories}"
            )
        if not document["images"]:
            raise ValueError(f"No images found in {annotation_path}")
        documents[split] = (annotation_path, document, categories)

    assert canonical_categories is not None
    split_counts: dict[str, int] = {}
    for split, (annotation_path, document, _) in documents.items():
        destination = staging_root / ("valid" if split == "val" else split)
        destination.mkdir(parents=True, exist_ok=True)
        image_names: set[str] = set()
        for image in document["images"]:
            try:
                original_file_name = str(image["file_name"])
            except KeyError as error:
                raise ValueError(f"Image record without file_name in {annotation_path}: {image!r}") from error
            source = resolve_image_path(dataset, annotation_path, original_file_name)
            name = source.name
            if name in image_names:
                raise ValueError(
                    f"Duplicate image filename in {annotation_path}: {name}. "
                    "Rename the source images before exporting."
                )
            image_names.add(name)
            target = destination / name
            try:
                os.link(source, target)
            except OSError:
                # Hard links are preferred because training staging should not duplicate
                # a potentially large production dataset. Copy only if the source lives
                # on a different volume or the filesystem does not support hard links.
                shutil.copy2(source, target)
            image["file_name"] = name
        (destination / "_annotations.coco.json").write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
        split_counts[split] = len(document["images"])

    classes = [canonical_categories[category_id] for category_id in sorted(canonical_categories)]
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
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset}")
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

    # RF-DETR reads its COCO annotations while model.train() is running, so the
    # temporary layout must remain alive for the whole call.
    with tempfile.TemporaryDirectory(prefix="rfdetr-dataset-", dir=output) as temporary_directory:
        prepared_dataset = Path(temporary_directory)
        classes, split_counts = prepare_rfdetr_dataset(dataset, prepared_dataset, args.run_test)
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
            "dataset_dir": str(prepared_dataset),
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
