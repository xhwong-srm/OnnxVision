"""Convert a Dataset Builder COCO export into RF-DETR's split-directory layout.

Input layout:
  images/train, images/val, optional images/test
  annotations/instances_train.json, instances_val.json, optional instances_test.json

Output layout:
  train/_annotations.coco.json and images
  valid/_annotations.coco.json and images
  optional test/_annotations.coco.json and images
"""

from __future__ import annotations

import json
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Any


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Dataset Builder COCO export root")
    parser.add_argument("--output", required=True, type=Path, help="New RF-DETR dataset directory")
    parser.add_argument(
        "--image-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to place images in the output. Hard links avoid duplicating data on the same volume.",
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


def write_split(
    dataset: Path,
    split: str,
    destination_name: str,
    output: Path,
    expected_categories: dict[int, str] | None,
    image_mode: str,
) -> tuple[dict[int, str], int]:
    annotation_path = dataset / "annotations" / f"instances_{split}.json"
    document = read_coco(annotation_path)
    categories = categories_by_id(document, annotation_path)
    if expected_categories is not None and categories != expected_categories:
        raise ValueError(
            f"Categories in {annotation_path} differ from train: {categories} != {expected_categories}"
        )
    if not document["images"]:
        raise ValueError(f"No images found in {annotation_path}")

    destination = output / destination_name
    destination.mkdir(parents=True)
    names: set[str] = set()
    for image in document["images"]:
        if "file_name" not in image:
            raise ValueError(f"Image record without file_name in {annotation_path}: {image!r}")
        source = resolve_image_path(dataset, annotation_path, str(image["file_name"]))
        if source.name in names:
            raise ValueError(f"Duplicate image filename in {annotation_path}: {source.name}")
        names.add(source.name)
        target = destination / source.name
        if image_mode == "copy":
            shutil.copy2(source, target)
        else:
            try:
                os.link(source, target)
            except OSError as error:
                raise OSError(
                    f"Cannot hard-link {source} to {target}. Use --image-mode copy for a cross-volume export."
                ) from error
        image["file_name"] = source.name
    (destination / "_annotations.coco.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    return categories, len(document["images"])


def main() -> None:
    args = parse_args()
    dataset = args.data.resolve()
    output = args.output.resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset}")
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError(f"Output directory must be new or empty: {output}")
    else:
        output.mkdir(parents=True)

    for split in ("train", "val"):
        annotation_path = dataset / "annotations" / f"instances_{split}.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing required annotations: {annotation_path}")

    categories, train_count = write_split(
        dataset, "train", "train", output, None, args.image_mode
    )
    _, val_count = write_split(dataset, "val", "valid", output, categories, args.image_mode)
    counts = {"train": train_count, "val": val_count}
    test_annotations = dataset / "annotations" / "instances_test.json"
    if test_annotations.is_file():
        _, test_count = write_split(dataset, "test", "test", output, categories, args.image_mode)
        counts["test"] = test_count

    classes = [categories[category_id] for category_id in sorted(categories)]
    (output / "conversion_metadata.json").write_text(
        json.dumps(
            {
                "source": str(dataset),
                "image_mode": args.image_mode,
                "classes": classes,
                "split_image_counts": counts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Converted {dataset} to {output}; classes={classes}; split images={counts}")


if __name__ == "__main__":
    main()
