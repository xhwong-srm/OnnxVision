"""Convert a Dataset Builder COCO detection export to Ultralytics YOLO format.

Input:
  images/train, images/val, optional images/test
  annotations/instances_train.json, instances_val.json, optional instances_test.json

Output:
  images/{train,val,test}, labels/{train,val,test}, and data.yaml
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
    parser.add_argument("--output", required=True, type=Path, help="New YOLO dataset directory")
    parser.add_argument(
        "--image-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Hard links avoid duplicating images when source and output are on the same volume.",
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


def read_categories(document: dict[str, Any], path: Path) -> dict[int, str]:
    categories: dict[int, str] = {}
    for item in document["categories"]:
        try:
            category_id = int(item["id"])
            name = str(item["name"]).strip()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid category in {path}: {item!r}") from error
        if not name or category_id in categories:
            raise ValueError(f"Empty or duplicate category in {path}: {item!r}")
        categories[category_id] = name
    if not categories:
        raise ValueError(f"No categories found in {path}")
    return categories


def resolve_image(dataset: Path, annotations: Path, file_name: str) -> Path:
    relative = Path(file_name)
    candidates = (
        dataset / relative,
        annotations.parent / relative,
        dataset / "images" / relative,
    )
    source = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if source is None:
        raise FileNotFoundError(f"COCO image was not found for {annotations}: {file_name}")
    return source


def place_image(source: Path, target: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(source, target)
        return
    try:
        os.link(source, target)
    except OSError as error:
        raise OSError(
            f"Cannot hard-link {source} to {target}. Use --image-mode copy for cross-volume output."
        ) from error


def convert_split(
    dataset: Path,
    output: Path,
    split: str,
    expected_categories: dict[int, str] | None,
    image_mode: str,
) -> tuple[dict[int, str], int, int]:
    annotation_path = dataset / "annotations" / f"instances_{split}.json"
    document = read_coco(annotation_path)
    categories = read_categories(document, annotation_path)
    if expected_categories is not None and categories != expected_categories:
        raise ValueError(f"Categories in {annotation_path} differ from train")

    class_by_category = {
        category_id: class_id for class_id, category_id in enumerate(sorted(categories))
    }
    images_by_id: dict[int, dict[str, Any]] = {}
    annotations_by_image: dict[int, list[str]] = {}
    for item in document["images"]:
        try:
            image_id = int(item["id"])
            width, height = int(item["width"]), int(item["height"])
            file_name = str(item["file_name"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid image in {annotation_path}: {item!r}") from error
        if image_id in images_by_id or width <= 0 or height <= 0:
            raise ValueError(f"Duplicate ID or invalid dimensions in {annotation_path}: {item!r}")
        images_by_id[image_id] = {"width": width, "height": height, "file_name": file_name}
        annotations_by_image[image_id] = []

    for item in document["annotations"]:
        try:
            image_id = int(item["image_id"])
            category_id = int(item["category_id"])
            x, y, width, height = (float(value) for value in item["bbox"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid annotation in {annotation_path}: {item!r}") from error
        if image_id not in images_by_id or category_id not in class_by_category:
            raise ValueError(f"Unknown image/category ID in {annotation_path}: {item!r}")
        image = images_by_id[image_id]
        image_width, image_height = image["width"], image["height"]
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(image_width), x + width), min(float(image_height), y + height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Empty bounding box after clipping in {annotation_path}: {item!r}")
        values = (
            class_by_category[category_id],
            ((x1 + x2) / 2.0) / image_width,
            ((y1 + y2) / 2.0) / image_height,
            (x2 - x1) / image_width,
            (y2 - y1) / image_height,
        )
        annotations_by_image[image_id].append(
            f"{values[0]} {values[1]:.9g} {values[2]:.9g} {values[3]:.9g} {values[4]:.9g}"
        )

    image_dir, label_dir = output / "images" / split, output / "labels" / split
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    used_names: set[str] = set()
    used_stems: set[str] = set()
    for image_id, item in images_by_id.items():
        source = resolve_image(dataset, annotation_path, item["file_name"])
        if source.name.casefold() in used_names:
            raise ValueError(f"Duplicate image filename in {annotation_path}: {source.name}")
        if source.stem.casefold() in used_stems:
            raise ValueError(
                f"Image filenames map to the same YOLO label filename in {annotation_path}: "
                f"{source.stem}.txt"
            )
        used_names.add(source.name.casefold())
        used_stems.add(source.stem.casefold())
        place_image(source, image_dir / source.name, image_mode)
        (label_dir / f"{source.stem}.txt").write_text(
            "\n".join(annotations_by_image[image_id])
            + ("\n" if annotations_by_image[image_id] else ""),
            encoding="utf-8",
        )
    return categories, len(images_by_id), len(document["annotations"])


def yaml_quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    dataset, output = args.data.resolve(), args.output.resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        path = dataset / "annotations" / f"instances_{split}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing required annotations: {path}")

    categories, train_images, train_boxes = convert_split(
        dataset, output, "train", None, args.image_mode
    )
    _, val_images, val_boxes = convert_split(
        dataset, output, "val", categories, args.image_mode
    )
    counts = {
        "train": {"images": train_images, "annotations": train_boxes},
        "val": {"images": val_images, "annotations": val_boxes},
    }
    splits = ["train", "val"]
    if (dataset / "annotations" / "instances_test.json").is_file():
        _, test_images, test_boxes = convert_split(
            dataset, output, "test", categories, args.image_mode
        )
        counts["test"] = {"images": test_images, "annotations": test_boxes}
        splits.append("test")

    names = [categories[category_id] for category_id in sorted(categories)]
    # Deliberately omit ``path`` so Ultralytics resolves these entries from the
    # directory containing data.yaml. This keeps the converted dataset portable.
    yaml_lines = [f"{split}: images/{split}" for split in splits]
    yaml_lines.append("names:")
    yaml_lines.extend(f"  {index}: {yaml_quote(name)}" for index, name in enumerate(names))
    (output / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    (output / "conversion_metadata.json").write_text(
        json.dumps(
            {
                "source": str(dataset),
                "image_mode": args.image_mode,
                "classes": names,
                "splits": counts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Converted {dataset} to {output}; classes={names}; splits={counts}")


if __name__ == "__main__":
    main()
