"""Convert object-detection datasets between COCO, YOLO, and RF-DETR layouts.

The converter uses a small canonical representation internally:

    input format -> canonical images/boxes/classes -> output format

Supported input/output layouts:

COCO
    ``images/{train,val,test}`` and
    ``annotations/instances_{train,val,test}.json``.

YOLO
    A ``data.yaml`` with ``images`` and ``labels`` split directories. Both
    list-style and mapping-style ``names`` values are accepted.

RF-DETR
    ``train/``, ``valid/``, and optional ``test/`` directories, each with
    ``_annotations.coco.json`` and its images.

Examples::

    uv run python python-scripts/convert_detection_dataset.py \
        --input-format coco --output-format yolo \
        --data images/seal_dataset --output artifacts/seal-yolo

    uv run python python-scripts/convert_detection_dataset.py \
        --input-format yolo --output-format rfdetr \
        --data artifacts/seal-yolo --output artifacts/seal-rfdetr

The converter handles bounding-box object detection only. Segmentation
polygons, keypoints, and other format-specific fields are not carried through
the canonical representation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FORMATS = ("coco", "yolo", "rfdetr")
REQUIRED_SPLITS = ("train", "val")
OPTIONAL_SPLITS = ("test",)
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class Detection:
    """A zero-based class and clipped pixel-space XYXY bounding box."""

    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    iscrowd: int = 0


@dataclass
class ImageRecord:
    source: Path
    width: int
    height: int
    detections: list[Detection] = field(default_factory=list)


@dataclass
class DetectionDataset:
    classes: list[str]
    splits: dict[str, list[ImageRecord]]
    source: str | None = None


def canonical_split(name: str) -> str:
    normalized = name.casefold()
    if normalized == "train":
        return "train"
    if normalized in {"val", "valid", "validation"}:
        return "val"
    if normalized == "test":
        return "test"
    raise ValueError(f"Unsupported detection dataset split: {name!r}")


def read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON file {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return document


def validate_coco_document(document: dict[str, Any], path: Path) -> None:
    for key in ("images", "annotations", "categories"):
        if not isinstance(document.get(key), list):
            raise ValueError(f"COCO annotations {path} must contain a list named '{key}'")


def parse_coco_categories(document: dict[str, Any], path: Path) -> tuple[list[str], dict[int, int]]:
    categories: list[tuple[int, str]] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for item in document["categories"]:
        try:
            category_id = int(item["id"])
            name = str(item["name"]).strip()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid category in {path}: {item!r}") from error
        normalized_name = name.casefold()
        if not name or category_id in seen_ids or normalized_name in seen_names:
            raise ValueError(f"Empty or duplicate category in {path}: {item!r}")
        seen_ids.add(category_id)
        seen_names.add(normalized_name)
        categories.append((category_id, name))
    if not categories:
        raise ValueError(f"No categories found in {path}")

    categories.sort(key=lambda item: item[0])
    names = [name for _, name in categories]
    category_to_class = {category_id: class_id for class_id, (category_id, _) in enumerate(categories)}
    return names, category_to_class


def category_mapping(
    document: dict[str, Any], path: Path, expected_classes: list[str] | None
) -> tuple[list[str], dict[int, int]]:
    names, category_to_class = parse_coco_categories(document, path)
    if expected_classes is None:
        return names, category_to_class

    expected_by_name = {name.casefold(): class_id for class_id, name in enumerate(expected_classes)}
    actual_by_name = {name.casefold(): name for name in names}
    if set(expected_by_name) != set(actual_by_name):
        raise ValueError(
            f"Categories in {path} differ from the training split: "
            f"{names!r} != {expected_classes!r}"
        )

    categories = document["categories"]
    remapped: dict[int, int] = {}
    for item in categories:
        remapped[int(item["id"])] = expected_by_name[str(item["name"]).strip().casefold()]
    return expected_classes, remapped


def candidate_image_paths(
    dataset_root: Path, annotation_path: Path, split: str, file_name: str
) -> tuple[Path, ...]:
    relative = Path(file_name)
    return (
        dataset_root / relative,
        annotation_path.parent / relative,
        dataset_root / "images" / relative,
        dataset_root / "images" / split / relative,
        dataset_root / split / relative,
        dataset_root / split / "images" / relative,
    )


def resolve_coco_image(
    dataset_root: Path, annotation_path: Path, split: str, file_name: str
) -> Path:
    for candidate in candidate_image_paths(dataset_root, annotation_path, split, file_name):
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        f"COCO image was not found for {annotation_path}: {file_name!r}"
    )


def finite_float(value: Any, field_name: str, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field_name} in {context}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"Invalid {field_name} in {context}: {value!r}")
    return result


def clipped_box(
    x: float, y: float, width: float, height: float, image_width: int, image_height: int, context: str
) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Bounding box must have positive size in {context}")
    x1 = max(0.0, min(float(image_width), x))
    y1 = max(0.0, min(float(image_height), y))
    x2 = max(0.0, min(float(image_width), x + width))
    y2 = max(0.0, min(float(image_height), y + height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Empty bounding box after clipping in {context}")
    return x1, y1, x2, y2


def read_coco_split(
    dataset_root: Path,
    annotation_path: Path,
    split: str,
    expected_classes: list[str] | None,
) -> tuple[list[str], list[ImageRecord]]:
    document = read_json(annotation_path)
    validate_coco_document(document, annotation_path)
    classes, category_to_class = category_mapping(document, annotation_path, expected_classes)

    images_by_id: dict[int, ImageRecord] = {}
    for item in document["images"]:
        context = f"{annotation_path} image {item!r}"
        try:
            image_id = int(item["id"])
            file_name = str(item["file_name"]).strip()
            width = int(item["width"])
            height = int(item["height"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid image in {context}") from error
        if not file_name or image_id in images_by_id or width <= 0 or height <= 0:
            raise ValueError(f"Duplicate ID, empty filename, or invalid dimensions in {context}")
        source = resolve_coco_image(dataset_root, annotation_path, split, file_name)
        images_by_id[image_id] = ImageRecord(source, width, height)

    for item in document["annotations"]:
        context = f"{annotation_path} annotation {item!r}"
        try:
            image_id = int(item["image_id"])
            category_id = int(item["category_id"])
            bbox = item["bbox"]
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("bbox must contain four values")
            x, y, width, height = (
                finite_float(value, "bbox value", context) for value in bbox
            )
            iscrowd = int(item.get("iscrowd", 0))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid annotation in {context}") from error
        if image_id not in images_by_id or category_id not in category_to_class:
            raise ValueError(f"Unknown image/category ID in {context}")
        image = images_by_id[image_id]
        x1, y1, x2, y2 = clipped_box(
            x, y, width, height, image.width, image.height, context
        )
        image.detections.append(
            Detection(category_to_class[category_id], x1, y1, x2, y2, iscrowd)
        )

    return classes, list(images_by_id.values())


def find_annotation(root: Path, split: str) -> Path | None:
    names = (split,) if split != "val" else ("val", "valid")
    candidates = [root / "annotations" / f"instances_{name}.json" for name in names]
    candidates.extend(root / f"instances_{name}.json" for name in names)
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    discovered = sorted(root.glob("**/instances_*.json"))
    for candidate in discovered:
        suffix = candidate.stem.removeprefix("instances_").casefold()
        if suffix in names:
            return candidate
    return None


def load_coco(root: Path) -> DetectionDataset:
    annotation_paths: dict[str, Path] = {}
    for split in (*REQUIRED_SPLITS, *OPTIONAL_SPLITS):
        annotation_path = find_annotation(root, split)
        if annotation_path is not None:
            annotation_paths[split] = annotation_path
    missing = [split for split in REQUIRED_SPLITS if split not in annotation_paths]
    if missing:
        raise FileNotFoundError(
            f"COCO dataset is missing required annotation split(s): {', '.join(missing)}"
        )

    classes: list[str] | None = None
    splits: dict[str, list[ImageRecord]] = {}
    for split in (*REQUIRED_SPLITS, *OPTIONAL_SPLITS):
        annotation_path = annotation_paths.get(split)
        if annotation_path is None:
            continue
        classes, images = read_coco_split(root, annotation_path, split, classes)
        splits[split] = images
    assert classes is not None
    return DetectionDataset(classes, splits, str(root))


def load_rfdetr(root: Path) -> DetectionDataset:
    annotation_paths: dict[str, Path] = {}
    for split_directory in ("train", "valid", "test"):
        path = root / split_directory / "_annotations.coco.json"
        if path.is_file():
            annotation_paths[canonical_split(split_directory)] = path
    missing = [split for split in REQUIRED_SPLITS if split not in annotation_paths]
    if missing:
        raise FileNotFoundError(
            f"RF-DETR dataset is missing required split(s): {', '.join(missing)}"
        )

    classes: list[str] | None = None
    splits: dict[str, list[ImageRecord]] = {}
    for split in (*REQUIRED_SPLITS, *OPTIONAL_SPLITS):
        annotation_path = annotation_paths.get(split)
        if annotation_path is None:
            continue
        source_split = "valid" if split == "val" else split
        classes, images = read_coco_split(root, annotation_path, source_split, classes)
        splits[split] = images
    assert classes is not None
    return DetectionDataset(classes, splits, str(root))


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "PyYAML is required to read YOLO data.yaml; it is installed with the repository dependencies."
        ) from error
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read YOLO dataset YAML {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"YOLO dataset YAML must contain a mapping: {path}")
    return document


def parse_yolo_names(document: dict[str, Any], yaml_path: Path) -> list[str]:
    names = document.get("names")
    if isinstance(names, list):
        values = names
    elif isinstance(names, dict):
        try:
            ordered_keys = sorted(names, key=lambda key: int(key))
            numeric_keys = [int(key) for key in ordered_keys]
        except (TypeError, ValueError) as error:
            raise ValueError(f"YOLO names must use numeric class IDs: {yaml_path}") from error
        if numeric_keys != list(range(len(names))):
            raise ValueError(f"YOLO names must have contiguous IDs starting at zero: {yaml_path}")
        values = [names[key] for key in ordered_keys]
    else:
        raise ValueError(f"YOLO data.yaml must contain names as a list or mapping: {yaml_path}")

    result = [str(value).strip() for value in values]
    if not result or any(not value for value in result):
        raise ValueError(f"YOLO data.yaml contains an empty class name: {yaml_path}")
    if len({value.casefold() for value in result}) != len(result):
        raise ValueError(f"YOLO data.yaml contains duplicate class names: {yaml_path}")
    return result


def resolve_yaml_root(yaml_path: Path, document: dict[str, Any]) -> Path:
    configured = document.get("path")
    if configured is None:
        return yaml_path.parent.resolve()
    path = Path(str(configured)).expanduser()
    return (path if path.is_absolute() else yaml_path.parent / path).resolve()


def split_value(document: dict[str, Any], split: str) -> Any:
    if split in document:
        return document[split]
    if split == "val" and "valid" in document:
        return document["valid"]
    raise FileNotFoundError(f"YOLO data.yaml does not define the '{split}' split")


def split_paths(value: Any, split: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError(f"YOLO '{split}' split must be a path or list of paths")


def resolve_yolo_image_paths(root: Path, path_value: str, split: str) -> list[Path]:
    path = Path(path_value).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    if path.is_file():
        paths: list[Path] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            image_path = Path(line).expanduser()
            image_path = (image_path if image_path.is_absolute() else root / image_path).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"YOLO image list references a missing image: {image_path}")
            paths.append(image_path)
        return paths
    if not path.is_dir():
        raise FileNotFoundError(f"YOLO '{split}' image path does not exist: {path}")
    return sorted(
        (candidate.resolve() for candidate in path.rglob("*") if candidate.suffix.casefold() in IMAGE_EXTENSIONS),
        key=lambda candidate: candidate.as_posix().casefold(),
    )


def yolo_label_path(root: Path, image_path: Path, split: str) -> Path:
    try:
        relative = image_path.relative_to(root)
    except ValueError:
        return root / "labels" / split / f"{image_path.stem}.txt"
    parts = list(relative.parts)
    for index, part in enumerate(parts):
        if part.casefold() == "images":
            parts[index] = "labels"
            return root.joinpath(*parts).with_suffix(".txt")
    return root / "labels" / split / f"{image_path.stem}.txt"


def read_image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required to read image dimensions from YOLO datasets; it is installed with the repository dependencies."
        ) from error
    try:
        with Image.open(path) as image:
            width, height = image.size
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read image dimensions from {path}: {error}") from error
    if width <= 0 or height <= 0:
        raise ValueError(f"Image has invalid dimensions: {path}")
    return width, height


def read_yolo_labels(
    label_path: Path, image_path: Path, width: int, height: int, class_count: int
) -> list[Detection]:
    if not label_path.is_file():
        return []
    detections: list[Detection] = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Cannot read YOLO labels {label_path}: {error}") from error
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        values = line.split()
        context = f"{label_path}:{line_number}"
        if len(values) != 5:
            raise ValueError(
                f"Expected class x_center y_center width height in {context}; "
                "segmentation and extra fields are not supported"
            )
        try:
            class_id = int(values[0])
        except ValueError as error:
            raise ValueError(f"Invalid YOLO class ID in {context}: {values[0]!r}") from error
        if class_id < 0 or class_id >= class_count:
            raise ValueError(f"YOLO class ID is outside names in {context}: {class_id}")
        cx, cy, box_width, box_height = (
            finite_float(value, "YOLO box value", context) for value in values[1:]
        )
        x = (cx - box_width / 2.0) * width
        y = (cy - box_height / 2.0) * height
        pixel_width = box_width * width
        pixel_height = box_height * height
        x1, y1, x2, y2 = clipped_box(
            x, y, pixel_width, pixel_height, width, height, context
        )
        detections.append(Detection(class_id, x1, y1, x2, y2))
    return detections


def load_yolo(data_path: Path) -> DetectionDataset:
    yaml_path = data_path / "data.yaml" if data_path.is_dir() else data_path
    yaml_path = yaml_path.resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YOLO data.yaml does not exist: {yaml_path}")
    document = load_yaml(yaml_path)
    classes = parse_yolo_names(document, yaml_path)
    root = resolve_yaml_root(yaml_path, document)
    splits: dict[str, list[ImageRecord]] = {}

    for split in (*REQUIRED_SPLITS, *OPTIONAL_SPLITS):
        value = split_value(document, split) if split in REQUIRED_SPLITS else document.get(split)
        if value is None:
            continue
        image_paths: list[Path] = []
        for path_value in split_paths(value, split):
            image_paths.extend(resolve_yolo_image_paths(root, path_value, split))
        records: list[ImageRecord] = []
        seen_names: set[str] = set()
        seen_stems: set[str] = set()
        for image_path in image_paths:
            name_key = image_path.name.casefold()
            stem_key = image_path.stem.casefold()
            if name_key in seen_names or stem_key in seen_stems:
                raise ValueError(
                    f"YOLO split '{split}' contains duplicate image names or label stems: {image_path.name}"
                )
            seen_names.add(name_key)
            seen_stems.add(stem_key)
            width, height = read_image_size(image_path)
            records.append(
                ImageRecord(
                    image_path,
                    width,
                    height,
                    read_yolo_labels(
                        yolo_label_path(root, image_path, split),
                        image_path,
                        width,
                        height,
                        len(classes),
                    ),
                )
            )
        if not records:
            raise ValueError(f"YOLO split '{split}' contains no images")
        splits[split] = records

    return DetectionDataset(classes, splits, str(root))


def prepare_output(output: Path, input_path: Path) -> None:
    if output == input_path or output == input_path.parent:
        raise ValueError("Output must be different from the input dataset")
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"Output path must be a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(f"Output directory must be new or empty: {output}")
    else:
        output.mkdir(parents=True)


def place_image(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, target)
        return
    try:
        os.link(source, target)
    except OSError as error:
        raise OSError(
            f"Cannot hard-link {source} to {target}. Use --image-mode copy for a cross-volume export."
        ) from error


def output_image_name(record: ImageRecord, names: set[str], stems: set[str]) -> str:
    image_name = record.source.name
    name_key = image_name.casefold()
    stem_key = record.source.stem.casefold()
    if name_key in names:
        raise ValueError(f"Output split contains duplicate image filename: {image_name}")
    if stem_key in stems:
        raise ValueError(
            f"Output split contains filenames that map to the same label stem: {record.source.stem}.txt"
        )
    names.add(name_key)
    stems.add(stem_key)
    return image_name


def write_coco(dataset: DetectionDataset, output: Path, image_mode: str) -> None:
    annotations_root = output / "annotations"
    annotations_root.mkdir(parents=True, exist_ok=True)
    categories = [
        {"id": class_id + 1, "name": name, "supercategory": ""}
        for class_id, name in enumerate(dataset.classes)
    ]
    counts: dict[str, dict[str, int]] = {}
    for split, records in dataset.splits.items():
        image_root = output / "images" / split
        image_root.mkdir(parents=True, exist_ok=True)
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        names: set[str] = set()
        stems: set[str] = set()
        for image_id, record in enumerate(records, start=1):
            image_name = output_image_name(record, names, stems)
            place_image(record.source, image_root / image_name, image_mode)
            images.append(
                {
                    "id": image_id,
                    "file_name": f"images/{split}/{image_name}",
                    "width": record.width,
                    "height": record.height,
                }
            )
            for detection in record.detections:
                width = detection.x2 - detection.x1
                height = detection.y2 - detection.y1
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": detection.class_id + 1,
                        "bbox": [detection.x1, detection.y1, width, height],
                        "area": width * height,
                        "iscrowd": detection.iscrowd,
                    }
                )
        document = {"images": images, "annotations": annotations, "categories": categories}
        (annotations_root / f"instances_{split}.json").write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
        counts[split] = {"images": len(images), "annotations": len(annotations)}
    write_metadata(dataset, output, image_mode, counts)


def write_rfdetr(dataset: DetectionDataset, output: Path, image_mode: str) -> None:
    categories = [
        {"id": class_id + 1, "name": name, "supercategory": ""}
        for class_id, name in enumerate(dataset.classes)
    ]
    counts: dict[str, dict[str, int]] = {}
    for split, records in dataset.splits.items():
        destination_split = "valid" if split == "val" else split
        split_root = output / destination_split
        split_root.mkdir(parents=True, exist_ok=True)
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        names: set[str] = set()
        stems: set[str] = set()
        for image_id, record in enumerate(records, start=1):
            image_name = output_image_name(record, names, stems)
            place_image(record.source, split_root / image_name, image_mode)
            images.append(
                {
                    "id": image_id,
                    "file_name": image_name,
                    "width": record.width,
                    "height": record.height,
                }
            )
            for detection in record.detections:
                width = detection.x2 - detection.x1
                height = detection.y2 - detection.y1
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": detection.class_id + 1,
                        "bbox": [detection.x1, detection.y1, width, height],
                        "area": width * height,
                        "iscrowd": detection.iscrowd,
                    }
                )
        document = {"images": images, "annotations": annotations, "categories": categories}
        (split_root / "_annotations.coco.json").write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
        counts[split] = {"images": len(images), "annotations": len(annotations)}
    write_metadata(dataset, output, image_mode, counts)


def yaml_quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_yolo(dataset: DetectionDataset, output: Path, image_mode: str) -> None:
    counts: dict[str, dict[str, int]] = {}
    splits = list(dataset.splits)
    for split, records in dataset.splits.items():
        image_root = output / "images" / split
        label_root = output / "labels" / split
        image_root.mkdir(parents=True, exist_ok=True)
        label_root.mkdir(parents=True, exist_ok=True)
        names: set[str] = set()
        stems: set[str] = set()
        annotation_count = 0
        for record in records:
            image_name = output_image_name(record, names, stems)
            place_image(record.source, image_root / image_name, image_mode)
            label_lines = []
            for detection in record.detections:
                center_x = ((detection.x1 + detection.x2) / 2.0) / record.width
                center_y = ((detection.y1 + detection.y2) / 2.0) / record.height
                box_width = (detection.x2 - detection.x1) / record.width
                box_height = (detection.y2 - detection.y1) / record.height
                label_lines.append(
                    f"{detection.class_id} {center_x:.9g} {center_y:.9g} "
                    f"{box_width:.9g} {box_height:.9g}"
                )
            (label_root / f"{record.source.stem}.txt").write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8"
            )
            annotation_count += len(label_lines)
        counts[split] = {"images": len(records), "annotations": annotation_count}

    yaml_lines = [f"{split}: images/{split}" for split in splits]
    yaml_lines.append("names:")
    yaml_lines.extend(f"  {class_id}: {yaml_quote(name)}" for class_id, name in enumerate(dataset.classes))
    (output / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    write_metadata(dataset, output, image_mode, counts)


def write_metadata(
    dataset: DetectionDataset, output: Path, image_mode: str, counts: dict[str, dict[str, int]]
) -> None:
    (output / "conversion_metadata.json").write_text(
        json.dumps(
            {
                "source": dataset.source,
                "classes": dataset.classes,
                "image_mode": image_mode,
                "split_counts": counts,
                "split_image_counts": {split: values["images"] for split, values in counts.items()},
                "splits": counts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser(
    default_input_format: str | None = None, default_output_format: str | None = None
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-format",
        choices=FORMATS,
        default=default_input_format,
        required=default_input_format is None,
        help="Input dataset layout",
    )
    parser.add_argument(
        "--output-format",
        choices=FORMATS,
        default=default_output_format,
        required=default_output_format is None,
        help="Output dataset layout",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Input dataset root, or YOLO data.yaml when --input-format yolo",
    )
    parser.add_argument("--output", required=True, type=Path, help="New or empty output dataset directory")
    parser.add_argument(
        "--image-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Hard links avoid duplicating images on the same volume; use copy across volumes.",
    )
    return parser


def load_dataset(input_format: str, data: Path) -> DetectionDataset:
    if not data.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {data}")
    if input_format == "coco":
        if not data.is_dir():
            raise ValueError("COCO input must be a dataset directory")
        return load_coco(data)
    if input_format == "rfdetr":
        if not data.is_dir():
            raise ValueError("RF-DETR input must be a dataset directory")
        return load_rfdetr(data)
    return load_yolo(data)


def main(
    argv: list[str] | None = None,
    *,
    default_input_format: str | None = None,
    default_output_format: str | None = None,
) -> None:
    args = build_parser(default_input_format, default_output_format).parse_args(argv)
    data = args.data.expanduser().resolve()
    output = args.output.expanduser().resolve()
    input_root = data.parent if args.input_format == "yolo" and data.is_file() else data
    dataset = load_dataset(args.input_format, data)
    prepare_output(output, input_root)
    if args.output_format == "coco":
        write_coco(dataset, output, args.image_mode)
    elif args.output_format == "rfdetr":
        write_rfdetr(dataset, output, args.image_mode)
    else:
        write_yolo(dataset, output, args.image_mode)
    counts = {split: len(records) for split, records in dataset.splits.items()}
    print(
        f"Converted {args.input_format} dataset to {args.output_format}; "
        f"classes={dataset.classes}; split images={counts}; output={output}"
    )


if __name__ == "__main__":
    main()
