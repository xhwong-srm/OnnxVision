from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image

from ...domain.datasets import (
    BoundingBox,
    ClassificationDataset,
    ClassificationSample,
    DatasetFormat,
    DetectionAnnotation,
    DetectionDataset,
    DetectionSample,
    ImageRef,
    MaterializationMode,
    Split,
)
from ...domain.errors import DatasetFormatError

IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


def image_ref(path: Path, *, source_id: str | None = None) -> ImageRef:
    try:
        with Image.open(path) as image:
            width, height = image.size
    except (OSError, ValueError) as error:
        raise DatasetFormatError(f"cannot read image {path}: {error}") from error
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise DatasetFormatError(f"cannot hash image {path}: {error}") from error
    return ImageRef(path.resolve(), width, height, digest.hexdigest(), source_id)


def canonical_split(value: str) -> Split:
    normalized = value.casefold()
    if normalized in {"valid", "validation"}:
        normalized = "val"
    try:
        return Split(normalized)
    except ValueError as error:
        raise DatasetFormatError(f"unsupported split: {value}") from error


def safe_image_path(root: Path, value: str, split: str | None = None) -> Path:
    relative = Path(value)
    candidates = [root / relative]
    if split:
        candidates.extend((root / "images" / split / relative.name, root / split / relative.name))
    candidates.append(root / relative.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise DatasetFormatError(f"image {value!r} was not found below {root}")


def materialize(source: Path, target: Path, mode: MaterializationMode) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == MaterializationMode.COPY:
        shutil.copy2(source, target)
        return
    try:
        os.link(source, target)
    except OSError as error:
        raise DatasetFormatError(
            f"cannot hard-link {source} to {target}; use materialization=copy across volumes"
        ) from error


def unique_name(source: Path, used: set[str]) -> str:
    name = source.name
    candidate = name
    index = 1
    while candidate.casefold() in used:
        candidate = f"{source.stem}_{index}{source.suffix}"
        index += 1
    used.add(candidate.casefold())
    return candidate


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetFormatError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise DatasetFormatError(f"JSON document must be an object: {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise DatasetFormatError("PyYAML is required for YOLO datasets") from error
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise DatasetFormatError(f"cannot read YAML {path}: {error}") from error
    if not isinstance(value, dict):
        raise DatasetFormatError(f"dataset YAML must be an object: {path}")
    return value


def class_names(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        names = tuple(str(item).strip() for item in value)
    elif isinstance(value, dict):
        try:
            items = sorted((int(key), str(name).strip()) for key, name in value.items())
        except (TypeError, ValueError) as error:
            raise DatasetFormatError("class names mapping must use integer IDs") from error
        if [index for index, _ in items] != list(range(len(items))):
            raise DatasetFormatError("class IDs must be contiguous and zero-based")
        names = tuple(name for _, name in items)
    else:
        raise DatasetFormatError("dataset must define class names")
    if not names or any(not name for name in names):
        raise DatasetFormatError("dataset class names must be non-empty")
    if len({name.casefold() for name in names}) != len(names):
        raise DatasetFormatError("dataset class names must be unique")
    return names


def _coco_split_path(root: Path, split: str) -> Path:
    path = root / "annotations" / f"instances_{split}.json"
    if path.is_file():
        return path
    raise DatasetFormatError(f"missing COCO annotation file: {path}")


def _read_coco_document(root: Path, split: str, annotation_path: Path) -> tuple[tuple[str, ...], list[DetectionSample]]:
    document = load_json(annotation_path)
    for key in ("images", "annotations", "categories"):
        if not isinstance(document.get(key), list):
            raise DatasetFormatError(f"COCO document must contain list '{key}': {annotation_path}")
    categories = []
    category_map: dict[int, int] = {}
    for item in document["categories"]:
        try:
            category_id = int(item["id"])
            name = str(item["name"]).strip()
        except (KeyError, TypeError, ValueError) as error:
            raise DatasetFormatError(f"invalid COCO category: {item!r}") from error
        if not name or name.casefold() in {name.casefold() for name in categories}:
            raise DatasetFormatError(f"duplicate or empty COCO category: {item!r}")
        category_map[category_id] = len(categories)
        categories.append(name)
    by_image: dict[int, list[DetectionAnnotation]] = {}
    for item in document["annotations"]:
        try:
            image_id = int(item["image_id"])
            category_id = category_map[int(item["category_id"])]
            x, y, width, height = (float(value) for value in item["bbox"][:4])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise DatasetFormatError(f"invalid COCO annotation: {item!r}") from error
        by_image.setdefault(image_id, []).append(
            DetectionAnnotation(category_id, BoundingBox(x, y, x + width, y + height), int(item.get("iscrowd", 0)))
        )
    samples: list[DetectionSample] = []
    for item in document["images"]:
        try:
            image_id = int(item["id"])
            path = safe_image_path(root, str(item["file_name"]), split)
            ref = image_ref(path, source_id=str(item.get("id", path.stem)))
        except (KeyError, TypeError, ValueError) as error:
            raise DatasetFormatError(f"invalid COCO image: {item!r}") from error
        samples.append(DetectionSample(ref, tuple(by_image.get(image_id, ())), Split(split), ("image", str(path.resolve()).casefold())))
    return tuple(categories), samples


def load_coco(root: Path, *, rfdetr: bool = False) -> DetectionDataset:
    split_paths = {}
    for split in ("train", "val", "test"):
        candidate = (root / ("valid" if rfdetr and split == "val" else split) / "_annotations.coco.json") if rfdetr else root / "annotations" / f"instances_{split}.json"
        if candidate.is_file():
            split_paths[split] = candidate
    if "train" not in split_paths or "val" not in split_paths:
        raise DatasetFormatError(f"{root} must contain train and val annotations")
    classes: tuple[str, ...] | None = None
    samples: list[DetectionSample] = []
    for split, annotation_path in split_paths.items():
        names, loaded = _read_rfdetr_document(root, split, annotation_path) if rfdetr else _read_coco_document(root, split, annotation_path)
        if classes is None:
            classes = names
        elif tuple(name.casefold() for name in names) != tuple(name.casefold() for name in classes):
            if {name.casefold() for name in names} != {name.casefold() for name in classes}:
                raise DatasetFormatError(f"class names differ between detection splits: {annotation_path}")
            remap = {index: next(global_index for global_index, value in enumerate(classes) if value.casefold() == name.casefold()) for index, name in enumerate(names)}
            loaded = tuple(replace(sample, annotations=tuple(replace(annotation, class_id=remap[annotation.class_id]) for annotation in sample.annotations)) for sample in loaded)
        samples.extend(loaded)
    return DetectionDataset(classes or (), tuple(samples), root.resolve(), DatasetFormat.RFDETR if rfdetr else DatasetFormat.COCO)


def _read_rfdetr_document(root: Path, split: str, annotation_path: Path) -> tuple[tuple[str, ...], list[DetectionSample]]:
    document = load_json(annotation_path)
    # RF-DETR uses the COCO schema; its images are relative to the split directory.
    split_root = root / ("valid" if split == "val" else split)
    rewritten = dict(document)
    rewritten["images"] = [dict(item, file_name=str(split_root / str(item["file_name"]))) for item in document.get("images", [])]
    temporary_root = root
    names = []
    category_map = {}
    for item in document.get("categories", []):
        category_map[int(item["id"])] = len(names)
        names.append(str(item["name"]).strip())
    by_image: dict[int, list[DetectionAnnotation]] = {}
    for item in document.get("annotations", []):
        x, y, width, height = (float(value) for value in item["bbox"][:4])
        by_image.setdefault(int(item["image_id"]), []).append(DetectionAnnotation(category_map[int(item["category_id"])], BoundingBox(x, y, x + width, y + height), int(item.get("iscrowd", 0))))
    samples = []
    for item in document.get("images", []):
        path = safe_image_path(temporary_root, str(split_root / str(item["file_name"])), split)
        ref = image_ref(path, source_id=str(item["id"]))
        samples.append(DetectionSample(ref, tuple(by_image.get(int(item["id"]), ())), Split(split), ("image", str(path.resolve()).casefold())))
    return tuple(names), samples


def load_yolo(data: Path) -> DetectionDataset:
    yaml_path = data if data.is_file() else data / "data.yaml"
    document = load_yaml(yaml_path)
    names = class_names(document.get("names"))
    root = yaml_path.parent / str(document.get("path", "."))
    root = root.resolve()
    samples: list[DetectionSample] = []
    for split in ("train", "val", "test"):
        if split not in document:
            continue
        value = document[split]
        paths = [value] if isinstance(value, str) else value if isinstance(value, list) else []
        if not paths:
            raise DatasetFormatError(f"YOLO split {split} must be a path or list of paths")
        for path_value in paths:
            image_root = root / str(path_value)
            if not image_root.is_dir():
                raise DatasetFormatError(f"YOLO image directory does not exist: {image_root}")
            for image_path in sorted((path for path in image_root.rglob("*") if path.suffix.casefold() in IMAGE_EXTENSIONS), key=lambda path: path.as_posix().casefold()):
                ref = image_ref(image_path, source_id=str(image_path.relative_to(root)))
                label_path = root / "labels" / split / f"{image_path.stem}.txt"
                if not label_path.is_file():
                    label_path = image_path.with_suffix(".txt")
                annotations = []
                if label_path.is_file():
                    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                        values = line.split()
                        if len(values) != 5:
                            raise DatasetFormatError(f"invalid YOLO label {label_path}:{line_number}")
                        class_id, cx, cy, width, height = (float(value) for value in values)
                        x1 = (cx - width / 2) * ref.width
                        y1 = (cy - height / 2) * ref.height
                        x2 = (cx + width / 2) * ref.width
                        y2 = (cy + height / 2) * ref.height
                        if int(class_id) != class_id or not 0 <= int(class_id) < len(names):
                            raise DatasetFormatError(f"invalid YOLO class ID {class_id}: {label_path}:{line_number}")
                        annotations.append(DetectionAnnotation(int(class_id), BoundingBox(x1, y1, x2, y2)))
                samples.append(DetectionSample(ref, tuple(annotations), Split(split), ("image", str(image_path.resolve()).casefold())))
    return DetectionDataset(names, tuple(samples), yaml_path.resolve(), DatasetFormat.YOLO)


def load_neurocle(root: Path) -> DetectionDataset:
    json_paths = [path for path in root.glob("*.json") if path.is_file()]
    json_path = next((path for path in json_paths if path.name.casefold() == "neurocle_labeling.json"), None)
    if json_path is None:
        for candidate in json_paths:
            document = load_json(candidate)
            if "data" in document and "classes" in document:
                json_path = candidate
                break
    if json_path is None:
        raise DatasetFormatError(f"no Neurocle labeling JSON found below {root}")
    document = load_json(json_path)
    classes = tuple(str(item["name"]).strip() for item in document.get("classes", []))
    if not classes:
        raise DatasetFormatError(f"Neurocle JSON has no classes: {json_path}")
    class_map = {name.casefold(): index for index, name in enumerate(classes)}
    temporary = None
    archive = next(iter(root.glob("*.zip")), None)
    image_root = root / "images"
    if archive is not None and not image_root.is_dir():
        import tempfile
        temporary = tempfile.TemporaryDirectory(prefix="seal-neurocle-")
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(temporary.name)
        image_root = Path(temporary.name)
    samples = []
    for item in document.get("data", []):
        file_name = str(item.get("fileName", ""))
        path = safe_image_path(image_root, file_name)
        ref = image_ref(path, source_id=file_name)
        regions = []
        for region in item.get("regionLabel", []):
            if str(region.get("type", "Rect")).casefold() != "rect":
                raise DatasetFormatError("Neurocle conversion supports rectangular regions only")
            name = str(region.get("className", "")).strip()
            if name.casefold() not in class_map:
                raise DatasetFormatError(f"Neurocle region references unknown class {name!r}")
            regions.append(DetectionAnnotation(class_map[name.casefold()], BoundingBox(float(region["x"]), float(region["y"]), float(region["x"]) + float(region["width"]), float(region["y"]) + float(region["height"]))))
        split = "train" if str(item.get("set", "train")).casefold() == "train" else "test"
        samples.append(DetectionSample(ref, tuple(regions), Split(split), ("image", file_name.casefold())))
    dataset = DetectionDataset(classes, tuple(samples), root.resolve(), DatasetFormat.NEUROCLE)
    if temporary is not None:
        object.__setattr__(dataset, "metadata", {"temporary_directory": temporary})
    return dataset


def load_image_folder(root: Path) -> ClassificationDataset:
    split_dirs = [split for split in ("train", "val", "test") if (root / split).is_dir()]
    locations = [(Split(split), root / split) for split in split_dirs] or [(None, root)]
    class_names_set: dict[str, str] = {}
    for _, location in locations:
        for path in location.iterdir():
            if path.is_dir():
                class_names_set.setdefault(path.name.casefold(), path.name)
    classes = tuple(class_names_set[key] for key in sorted(class_names_set))
    class_map = {name.casefold(): index for index, name in enumerate(classes)}
    samples = []
    for split, location in locations:
        for class_dir in sorted((path for path in location.iterdir() if path.is_dir()), key=lambda path: path.name.casefold()):
            for path in sorted((path for path in class_dir.rglob("*") if path.suffix.casefold() in IMAGE_EXTENSIONS), key=lambda path: path.as_posix().casefold()):
                samples.append(ClassificationSample(image_ref(path), class_map[class_dir.name.casefold()], class_map and classes[class_map[class_dir.name.casefold()]], split, ("image", str(path.relative_to(root)).casefold())))
    return ClassificationDataset(classes, tuple(samples), root.resolve(), DatasetFormat.IMAGE_FOLDER)


def detect_format(path: Path) -> DatasetFormat:
    if not path.exists():
        raise DatasetFormatError(f"dataset does not exist: {path}")
    if path.is_file() and path.suffix.casefold() in {".yaml", ".yml"}:
        return DatasetFormat.YOLO
    root = path
    candidates = []
    if (root / "data.yaml").is_file():
        candidates.append(DatasetFormat.YOLO)
    if (root / "annotations" / "instances_train.json").is_file() and (root / "annotations" / "instances_val.json").is_file():
        candidates.append(DatasetFormat.COCO)
    if (root / "train" / "_annotations.coco.json").is_file() and (root / "valid" / "_annotations.coco.json").is_file():
        candidates.append(DatasetFormat.RFDETR)
    if root.is_dir():
        for json_path in root.glob("*.json"):
            try:
                document = load_json(json_path)
            except DatasetFormatError:
                continue
            if isinstance(document.get("classes"), (list, dict)) and isinstance(document.get("data"), list):
                candidates.append(DatasetFormat.NEUROCLE)
                break
    if not candidates and root.is_dir():
        candidates.append(DatasetFormat.IMAGE_FOLDER)
    if len(set(candidates)) != 1:
        names = ", ".join(item.value for item in sorted(set(candidates), key=lambda item: item.value))
        raise DatasetFormatError(f"dataset format is ambiguous or unknown for {path}: {names or 'none'}")
    return candidates[0]


def load_dataset(path: Path, format: DatasetFormat | None = None):
    selected = format or detect_format(path)
    root = path.parent if path.is_file() else path
    if selected == DatasetFormat.COCO:
        return load_coco(root)
    if selected == DatasetFormat.RFDETR:
        return load_coco(root, rfdetr=True)
    if selected == DatasetFormat.YOLO:
        return load_yolo(path)
    if selected == DatasetFormat.NEUROCLE:
        return load_neurocle(root)
    return load_image_folder(root)


def write_image_folder(dataset: ClassificationDataset, output: Path, mode: MaterializationMode) -> None:
    used: set[str] = set()
    for sample in dataset.samples:
        prefix = sample.split.value if sample.split is not None else ""
        target_dir = output / prefix / sample.class_name if prefix else output / sample.class_name
        name = unique_name(sample.image.path, used)
        materialize(sample.image.path, target_dir / name, mode)


def write_coco(dataset: DetectionDataset, output: Path, mode: MaterializationMode, *, rfdetr: bool = False) -> None:
    classes = [{"id": index + 1, "name": name, "supercategory": ""} for index, name in enumerate(dataset.classes)]
    by_split: dict[Split, list[DetectionSample]] = {split: [sample for sample in dataset.samples if sample.split == split] for split in Split}
    used_by_split: dict[Split, set[str]] = {split: set() for split in Split}
    for split, samples in by_split.items():
        if not samples:
            continue
        split_name = "valid" if rfdetr and split == Split.VAL else split.value
        split_root = output / split_name if rfdetr else output / "images" / split.value
        annotation_root = split_root if rfdetr else output / "annotations"
        images = []
        annotations = []
        for image_id, sample in enumerate(samples, start=1):
            name = unique_name(sample.image.path, used_by_split[split])
            materialize(sample.image.path, split_root / name, mode)
            images.append({"id": image_id, "file_name": name if rfdetr else f"images/{split.value}/{name}", "width": sample.image.width, "height": sample.image.height})
            for annotation in sample.annotations:
                annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": annotation.class_id + 1, "bbox": [annotation.box.x1, annotation.box.y1, annotation.box.width, annotation.box.height], "area": annotation.box.area, "iscrowd": annotation.iscrowd})
        annotation_root.mkdir(parents=True, exist_ok=True)
        name = "_annotations.coco.json" if rfdetr else f"instances_{split.value}.json"
        (annotation_root / name).write_text(json.dumps({"images": images, "annotations": annotations, "categories": classes}, indent=2) + "\n", encoding="utf-8")


def write_yolo(dataset: DetectionDataset, output: Path, mode: MaterializationMode) -> None:
    used_by_split = {split: set() for split in Split}
    for split in Split:
        samples = [sample for sample in dataset.samples if sample.split == split]
        for sample in samples:
            name = unique_name(sample.image.path, used_by_split[split])
            materialize(sample.image.path, output / "images" / split.value / name, mode)
            labels = []
            for annotation in sample.annotations:
                box = annotation.box
                labels.append(f"{annotation.class_id} {(box.x1 + box.x2) / 2 / sample.image.width:.9g} {(box.y1 + box.y2) / 2 / sample.image.height:.9g} {box.width / sample.image.width:.9g} {box.height / sample.image.height:.9g}")
            (output / "labels" / split.value / f"{Path(name).stem}.txt").parent.mkdir(parents=True, exist_ok=True)
            (output / "labels" / split.value / f"{Path(name).stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    (output / "data.yaml").write_text("\n".join(f"{split.value}: images/{split.value}" for split in Split) + "\nnames:\n" + "\n".join(f"  {index}: {json.dumps(name)}" for index, name in enumerate(dataset.classes)) + "\n", encoding="utf-8")


def write_neurocle(dataset: DetectionDataset, output: Path, mode: MaterializationMode) -> None:
    used: set[str] = set()
    data = []
    for sample in dataset.samples:
        name = unique_name(sample.image.path, used)
        materialize(sample.image.path, output / "images" / name, mode)
        split = "train" if sample.split == Split.VAL else (sample.split.value if sample.split else "train")
        data.append({"fileName": name, "set": split, "classLabel": "", "width": sample.image.width, "height": sample.image.height, "regionLabel": [{"className": dataset.classes[item.class_id], "type": "Rect", "x": item.box.x1, "y": item.box.y1, "width": item.box.width, "height": item.box.height} for item in sample.annotations]})
    document = {"label_type": "obd", "source": "labelset", "version": "4.4.1.6", "classes": [{"name": name} for name in dataset.classes], "data": data}
    (output / "neurocle_labeling.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def write_dataset(dataset, output: Path, format: DatasetFormat, mode: MaterializationMode) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if isinstance(dataset, ClassificationDataset):
        if format != DatasetFormat.IMAGE_FOLDER:
            raise DatasetFormatError("classification datasets only support image-folder output")
        write_image_folder(dataset, output, mode)
        return
    if format == DatasetFormat.COCO:
        write_coco(dataset, output, mode)
    elif format == DatasetFormat.RFDETR:
        write_coco(dataset, output, mode, rfdetr=True)
    elif format == DatasetFormat.YOLO:
        write_yolo(dataset, output, mode)
    elif format == DatasetFormat.NEUROCLE:
        write_neurocle(dataset, output, mode)
    else:
        raise DatasetFormatError(f"unsupported output format: {format}")
