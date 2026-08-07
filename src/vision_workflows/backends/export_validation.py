from __future__ import annotations

from pathlib import Path
from typing import Any

from ..workflows.context import optional_import


_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def native_validation_metrics(values: Any) -> dict[str, Any]:
    raw = values.get("metrics", values) if isinstance(values, dict) else getattr(values, "results_dict", {})
    if not isinstance(raw, dict):
        return {"native_metric_count": 0}
    result: dict[str, Any] = {}
    for key, value in raw.items():
        scalar = value.item() if hasattr(value, "item") else value
        if not isinstance(scalar, (int, float)):
            continue
        result[f"native_{key}"] = scalar
        normalized = str(key).casefold().replace("_", "-")
        if "map50-95" in normalized:
            result.setdefault("native_map50_95", scalar)
        elif "map50" in normalized:
            result.setdefault("native_map50", scalar)
        if "precision" in normalized:
            result.setdefault("native_precision", scalar)
        if "recall" in normalized:
            result.setdefault("native_recall", scalar)
        if "accuracy" in normalized:
            result.setdefault("native_accuracy", scalar)
    result["native_metric_count"] = len(raw)
    return result


def validate_classification_wrappers(
    outputs: dict[str, Path],
    data_root: Path,
    *,
    classes: list[str],
    image_size: int,
    batch_size: int | None,
    reference_probabilities: Any | None = None,
) -> dict[str, Any]:
    np = optional_import("numpy")
    ort = optional_import("onnxruntime")
    torchvision = optional_import("torchvision")
    dataset = torchvision.datasets.ImageFolder(str(data_root / "val"))
    if dataset.classes != classes:
        raise ValueError("dataset classes differ from checkpoint classes")
    if len(dataset) == 0:
        raise ValueError("classification validation split is empty")

    labels = np.asarray([label for _, label in dataset.samples], dtype=np.int64)
    reference_predictions = reference_probabilities.argmax(axis=1) if reference_probabilities is not None else None
    result: dict[str, Any] = {"validation_split": "val", "validation_images": len(dataset)}
    probabilities_by_variant: dict[str, Any] = {}
    for variant, output in outputs.items():
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        run_batch = batch_size or 32
        probabilities: list[Any] = []
        for start in range(0, len(dataset.samples), run_batch):
            current = dataset.samples[start:start + run_batch]
            raw_images = [_classification_raw_image(path, variant, np) for path, _ in current]
            actual_batch = len(raw_images)
            while len(raw_images) < run_batch:
                raw_images.append(np.zeros_like(raw_images[0]))
            values = session.run(None, {input_name: np.stack(raw_images, axis=0)})
            batch_probabilities = np.asarray(values[0])
            if batch_probabilities.shape != (run_batch, len(classes)):
                raise ValueError(
                    f"{variant} ONNX validation output shape mismatch: "
                    f"{batch_probabilities.shape}; expected {(run_batch, len(classes))}"
                )
            probabilities.append(batch_probabilities[:actual_batch])
        variant_probabilities = np.concatenate(probabilities, axis=0)
        probabilities_by_variant[variant] = variant_probabilities
        predictions = variant_probabilities.argmax(axis=1)
        correct = int(np.sum(predictions == labels))
        result.update({
            f"{variant}_accuracy": correct / len(labels),
            f"{variant}_loss": float(np.mean(-np.log(np.clip(variant_probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)))),
            f"{variant}_correct": correct,
            f"{variant}_images": len(labels),
        })
        if reference_probabilities is not None and reference_predictions is not None:
            probability_error = np.abs(variant_probabilities - reference_probabilities)
            result.update({
                f"{variant}_native_agreement": float(np.mean(predictions == reference_predictions)),
                f"{variant}_native_probability_mae": float(np.mean(probability_error)),
                f"{variant}_native_max_probability_error": float(np.max(probability_error)),
            })

    bw8 = probabilities_by_variant["bw8"]
    c24 = probabilities_by_variant["c24"]
    result.update({
        "bw8_c24_agreement": float(np.mean(bw8.argmax(axis=1) == c24.argmax(axis=1))),
        "bw8_c24_probability_mae": float(np.mean(np.abs(bw8 - c24))),
        "bw8_c24_max_probability_error": float(np.max(np.abs(bw8 - c24))),
    })
    return result


def _classification_raw_image(path: str, variant: str, np: Any) -> Any:
    image_module = optional_import("PIL.Image")
    image = image_module.open(path)
    try:
        if variant == "bw8":
            return np.asarray(image.convert("L"), dtype=np.uint8)
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return np.ascontiguousarray(rgb[..., ::-1])
    finally:
        image.close()


def validate_detection_wrappers(
    outputs: dict[str, Path],
    data_yaml: Path,
    *,
    class_count: int,
    image_size: int,
    batch_size: int | None,
    confidence: float = 0.0,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate standardized detection wrappers on the YOLO validation split.

    The returned mAP50/precision/recall values use the same normalized XYXY
    contract for both variants. Native provider metrics are intentionally kept
    separate because Ultralytics and LibreYOLO expose different metric keys.
    """
    if not 0.0 <= confidence < 1.0:
        raise ValueError("confidence must be in [0, 1)")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")

    np = optional_import("numpy")
    ort = optional_import("onnxruntime")
    samples = _yolo_validation_samples(data_yaml)
    if not samples:
        raise ValueError("detection validation split is empty")

    predictions_by_variant: dict[str, list[tuple[Any, Any, Any]]] = {}
    result: dict[str, Any] = {"validation_split": "val", "validation_images": len(samples)}
    for variant, output in outputs.items():
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        run_batch = batch_size or 16
        predictions: list[tuple[Any, Any, Any]] = []
        for start in range(0, len(samples), run_batch):
            current = samples[start:start + run_batch]
            raw_images = [_raw_image(path, variant, np) for path, _ in current]
            actual_batch = len(raw_images)
            while len(raw_images) < run_batch:
                raw_images.append(np.zeros_like(raw_images[0]))
            values = session.run(None, {input_name: np.stack(raw_images, axis=0)})
            boxes, scores, class_ids = (np.asarray(value) for value in values[:3])
            if boxes.shape[0] != run_batch or scores.shape[0] != run_batch or class_ids.shape[0] != run_batch:
                raise ValueError(f"{variant} ONNX validation batch dimension mismatch")
            for index in range(actual_batch):
                predictions.append((boxes[index], scores[index], class_ids[index]))
        predictions_by_variant[variant] = predictions
        result.update(_detection_metrics(
            variant,
            predictions,
            [annotations for _, annotations in samples],
            class_count=class_count,
            confidence=confidence,
            iou_threshold=iou_threshold,
            np=np,
        ))

    result.update(_variant_agreement(
        predictions_by_variant["bw8"],
        predictions_by_variant["c24"],
        confidence=confidence,
        iou_threshold=iou_threshold,
        np=np,
    ))
    return result


def _yolo_validation_samples(data_yaml: Path) -> list[tuple[Path, list[tuple[int, tuple[float, float, float, float]]]]]:
    yaml = optional_import("yaml")
    data_yaml = data_yaml.expanduser().resolve()
    document = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "val" not in document:
        raise ValueError(f"dataset YAML has no val split: {data_yaml}")
    root = Path(document.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    value = document["val"]
    if isinstance(value, list):
        image_paths = [Path(item) for item in value]
    else:
        image_paths = [Path(str(value))]
    resolved_images: list[Path] = []
    for value_path in image_paths:
        path = value_path if value_path.is_absolute() else root / value_path
        if path.suffix.casefold() == ".txt":
            resolved_images.extend(_read_image_list(path, root))
        elif path.is_dir():
            resolved_images.extend(
                item for item in sorted(path.rglob("*"))
                if item.is_file() and item.suffix.casefold() in _IMAGE_SUFFIXES
            )
        elif path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES:
            resolved_images.append(path)
    return [(path, _read_yolo_labels(_label_path(path))) for path in resolved_images]


def _read_image_list(path: Path, root: Path) -> list[Path]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value:
            target = Path(value)
            result.append(target if target.is_absolute() else root / target)
    return result


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = next(index for index, part in enumerate(parts) if part.casefold() == "images")
    except StopIteration:
        return image_path.parent / "labels" / f"{image_path.stem}.txt"
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _read_yolo_labels(path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    if not path.is_file():
        return []
    annotations = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"invalid YOLO label at {path}:{line_number}")
        class_id = int(fields[0])
        x_center, y_center, width, height = (float(value) for value in fields[1:5])
        annotations.append((class_id, (
            x_center - width / 2.0,
            y_center - height / 2.0,
            x_center + width / 2.0,
            y_center + height / 2.0,
        )))
    return annotations


def _raw_image(path: Path, variant: str, np: Any) -> Any:
    image_module = optional_import("PIL.Image")
    image = image_module.open(path)
    try:
        if variant == "bw8":
            return np.asarray(image.convert("L"), dtype=np.uint8)
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return np.ascontiguousarray(rgb[..., ::-1])
    finally:
        image.close()


def _detection_metrics(
    prefix: str,
    predictions: list[tuple[Any, Any, Any]],
    annotations: list[list[tuple[int, tuple[float, float, float, float]]]],
    *,
    class_count: int,
    confidence: float,
    iou_threshold: float,
    np: Any,
) -> dict[str, Any]:
    per_class: list[float] = []
    true_positive = false_positive = false_negative = 0
    matched_ious: list[float] = []
    prediction_count = 0
    for class_id in range(class_count):
        detections: list[tuple[float, bool, float]] = []
        ground_truth_count = 0
        for (boxes, scores, ids), expected in zip(predictions, annotations):
            expected_boxes = [box for label, box in expected if label == class_id]
            ground_truth_count += len(expected_boxes)
            candidates = [
                (float(score), tuple(float(value) for value in box))
                for box, score, label in zip(boxes, scores, ids)
                if int(label) == class_id and float(score) > confidence
            ]
            candidates.sort(reverse=True)
            prediction_count += len(candidates)
            used: set[int] = set()
            for score, box in candidates:
                best_index, best_iou = _best_match(box, expected_boxes, used)
                matched = best_index is not None and best_iou >= iou_threshold
                if matched:
                    used.add(best_index)
                detections.append((score, matched, best_iou if matched else 0.0))
                if matched:
                    true_positive += 1
                    matched_ious.append(best_iou)
                else:
                    false_positive += 1
            false_negative += len(expected_boxes) - len(used)
        average_precision = _average_precision(detections, ground_truth_count)
        if average_precision is not None:
            per_class.append(average_precision)

    images = len(predictions)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        f"{prefix}_map50": float(np.mean(per_class)) if per_class else 0.0,
        f"{prefix}_precision50": precision,
        f"{prefix}_recall50": recall,
        f"{prefix}_f1_50": 2.0 * precision * recall / max(1e-12, precision + recall),
        f"{prefix}_mean_iou50": float(np.mean(matched_ious)) if matched_ious else 0.0,
        f"{prefix}_true_positives50": true_positive,
        f"{prefix}_false_positives50": false_positive,
        f"{prefix}_false_negatives50": false_negative,
        f"{prefix}_predictions": prediction_count,
        f"{prefix}_images": images,
    }


def _average_precision(detections: list[tuple[float, bool, float]], ground_truth_count: int) -> float | None:
    if ground_truth_count == 0:
        return None
    detections.sort(key=lambda value: value[0], reverse=True)
    true_positive = false_positive = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for _, matched, _ in detections:
        if matched:
            true_positive += 1
        else:
            false_positive += 1
        precisions.append(true_positive / max(1, true_positive + false_positive))
        recalls.append(true_positive / ground_truth_count)
    return sum(
        max((precision for precision, recall in zip(precisions, recalls) if recall >= threshold), default=0.0)
        for threshold in [index / 100.0 for index in range(101)]
    ) / 101.0


def _best_match(box: tuple[float, float, float, float], expected: list[tuple[float, float, float, float]], used: set[int]) -> tuple[int | None, float]:
    best_index = None
    best_iou = 0.0
    for index, candidate in enumerate(expected):
        if index in used:
            continue
        overlap = _iou(box, candidate)
        if overlap > best_iou:
            best_index, best_iou = index, overlap
    return best_index, best_iou


def _variant_agreement(
    first: list[tuple[Any, Any, Any]],
    second: list[tuple[Any, Any, Any]],
    *,
    confidence: float,
    iou_threshold: float,
    np: Any,
) -> dict[str, Any]:
    matched = total = 0
    score_error: list[float] = []
    for (first_boxes, first_scores, first_ids), (second_boxes, second_scores, second_ids) in zip(first, second):
        left = [
            (tuple(float(value) for value in box), float(score), int(class_id))
            for box, score, class_id in zip(first_boxes, first_scores, first_ids)
            if float(score) > confidence
        ]
        right = [
            (tuple(float(value) for value in box), float(score), int(class_id))
            for box, score, class_id in zip(second_boxes, second_scores, second_ids)
            if float(score) > confidence
        ]
        used: set[int] = set()
        total += max(len(left), len(right))
        for box, score, class_id in left:
            candidates = [
                (index, _iou(box, other_box))
                for index, (other_box, other_score, other_class_id) in enumerate(right)
                if index not in used and other_class_id == class_id
            ]
            if not candidates:
                continue
            index, overlap = max(candidates, key=lambda value: value[1])
            if overlap >= iou_threshold:
                used.add(index)
                matched += 1
                score_error.append(abs(score - right[index][1]))
    agreement = 1.0 if total == 0 else matched / total
    return {
        "bw8_c24_agreement50": agreement,
        "bw8_c24_agreement": agreement,
        "bw8_c24_score_mae": float(np.mean(score_error)) if score_error else 0.0,
        "bw8_c24_matched_predictions50": matched,
    }


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(1e-12, first_area + second_area - intersection)
