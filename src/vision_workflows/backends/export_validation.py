from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..workflows.context import optional_import


_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def native_validation_metrics(values: Any) -> dict[str, Any]:
    raw = values.get("metrics", values) if isinstance(values, dict) else getattr(values, "results_dict", {})
    if not isinstance(raw, dict):
        return {"native": {"metric_count": 0}}
    native: dict[str, Any] = {}
    for key, value in raw.items():
        scalar = value.item() if hasattr(value, "item") else value
        if not isinstance(scalar, (int, float)):
            continue
        name = str(key).rsplit("/", 1)[-1]
        name = re.sub(r"\([^)]*\)$", "", name)
        name = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").casefold()
        if name:
            native[name] = scalar
    native["metric_count"] = len(raw)
    return {"native": native}


def validate_classification_wrappers(
    outputs: dict[str, Path],
    data_root: Path,
    *,
    classes: list[str],
    image_size: int,
    batch_size: int | None,
    reference_probabilities: Any | None = None,
    reference_name: str = "native",
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
        section = {
            "accuracy": correct / len(labels),
            "loss": float(np.mean(-np.log(np.clip(variant_probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)))),
            "correct": correct,
            "images": len(labels),
        }
        if reference_probabilities is not None and reference_predictions is not None:
            probability_error = np.abs(variant_probabilities - reference_probabilities)
            section.update({
                f"{reference_name}_agreement": float(np.mean(predictions == reference_predictions)),
                f"{reference_name}_probability_mae": float(np.mean(probability_error)),
                f"{reference_name}_max_probability_error": float(np.max(probability_error)),
            })
        result[variant] = section

    bw8 = probabilities_by_variant["bw8"]
    c24 = probabilities_by_variant["c24"]
    result.update({
        "bw8_c24_agreement": float(np.mean(bw8.argmax(axis=1) == c24.argmax(axis=1))),
        "bw8_c24_probability_mae": float(np.mean(np.abs(bw8 - c24))),
        "bw8_c24_max_probability_error": float(np.max(np.abs(bw8 - c24))),
    })
    return result


def validate_classification_native_export(
    output: Path,
    data_root: Path,
    *,
    classes: list[str],
    image_size: int,
    batch_size: int | None,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    pixel_scale: float = 255.0,
    apply_softmax: bool = False,
    reference_probabilities: Any | None = None,
) -> tuple[dict[str, Any], Any]:
    """Evaluate the exported float ONNX core using its native tensor input."""
    np = optional_import("numpy")
    ort = optional_import("onnxruntime")
    torchvision = optional_import("torchvision")
    dataset = torchvision.datasets.ImageFolder(str(data_root / "val"))
    if dataset.classes != classes:
        raise ValueError("dataset classes differ from checkpoint classes")
    if len(dataset) == 0:
        raise ValueError("classification validation split is empty")

    labels = np.asarray([label for _, label in dataset.samples], dtype=np.int64)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    run_batch = batch_size or 32
    probabilities: list[Any] = []
    for start in range(0, len(dataset.samples), run_batch):
        current = dataset.samples[start:start + run_batch]
        raw_images = [
            _classification_raw_image(
                path,
                "native-export",
                np,
                image_size=image_size,
                mean=mean,
                std=std,
                pixel_scale=pixel_scale,
            )
            for path, _ in current
        ]
        actual_batch = len(raw_images)
        while len(raw_images) < run_batch:
            raw_images.append(np.zeros_like(raw_images[0]))
        values = session.run(None, {input_name: np.stack(raw_images, axis=0)})
        batch_probabilities = np.asarray(values[0])
        if apply_softmax:
            shifted = batch_probabilities - np.max(batch_probabilities, axis=1, keepdims=True)
            exponentials = np.exp(shifted)
            batch_probabilities = exponentials / np.sum(exponentials, axis=1, keepdims=True)
        if batch_probabilities.shape != (run_batch, len(classes)):
            raise ValueError(
                "native-export ONNX validation output shape mismatch: "
                f"{batch_probabilities.shape}; expected {(run_batch, len(classes))}"
            )
        probabilities.append(batch_probabilities[:actual_batch])

    native_export_probabilities = np.concatenate(probabilities, axis=0)
    predictions = native_export_probabilities.argmax(axis=1)
    correct = int(np.sum(predictions == labels))
    section: dict[str, Any] = {
        "accuracy": correct / len(labels),
        "loss": float(np.mean(-np.log(np.clip(native_export_probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)))),
        "correct": correct,
        "images": len(labels),
    }
    if reference_probabilities is not None:
        reference_predictions = reference_probabilities.argmax(axis=1)
        probability_error = np.abs(native_export_probabilities - reference_probabilities)
        section.update({
            "native_agreement": float(np.mean(predictions == reference_predictions)),
            "native_probability_mae": float(np.mean(probability_error)),
            "native_max_probability_error": float(np.max(probability_error)),
        })
    return {"native-export": section}, native_export_probabilities


def _classification_raw_image(
    path: str,
    variant: str,
    np: Any,
    *,
    image_size: int | None = None,
    mean: tuple[float, ...] = (0.0, 0.0, 0.0),
    std: tuple[float, ...] = (1.0, 1.0, 1.0),
    pixel_scale: float = 255.0,
) -> Any:
    image_module = optional_import("PIL.Image")
    image = image_module.open(path)
    try:
        if variant == "bw8":
            # Keep the channel axis per image; batching then produces [B, 1, H, W].
            return np.asarray(image.convert("L"), dtype=np.uint8)[None, ...]
        if variant == "native-export":
            if image_size is None or image_size <= 0:
                raise ValueError("native-export image_size must be positive")
            if len(mean) != 3 or len(std) != 3 or pixel_scale <= 0:
                raise ValueError("native-export preprocessing requires three mean/std values and a positive pixel scale")
            resized = image.convert("RGB").resize((image_size, image_size), image_module.Resampling.BILINEAR)
            rgb = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / pixel_scale
            return (rgb - np.asarray(mean, dtype=np.float32)[:, None, None]) / np.asarray(std, dtype=np.float32)[:, None, None]
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return np.ascontiguousarray(rgb[..., ::-1])
    finally:
        image.close()


def validate_detection_native_export(
    output: Path,
    data_yaml: Path,
    *,
    class_count: int,
    image_size: int,
    batch_size: int | None,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    pixel_scale: float,
    resize_mode: str,
    resize_antialias: bool,
) -> tuple[dict[str, Any], list[tuple[Any, Any, Any]]]:
    """Evaluate the standardized exported detection core with native input."""
    if resize_mode not in {"stretch", "letterbox"}:
        raise ValueError("native-export resize_mode must be stretch or letterbox")
    if len(mean) != 3 or len(std) != 3 or pixel_scale <= 0:
        raise ValueError("native-export preprocessing requires three mean/std values and a positive pixel scale")

    np = optional_import("numpy")
    ort = optional_import("onnxruntime")
    samples = _yolo_validation_samples(data_yaml)
    if not samples:
        raise ValueError("detection validation split is empty")

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    run_batch = batch_size or 16
    predictions: list[tuple[Any, Any, Any]] = []
    for start in range(0, len(samples), run_batch):
        current = samples[start:start + run_batch]
        raw_images = [
            _native_detection_image(
                path,
                image_size=image_size,
                mean=mean,
                std=std,
                pixel_scale=pixel_scale,
                resize_mode=resize_mode,
                resize_antialias=resize_antialias,
                np=np,
            )
            for path, _ in current
        ]
        actual_batch = len(raw_images)
        while len(raw_images) < run_batch:
            raw_images.append(np.zeros_like(raw_images[0]))
        values = session.run(None, {input_name: np.stack(raw_images, axis=0)})
        boxes, scores, class_ids = (np.asarray(value) for value in values[:3])
        if boxes.shape[0] != run_batch or scores.shape[0] != run_batch or class_ids.shape[0] != run_batch:
            raise ValueError("native-export ONNX validation batch dimension mismatch")
        for index, (path, _) in enumerate(current):
            predictions.append((
                _restore_native_detection_boxes(
                    boxes[index],
                    path,
                    image_size=image_size,
                    resize_mode=resize_mode,
                    np=np,
                ),
                scores[index],
                class_ids[index],
            ))

    metrics = _detection_metrics(
        "native_export",
        predictions,
        [annotations for _, annotations in samples],
        class_count=class_count,
        confidence=0.0,
        iou_threshold=0.5,
        np=np,
    )
    return ({
        "native-export": {
            key.removeprefix("native_export_"): value
            for key, value in metrics.items()
        }
    }, predictions)


def _native_detection_image(
    path: Path,
    *,
    image_size: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    pixel_scale: float,
    resize_mode: str,
    resize_antialias: bool,
    np: Any,
) -> Any:
    image_module = optional_import("PIL.Image")
    image = image_module.open(path)
    try:
        rgb = image.convert("RGB")
        width, height = rgb.size
        resample = image_module.Resampling.LANCZOS if resize_antialias else image_module.Resampling.BILINEAR
        if resize_mode == "stretch":
            prepared = rgb.resize((image_size, image_size), resample)
        else:
            scale = image_size / max(height, width)
            resized_height = max(1, int(np.floor(height * scale + 0.5)))
            resized_width = max(1, int(np.floor(width * scale + 0.5)))
            resized = rgb.resize((resized_width, resized_height), resample)
            prepared = image_module.new("RGB", (image_size, image_size), (114, 114, 114))
            prepared.paste(resized, ((image_size - resized_width) // 2, (image_size - resized_height) // 2))
        pixels = np.asarray(prepared, dtype=np.float32).transpose(2, 0, 1) / pixel_scale
        return (pixels - np.asarray(mean, dtype=np.float32)[:, None, None]) / np.asarray(std, dtype=np.float32)[:, None, None]
    finally:
        image.close()


def _restore_native_detection_boxes(
    boxes: Any,
    path: Path,
    *,
    image_size: int,
    resize_mode: str,
    np: Any,
) -> Any:
    if resize_mode == "stretch":
        return boxes
    image_module = optional_import("PIL.Image")
    image = image_module.open(path)
    try:
        width, height = image.size
    finally:
        image.close()
    scale = image_size / max(height, width)
    resized_height = max(1, int(np.floor(height * scale + 0.5)))
    resized_width = max(1, int(np.floor(width * scale + 0.5)))
    top = (image_size - resized_height) // 2
    left = (image_size - resized_width) // 2
    restored = np.asarray(boxes, dtype=np.float32).copy() * image_size
    restored[..., [0, 2]] = (restored[..., [0, 2]] - left) / resized_width
    restored[..., [1, 3]] = (restored[..., [1, 3]] - top) / resized_height
    return np.clip(restored, 0.0, 1.0)


def validate_detection_wrappers(
    outputs: dict[str, Path],
    data_yaml: Path,
    *,
    class_count: int,
    image_size: int,
    batch_size: int | None,
    confidence: float = 0.0,
    iou_threshold: float = 0.5,
    reference_predictions: list[tuple[Any, Any, Any]] | None = None,
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
        metrics = _detection_metrics(
            variant,
            predictions,
            [annotations for _, annotations in samples],
            class_count=class_count,
            confidence=confidence,
            iou_threshold=iou_threshold,
            np=np,
        )
        result[variant] = {
            key.removeprefix(f"{variant}_"): value
            for key, value in metrics.items()
        }

    result.update(_variant_agreement(
        predictions_by_variant["bw8"],
        predictions_by_variant["c24"],
        prefix="bw8_c24",
        confidence=confidence,
        iou_threshold=iou_threshold,
        np=np,
    ))
    if reference_predictions is not None:
        result.update(_variant_agreement(
            predictions_by_variant["bw8"],
            reference_predictions,
            prefix="bw8_native_export",
            confidence=confidence,
            iou_threshold=iou_threshold,
            np=np,
        ))
        result.update(_variant_agreement(
            predictions_by_variant["c24"],
            reference_predictions,
            prefix="c24_native_export",
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
        index = next(index for index in range(len(parts) - 1, -1, -1) if parts[index].casefold() == "images")
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
            # Keep the channel axis per image; batching then produces [B, 1, H, W].
            return np.asarray(image.convert("L"), dtype=np.uint8)[None, ...]
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
    true_positive = false_positive = false_negative = 0
    matched_ious: list[float] = []
    prediction_count = 0
    for class_id in range(class_count):
        for (boxes, scores, ids), expected in zip(predictions, annotations):
            expected_boxes = [box for label, box in expected if label == class_id]
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
                if matched:
                    true_positive += 1
                    matched_ious.append(best_iou)
                else:
                    false_positive += 1
            false_negative += len(expected_boxes) - len(used)
    images = len(predictions)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        f"{prefix}_map50": _mean_average_precision_at_iou(
            predictions,
            annotations,
            class_count=class_count,
            confidence=confidence,
            np=np,
            iou_thresholds=[0.5],
        ),
        f"{prefix}_map50_95": _mean_average_precision_at_iou(
            predictions,
            annotations,
            class_count=class_count,
            confidence=confidence,
            np=np,
        ),
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


def _mean_average_precision_at_iou(
    predictions: list[tuple[Any, Any, Any]],
    annotations: list[list[tuple[int, tuple[float, float, float, float]]]],
    *,
    class_count: int,
    confidence: float,
    np: Any,
    iou_thresholds: Any | None = None,
) -> float:
    """Compute the Ultralytics-style mean AP over IoU 0.50:0.95."""
    average_precisions: list[float] = []
    thresholds = np.linspace(0.5, 0.95, 10) if iou_thresholds is None else iou_thresholds
    for iou_threshold in thresholds:
        per_class: list[float] = []
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
                matched = _matched_prediction_indices(candidates, expected_boxes, float(iou_threshold), np=np)
                detections.extend(
                    (score, index in matched, 0.0)
                    for index, (score, _) in enumerate(candidates)
                )
            average_precision = _ultralytics_average_precision(detections, ground_truth_count, np)
            if average_precision is not None:
                per_class.append(average_precision)
        average_precisions.append(float(np.mean(per_class)) if per_class else 0.0)
    return float(np.mean(average_precisions)) if average_precisions else 0.0


def _matched_prediction_indices(
    candidates: list[tuple[float, tuple[float, float, float, float]]],
    expected: list[tuple[float, float, float, float]],
    iou_threshold: float,
    *,
    np: Any,
) -> set[int]:
    if not candidates or not expected:
        return set()
    candidate_boxes = np.asarray([box for _, box in candidates], dtype=np.float32)
    expected_boxes = np.asarray(expected, dtype=np.float32)
    left_top = np.maximum(expected_boxes[:, None, :2], candidate_boxes[None, :, :2])
    right_bottom = np.minimum(expected_boxes[:, None, 2:], candidate_boxes[None, :, 2:])
    intersection = np.maximum(right_bottom - left_top, 0).prod(axis=2)
    expected_area = (expected_boxes[:, 2:] - expected_boxes[:, :2]).prod(axis=1)[:, None]
    candidate_area = (candidate_boxes[:, 2:] - candidate_boxes[:, :2]).prod(axis=1)[None, :]
    iou = intersection / (expected_area + candidate_area - intersection + np.float32(1e-7))
    matches = np.asarray(np.nonzero(iou >= iou_threshold)).T
    if len(matches) > 1:
        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    return {int(prediction_index) for _, prediction_index in matches}


def _ultralytics_average_precision(
    detections: list[tuple[float, bool, float]],
    ground_truth_count: int,
    np: Any,
) -> float | None:
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
    modified_recall = np.asarray([0.0, *recalls, recalls[-1] if recalls else 1.0, 1.0], dtype=float)
    precision_envelope = np.asarray([1.0, *precisions, 0.0, 0.0], dtype=float)
    precision_envelope = np.flip(np.maximum.accumulate(np.flip(precision_envelope)))
    x = np.linspace(0.0, 1.0, 101)
    interpolated = np.interp(x, modified_recall, precision_envelope)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(interpolated, x))


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
    prefix: str,
    confidence: float,
    iou_threshold: float,
    np: Any,
) -> dict[str, Any]:
    if len(first) != len(second):
        raise ValueError("variant agreement requires predictions for the same images")
    matched = total = 0
    score_error: list[float] = []
    for (first_boxes, first_scores, first_ids), (second_boxes, second_scores, second_ids) in zip(first, second):
        left = _valid_detection_candidates(first_boxes, first_scores, first_ids, confidence=confidence, np=np)
        right = _valid_detection_candidates(second_boxes, second_scores, second_ids, confidence=confidence, np=np)
        total += max(len(left), len(right))
        pairs = _maximum_iou_matching(left, right, iou_threshold)
        matched += len(pairs)
        score_error.extend(abs(left[left_index][1] - right[right_index][1]) for left_index, right_index in pairs)
    agreement = 1.0 if total == 0 else matched / total
    return {
        f"{prefix}_agreement50": agreement,
        f"{prefix}_agreement": agreement,
        f"{prefix}_score_mae": float(np.mean(score_error)) if score_error else 0.0,
        f"{prefix}_matched_predictions50": matched,
    }


def _valid_detection_candidates(
    boxes: Any,
    scores: Any,
    class_ids: Any,
    *,
    confidence: float,
    np: Any,
) -> list[tuple[tuple[float, float, float, float], float, int]]:
    candidates: list[tuple[tuple[float, float, float, float], float, int]] = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        score = float(score)
        coordinates = tuple(float(value) for value in box)
        class_id = int(class_id)
        if (
            not np.isfinite(score)
            or score <= confidence
            or len(coordinates) != 4
            or not np.isfinite(np.asarray(coordinates, dtype=np.float32)).all()
            or any(value < 0.0 or value > 1.0 for value in coordinates)
            or coordinates[2] <= coordinates[0]
            or coordinates[3] <= coordinates[1]
            or class_id < 0
        ):
            continue
        candidates.append((coordinates, score, class_id))
    return candidates


def _maximum_iou_matching(
    first: list[tuple[tuple[float, float, float, float], float, int]],
    second: list[tuple[tuple[float, float, float, float], float, int]],
    iou_threshold: float,
) -> list[tuple[int, int]]:
    edges: dict[int, list[tuple[int, float]]] = {}
    for first_index, (first_box, _, first_class_id) in enumerate(first):
        candidates: list[tuple[int, float]] = []
        for second_index, (second_box, _, second_class_id) in enumerate(second):
            if first_class_id != second_class_id:
                continue
            overlap = _iou(first_box, second_box)
            if overlap >= iou_threshold:
                candidates.append((second_index, overlap))
        edges[first_index] = sorted(
            candidates,
            key=lambda value: (-value[1], value[0]),
        )

    second_to_first: dict[int, int] = {}

    def augment(first_index: int, visited: set[int]) -> bool:
        for second_index, _ in edges[first_index]:
            if second_index in visited:
                continue
            visited.add(second_index)
            previous = second_to_first.get(second_index)
            if previous is None or augment(previous, visited):
                second_to_first[second_index] = first_index
                return True
        return False

    for first_index in sorted(edges, key=lambda index: (len(edges[index]), index)):
        augment(first_index, set())
    return sorted((first_index, second_index) for second_index, first_index in second_to_first.items())


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(1e-12, first_area + second_area - intersection)
