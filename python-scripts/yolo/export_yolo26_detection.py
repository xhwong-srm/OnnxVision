"""Export YOLO26 detection weights to ONNX and the repository detection contract.

The contract outputs normalized ``boxes`` [1,N,4] in xyxy form, ``scores``
[1,N], and zero-based int64 ``class_ids`` [1,N]. Optional BW8 and C24 models
embed stretch-resize and uint8 preprocessing for the .NET ObjectDetector.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, compose, helper, numpy_helper
from PIL import Image


CONTRACT_VERSION = "onnx-vision-detection-v1"


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="YOLO26 .pt checkpoint")
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--output", type=Path, default=Path("artifacts/yolo26-detection.onnx"))
    parser.add_argument("--bw8-output", type=Path)
    parser.add_argument("--c24-output", type=Path)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--quantize",
        type=int,
        choices=(16, 32),
        default=32,
        help="ONNX floating-point precision: 16 reduces model size; 32 maximizes compatibility (default: 32)",
    )
    parser.add_argument(
        "--export-confidence",
        type=float,
        default=0.001,
        help="Minimum confidence retained inside the exported end-to-end graph (default: 0.001)",
    )
    parser.add_argument(
        "--simplify", action=BooleanOptionalAction, default=True,
        help="Let Ultralytics simplify the exported graph (default: enabled)",
    )
    parser.add_argument("--skip-embedded-preprocessing", action="store_true")
    parser.add_argument(
        "--validate-image",
        nargs="?",
        const="__dataset__",
        metavar="IMAGE",
        help=(
            "Evaluate the dataset test split against annotations and compare native PyTorch "
            "with every exported ONNX model. Optionally supply one dataset image instead."
        ),
    )
    parser.add_argument("--data", type=Path, help="YOLO data.yaml used by --validate-image")
    parser.add_argument("--validation-split", default="test", help="Dataset split to evaluate (default: test)")
    parser.add_argument(
        "--validation-limit",
        type=int,
        default=0,
        help="Maximum dataset images to evaluate; 0 evaluates the complete split",
    )
    parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU threshold for TP/FP matching")
    parser.add_argument("--validation-report", type=Path, help="JSON report path")
    return parser.parse_args()


@dataclass(frozen=True)
class Detection:
    box: np.ndarray
    score: float
    class_id: int


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path
    ground_truth: tuple[Detection, ...]


def add_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> str:
    model.graph.initializer.append(numpy_helper.from_array(value, name))
    return name


def set_metadata(model: onnx.ModelProto, values: dict[str, str]) -> None:
    existing = {item.key: item.value for item in model.metadata_props}
    existing.update(values)
    del model.metadata_props[:]
    for key, value in existing.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value


def topologically_sort_nodes(model: onnx.ModelProto) -> None:
    """Stable-sort graph nodes after Ultralytics FP16 conversion moves input Casts."""
    available = (
        {item.name for item in model.graph.input}
        | {item.name for item in model.graph.initializer}
        | {item.name for item in model.graph.sparse_initializer}
    )
    pending = list(model.graph.node)
    ordered: list[onnx.NodeProto] = []
    while pending:
        ready = [
            node for node in pending
            if all(not name or name in available for name in node.input)
        ]
        if not ready:
            unresolved = sorted({
                name
                for node in pending
                for name in node.input
                if name and name not in available
            })
            raise ValueError(f"Cannot topologically sort ONNX graph; unresolved inputs: {unresolved}")
        ready_ids = {id(node) for node in ready}
        pending = [node for node in pending if id(node) not in ready_ids]
        ordered.extend(ready)
        available.update(name for node in ready for name in node.output if name)
    del model.graph.node[:]
    model.graph.node.extend(ordered)


def canonicalize(official: Path, output: Path, resolution: int, names: dict[int, str]) -> None:
    model = onnx.load(official)
    if len(model.graph.output) != 1:
        raise ValueError("Expected one YOLO26 end-to-end detection output")
    source = model.graph.output[0]
    shape = [dimension.dim_value for dimension in source.type.tensor_type.shape.dim]
    if len(shape) != 3 or shape[-1] != 6:
        raise ValueError(f"Expected YOLO26 [batch,N,6] output, received shape {shape}")
    source_name = source.name
    del model.graph.output[:]
    prefix = "contract/"
    box_indices = add_initializer(model, prefix + "box_indices", np.arange(4, dtype=np.int64))
    score_index = add_initializer(model, prefix + "score_index", np.asarray([4], dtype=np.int64))
    class_index = add_initializer(model, prefix + "class_index", np.asarray([5], dtype=np.int64))
    scale = add_initializer(model, prefix + "resolution", np.asarray(float(resolution), dtype=np.float32))
    squeeze_axis = add_initializer(model, prefix + "squeeze_axis", np.asarray([2], dtype=np.int64))
    model.graph.node.extend(
        [
            helper.make_node("Gather", [source_name, box_indices], [prefix + "pixel_boxes"], axis=2),
            helper.make_node("Div", [prefix + "pixel_boxes", scale], ["boxes"]),
            helper.make_node("Gather", [source_name, score_index], [prefix + "scores_3d"], axis=2),
            helper.make_node("Squeeze", [prefix + "scores_3d", squeeze_axis], ["scores"]),
            helper.make_node("Gather", [source_name, class_index], [prefix + "classes_float"], axis=2),
            helper.make_node("Squeeze", [prefix + "classes_float", squeeze_axis], [prefix + "classes_2d"]),
            helper.make_node("Cast", [prefix + "classes_2d"], ["class_ids"], to=TensorProto.INT64),
        ]
    )
    candidate_count = shape[1] or 300
    model.graph.output.extend(
        [
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, candidate_count, 4]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, candidate_count]),
            helper.make_tensor_value_info("class_ids", TensorProto.INT64, [1, candidate_count]),
        ]
    )
    set_metadata(
        model,
        {
            "vision_task": "object_detection",
            "detection_contract": CONTRACT_VERSION,
            "box_format": "xyxy",
            "box_coordinates": "normalized",
            "names": json.dumps({index: names[index] for index in sorted(names)}),
            "source_model": "ultralytics-yolo26",
            "candidate_count": str(candidate_count),
            "resize_mode": "stretch",
        },
    )
    topologically_sort_nodes(model)
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)


def prepare_core(path: Path) -> tuple[onnx.ModelProto, str, dict[str, str]]:
    model = onnx.load(path)
    metadata = {item.key: item.value for item in model.metadata_props}
    core = compose.add_prefix(model, "core/", rename_outputs=False)
    input_name = core.graph.input[0].name
    del core.graph.input[:]
    return core, input_name, metadata


def resize(source: str, target: str, sizes: str) -> onnx.NodeProto:
    return helper.make_node(
        "Resize", [source, "", "", sizes], [target],
        mode="linear", coordinate_transformation_mode="half_pixel",
    )


def finish_wrapper(
    core: onnx.ModelProto,
    nodes: list[onnx.NodeProto],
    output: Path,
    metadata: dict[str, str],
) -> None:
    original = list(core.graph.node)
    del core.graph.node[:]
    core.graph.node.extend(nodes + original)
    set_metadata(core, metadata)
    topologically_sort_nodes(core)
    onnx.checker.check_model(core)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(core, output)


def wrap_bw8(core_path: Path, output: Path, resolution: int) -> None:
    core, core_input, metadata = prepare_core(core_path)
    core.graph.input.append(helper.make_tensor_value_info(
        "images_bw8_uint8_nchw", TensorProto.UINT8, [1, 1, "height", "width"]
    ))
    sizes = add_initializer(
        core, "preprocess/sizes", np.asarray([1, 1, resolution, resolution], dtype=np.int64)
    )
    rgb_shape = add_initializer(
        core, "preprocess/rgb_shape", np.asarray([1, 3, resolution, resolution], dtype=np.int64)
    )
    scale = add_initializer(
        core, "preprocess/pixel_scale", np.asarray([[[[255.0]]]], dtype=np.float32)
    )
    nodes = [
        resize("images_bw8_uint8_nchw", "preprocess/resized", sizes),
        helper.make_node("Cast", ["preprocess/resized"], ["preprocess/float"], to=TensorProto.FLOAT),
        helper.make_node("Div", ["preprocess/float", scale], ["preprocess/scaled"]),
        helper.make_node("Expand", ["preprocess/scaled", rgb_shape], [core_input]),
    ]
    finish_wrapper(core, nodes, output, metadata)


def wrap_c24(core_path: Path, output: Path, resolution: int) -> None:
    core, core_input, metadata = prepare_core(core_path)
    core.graph.input.append(helper.make_tensor_value_info(
        "images_c24_uint8_nhwc_bgr", TensorProto.UINT8, [1, "height", "width", 3]
    ))
    sizes = add_initializer(
        core, "preprocess/sizes", np.asarray([1, 3, resolution, resolution], dtype=np.int64)
    )
    indices = add_initializer(
        core, "preprocess/bgr_to_rgb", np.asarray([2, 1, 0], dtype=np.int64)
    )
    scale = add_initializer(
        core, "preprocess/pixel_scale", np.asarray([[[[255.0]]]], dtype=np.float32)
    )
    nodes = [
        helper.make_node("Transpose", ["images_c24_uint8_nhwc_bgr"], ["preprocess/bgr"], perm=[0, 3, 1, 2]),
        helper.make_node("Gather", ["preprocess/bgr", indices], ["preprocess/rgb"], axis=1),
        resize("preprocess/rgb", "preprocess/resized", sizes),
        helper.make_node("Cast", ["preprocess/resized"], ["preprocess/float"], to=TensorProto.FLOAT),
        helper.make_node("Div", ["preprocess/float", scale], [core_input]),
    ]
    finish_wrapper(core, nodes, output, metadata)


def load_dataset_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required and is installed with Ultralytics") from error
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read dataset YAML {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {path}")
    return document


def dataset_names(document: dict[str, Any]) -> dict[int, str]:
    names = document.get("names")
    if isinstance(names, list):
        result = {index: str(name) for index, name in enumerate(names)}
    elif isinstance(names, dict):
        try:
            result = {int(index): str(name) for index, name in names.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("Dataset names must use integer class IDs") from error
    else:
        raise ValueError("Dataset YAML must define names as a list or mapping")
    if sorted(result) != list(range(len(result))):
        raise ValueError("Dataset class IDs must be contiguous and zero-based")
    return result


def dataset_root(yaml_path: Path, document: dict[str, Any]) -> Path:
    configured = Path(str(document.get("path", yaml_path.parent)))
    return (configured if configured.is_absolute() else yaml_path.parent / configured).resolve()


def image_files_for_split(root: Path, split_value: Any) -> list[Path]:
    values = split_value if isinstance(split_value, list) else [split_value]
    extensions = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    images: list[Path] = []
    for value in values:
        source = Path(str(value))
        source = source if source.is_absolute() else root / source
        source = source.resolve()
        if source.is_dir():
            images.extend(path for path in source.rglob("*") if path.suffix.lower() in extensions)
        elif source.is_file() and source.suffix.lower() == ".txt":
            for line in source.read_text(encoding="utf-8").splitlines():
                candidate = Path(line.strip())
                if not candidate.is_absolute():
                    candidate = source.parent / candidate
                if candidate.suffix.lower() in extensions:
                    images.append(candidate.resolve())
        elif source.is_file() and source.suffix.lower() in extensions:
            images.append(source)
        else:
            raise FileNotFoundError(f"Dataset image source does not exist: {source}")
    return sorted(set(images), key=lambda path: str(path).lower())


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_indices = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not image_indices:
        raise ValueError(f"Cannot map image to YOLO label because path has no 'images' folder: {image_path}")
    parts[image_indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_ground_truth(image_path: Path) -> Sample:
    label_path = label_path_for_image(image_path)
    if not label_path.is_file():
        raise FileNotFoundError(f"Annotation does not exist for {image_path}: {label_path}")
    detections: list[Detection] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Expected YOLO detection label with 5 fields at {label_path}:{line_number}")
        class_id = int(fields[0])
        cx, cy, width, height = (float(value) for value in fields[1:])
        box = np.asarray(
            [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
            dtype=np.float32,
        )
        detections.append(Detection(np.clip(box, 0.0, 1.0), 1.0, class_id))
    return Sample(image_path, label_path, tuple(detections))


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    top_left = np.maximum(first[:2], second[:2])
    bottom_right = np.minimum(first[2:], second[2:])
    intersection = float(np.prod(np.maximum(bottom_right - top_left, 0.0)))
    first_area = float(np.prod(np.maximum(first[2:] - first[:2], 0.0)))
    second_area = float(np.prod(np.maximum(second[2:] - second[:2], 0.0)))
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def match_detections(
    predictions: tuple[Detection, ...],
    ground_truth: tuple[Detection, ...],
    iou_threshold: float,
) -> tuple[int, int, int, list[float]]:
    matched: set[int] = set()
    matched_ious: list[float] = []
    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        candidates = [
            (box_iou(prediction.box, truth.box), index)
            for index, truth in enumerate(ground_truth)
            if index not in matched and truth.class_id == prediction.class_id
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= iou_threshold:
            matched.add(best_index)
            matched_ious.append(best_iou)
    true_positive = len(matched)
    return true_positive, len(predictions) - true_positive, len(ground_truth) - true_positive, matched_ious


def average_precision(
    samples: list[Sample],
    predictions: dict[Path, tuple[Detection, ...]],
    class_id: int,
    iou_threshold: float,
) -> float | None:
    truth_by_image = {
        sample.image: [item for item in sample.ground_truth if item.class_id == class_id]
        for sample in samples
    }
    truth_count = sum(len(items) for items in truth_by_image.values())
    if truth_count == 0:
        return None
    ranked = sorted(
        (
            (prediction.score, image_path, prediction)
            for image_path, items in predictions.items()
            for prediction in items
            if prediction.class_id == class_id
        ),
        reverse=True,
        key=lambda item: item[0],
    )
    used = {image_path: set() for image_path in truth_by_image}
    tp_values: list[float] = []
    fp_values: list[float] = []
    for _, image_path, prediction in ranked:
        candidates = [
            (box_iou(prediction.box, truth.box), index)
            for index, truth in enumerate(truth_by_image[image_path])
            if index not in used[image_path]
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        is_match = best_iou >= iou_threshold
        if is_match:
            used[image_path].add(best_index)
        tp_values.append(float(is_match))
        fp_values.append(float(not is_match))
    if not ranked:
        return 0.0
    tp = np.cumsum(tp_values)
    fp = np.cumsum(fp_values)
    recall = tp / truth_count
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return float(np.mean([
        np.max(precision[recall >= level], initial=0.0)
        for level in np.linspace(0.0, 1.0, 101)
    ]))


def quality_metrics(
    samples: list[Sample],
    predictions: dict[Path, tuple[Detection, ...]],
    names: dict[int, str],
    iou_threshold: float,
    confidence: float,
) -> dict[str, Any]:
    thresholded = {
        path: tuple(item for item in items if item.score >= confidence)
        for path, items in predictions.items()
    }
    tp = fp = fn = 0
    matched_ious: list[float] = []
    for sample in samples:
        counts = match_detections(thresholded[sample.image], sample.ground_truth, iou_threshold)
        tp += counts[0]
        fp += counts[1]
        fn += counts[2]
        matched_ious.extend(counts[3])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    thresholds = [value / 100 for value in range(50, 100, 5)]
    per_class: dict[str, Any] = {}
    all_ap_values: list[float] = []
    ap50_values: list[float] = []
    for class_id, name in names.items():
        aps = [average_precision(samples, predictions, class_id, threshold) for threshold in thresholds]
        valid = [value for value in aps if value is not None]
        if valid:
            all_ap_values.extend(valid)
            ap50_values.append(valid[0])
        class_samples = [
            Sample(
                sample.image,
                sample.label,
                tuple(item for item in sample.ground_truth if item.class_id == class_id),
            )
            for sample in samples
        ]
        class_predictions = {
            path: tuple(item for item in items if item.class_id == class_id)
            for path, items in thresholded.items()
        }
        class_tp = class_fp = class_fn = 0
        for sample in class_samples:
            counts = match_detections(class_predictions[sample.image], sample.ground_truth, iou_threshold)
            class_tp += counts[0]
            class_fp += counts[1]
            class_fn += counts[2]
        class_precision = class_tp / (class_tp + class_fp) if class_tp + class_fp else 0.0
        class_recall = class_tp / (class_tp + class_fn) if class_tp + class_fn else 0.0
        per_class[name] = {
            "class_id": class_id,
            "ground_truth": class_tp + class_fn,
            "predictions": class_tp + class_fp,
            "true_positive": class_tp,
            "false_positive": class_fp,
            "false_negative": class_fn,
            "precision": class_precision,
            "recall": class_recall,
            "f1": (
                2 * class_precision * class_recall / (class_precision + class_recall)
                if class_precision + class_recall else 0.0
            ),
            "ap50": valid[0] if valid else None,
            "ap50_95": float(np.mean(valid)) if valid else None,
        }
    return {
        "images": len(samples),
        "ground_truth": tp + fn,
        "predictions": tp + fp,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else None,
        "ap50": float(np.mean(ap50_values)) if ap50_values else None,
        "map50_95": float(np.mean(all_ap_values)) if all_ap_values else None,
        "per_class": per_class,
    }


def onnx_session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def run_onnx(
    session: ort.InferenceSession,
    image_path: Path,
    confidence: float,
) -> tuple[Detection, ...]:
    input_info = session.get_inputs()[0]
    with Image.open(image_path) as image:
        if input_info.type == "tensor(uint8)" and input_info.name.startswith("images_bw8"):
            value = np.asarray(image.convert("L"), dtype=np.uint8)[None, None]
        elif input_info.type == "tensor(uint8)":
            value = np.asarray(image.convert("RGB"), dtype=np.uint8)[..., ::-1].copy()[None]
        else:
            height, width = int(input_info.shape[2]), int(input_info.shape[3])
            value = np.asarray(image.convert("RGB").resize((width, height)), dtype=np.float32)
            value = value.transpose(2, 0, 1)[None] / 255.0
    outputs = {
        item.name: np.asarray(result)
        for item, result in zip(
            session.get_outputs(), session.run(None, {input_info.name: value})
        )
    }
    return tuple(
        Detection(np.asarray(box, dtype=np.float32), float(score), int(class_id))
        for box, score, class_id in zip(
            outputs["boxes"][0], outputs["scores"][0], outputs["class_ids"][0]
        )
        if float(score) >= confidence
    )


def run_native(
    yolo: Any,
    image_path: Path,
    resolution: int,
    confidence: float,
) -> tuple[Detection, ...]:
    with Image.open(image_path) as source:
        # Ultralytics treats numpy sources as OpenCV-style BGR and converts them to RGB.
        resized = np.asarray(source.convert("RGB").resize((resolution, resolution)))[..., ::-1].copy()
    result = yolo.predict(
        source=resized,
        imgsz=resolution,
        conf=confidence,
        iou=0.7,
        device="cpu",
        verbose=False,
    )[0]
    boxes = result.boxes
    if boxes is None:
        return ()
    return tuple(
        Detection(
            np.asarray(box, dtype=np.float32) / float(resolution),
            float(score),
            int(class_id),
        )
        for box, score, class_id in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy(),
        )
    )


def parity_metrics(
    reference: dict[Path, tuple[Detection, ...]],
    candidate: dict[Path, tuple[Detection, ...]],
    iou_threshold: float,
) -> dict[str, Any]:
    reference_count = sum(len(items) for items in reference.values())
    candidate_count = sum(len(items) for items in candidate.values())
    matched_ious: list[float] = []
    box_differences: list[float] = []
    score_differences: list[float] = []
    class_matches = 0
    for image_path, reference_items in reference.items():
        available = set(range(len(candidate[image_path])))
        for item in sorted(reference_items, key=lambda detection: detection.score, reverse=True):
            choices = [
                (box_iou(item.box, candidate[image_path][index].box), index)
                for index in available
            ]
            best_iou, best_index = max(choices, default=(0.0, -1))
            if best_iou < iou_threshold:
                continue
            available.remove(best_index)
            other = candidate[image_path][best_index]
            matched_ious.append(best_iou)
            box_differences.extend(np.abs(item.box - other.box).tolist())
            score_differences.append(abs(item.score - other.score))
            class_matches += int(item.class_id == other.class_id)
    matched = len(matched_ious)
    return {
        "native_detections": reference_count,
        "onnx_detections": candidate_count,
        "matched_detections": matched,
        "native_unmatched": reference_count - matched,
        "onnx_unmatched": candidate_count - matched,
        "class_agreement": class_matches / matched if matched else None,
        "mean_pair_iou": float(np.mean(matched_ious)) if matched_ious else None,
        "minimum_pair_iou": float(np.min(matched_ious)) if matched_ious else None,
        "mean_abs_box_coordinate_delta": (
            float(np.mean(box_differences)) if box_differences else None
        ),
        "max_abs_box_coordinate_delta": (
            float(np.max(box_differences)) if box_differences else None
        ),
        "mean_abs_score_delta": float(np.mean(score_differences)) if score_differences else None,
        "max_abs_score_delta": float(np.max(score_differences)) if score_differences else None,
    }


def print_quality(name: str, metrics: dict[str, Any]) -> None:
    def formatted(value: float | None) -> str:
        return f"{value:.6f}" if value is not None else "n/a"

    print(
        f"{name}: images={metrics['images']} gt={metrics['ground_truth']} "
        f"pred={metrics['predictions']} TP={metrics['true_positive']} "
        f"FP={metrics['false_positive']} FN={metrics['false_negative']}"
    )
    print(
        f"{name}: precision={metrics['precision']:.6f} recall={metrics['recall']:.6f} "
        f"f1={metrics['f1']:.6f} AP50={formatted(metrics['ap50'])} "
        f"mAP50-95={formatted(metrics['map50_95'])}"
    )
    for class_name, values in metrics["per_class"].items():
        print(
            f"{name}/{class_name}: gt={values['ground_truth']} pred={values['predictions']} "
            f"precision={values['precision']:.6f} recall={values['recall']:.6f} "
            f"AP50={formatted(values['ap50'])} mAP50-95={formatted(values['ap50_95'])}"
        )


def evaluate(
    yolo: Any,
    checkpoint: Path,
    generated: list[Path],
    data_path: Path,
    requested_image: str,
    split: str,
    limit: int,
    resolution: int,
    confidence: float,
    inference_floor: float,
    iou_threshold: float,
    report_path: Path,
) -> None:
    document = load_dataset_yaml(data_path)
    names = dataset_names(document)
    if split not in document:
        raise ValueError(f"Dataset YAML does not define split '{split}': {data_path}")
    images = image_files_for_split(dataset_root(data_path, document), document[split])
    if requested_image != "__dataset__":
        selected = Path(requested_image).expanduser().resolve()
        if selected not in images:
            raise ValueError(f"--validate-image must belong to dataset split '{split}': {selected}")
        images = [selected]
    elif limit:
        images = images[:limit]
    if not images:
        raise ValueError(f"Dataset split '{split}' contains no supported images")
    samples = [read_ground_truth(path) for path in images]
    print(
        f"validation_dataset={data_path}; split={split}; images={len(samples)}; "
        f"confidence={confidence}; iou={iou_threshold}"
    )

    predictions: dict[str, dict[Path, tuple[Detection, ...]]] = {"native_pt": {}}
    for index, sample in enumerate(samples, 1):
        predictions["native_pt"][sample.image] = run_native(
            yolo, sample.image, resolution, inference_floor
        )
        print(f"validation_progress={index}/{len(samples)} image={sample.image.name}")
    for path in generated:
        session = onnx_session(path)
        backend = path.name
        predictions[backend] = {
            sample.image: run_onnx(session, sample.image, inference_floor)
            for sample in samples
        }

    quality = {
        backend: quality_metrics(samples, values, names, iou_threshold, confidence)
        for backend, values in predictions.items()
    }
    thresholded_predictions = {
        backend: {
            path: tuple(item for item in items if item.score >= confidence)
            for path, items in values.items()
        }
        for backend, values in predictions.items()
    }
    parity = {
        backend: parity_metrics(thresholded_predictions["native_pt"], values, iou_threshold)
        for backend, values in thresholded_predictions.items()
        if backend != "native_pt"
    }
    per_image: list[dict[str, Any]] = []
    for sample in samples:
        item: dict[str, Any] = {
            "image": str(sample.image),
            "annotation": str(sample.label),
            "ground_truth": len(sample.ground_truth),
            "backends": {},
        }
        for backend, values in thresholded_predictions.items():
            counts = match_detections(values[sample.image], sample.ground_truth, iou_threshold)
            item["backends"][backend] = {
                "predictions": len(values[sample.image]),
                "true_positive": counts[0],
                "false_positive": counts[1],
                "false_negative": counts[2],
                "mean_matched_iou": float(np.mean(counts[3])) if counts[3] else None,
            }
        item["native_vs_onnx"] = {
            backend: parity_metrics(
                {sample.image: thresholded_predictions["native_pt"][sample.image]},
                {sample.image: values[sample.image]},
                iou_threshold,
            )
            for backend, values in thresholded_predictions.items()
            if backend != "native_pt"
        }
        per_image.append(item)
    for backend, values in quality.items():
        print_quality(backend, values)
    for backend, values in parity.items():
        print(
            f"parity/{backend}: matched={values['matched_detections']} "
            f"native_unmatched={values['native_unmatched']} onnx_unmatched={values['onnx_unmatched']} "
            f"mean_IoU={values['mean_pair_iou']} mean_box_delta={values['mean_abs_box_coordinate_delta']} "
            f"mean_score_delta={values['mean_abs_score_delta']}"
        )
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "dataset": str(data_path),
        "split": split,
        "images": [str(sample.image) for sample in samples],
        "settings": {
            "resolution": resolution,
            "confidence": confidence,
            "inference_floor": inference_floor,
            "iou": iou_threshold,
            "onnx_provider": "CPUExecutionProvider",
            "resize_mode": "stretch",
        },
        "interpretation": {
            "quality": "Predictions compared with YOLO ground-truth annotations.",
            "parity": "Each ONNX result compared with native PyTorch on the same stretched image.",
            "ap": (
                "101-point interpolated AP using detections down to inference_floor; "
                "mAP50-95 averages IoU thresholds 0.50 through 0.95."
            ),
        },
        "quality": quality,
        "native_vs_onnx": parity,
        "per_image": per_image,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"validation_report={report_path}")


def main() -> None:
    args = parse_args()
    if args.resolution < 32 or args.resolution % 32:
        raise ValueError("--resolution must be at least 32 and divisible by 32")
    if args.validation_limit < 0:
        raise ValueError("--validation-limit cannot be negative")
    if (
        not 0.0 <= args.export_confidence <= 1.0
        or not 0.0 <= args.confidence <= 1.0
        or not 0.0 < args.iou <= 1.0
    ):
        raise ValueError(
            "--export-confidence and --confidence must be in [0,1]; --iou must be in (0,1]"
        )
    if args.export_confidence > args.confidence:
        raise ValueError("--export-confidence cannot exceed --confidence")
    if args.validate_image and args.data is None:
        raise ValueError("--validate-image requires --data pointing to the YOLO data.yaml")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Ultralytics is not installed. Run: uv add ultralytics") from error

    yolo = YOLO(str(checkpoint))
    exported = Path(
        yolo.export(
            format="onnx", imgsz=args.resolution, batch=1, dynamic=False,
            simplify=args.simplify, opset=args.opset, conf=args.export_confidence,
            quantize=args.quantize,
        )
    ).resolve()
    output = args.output.resolve()
    names = {int(index): str(name) for index, name in yolo.names.items()}
    canonicalize(exported, output, args.resolution, names)
    print(f"created_onnx={output}")

    generated = [output]
    if not args.skip_embedded_preprocessing:
        bw8 = (args.bw8_output or output.with_name(output.stem + "-bw8.onnx")).resolve()
        c24 = (args.c24_output or output.with_name(output.stem + "-c24.onnx")).resolve()
        wrap_bw8(output, bw8, args.resolution)
        wrap_c24(output, c24, args.resolution)
        generated.extend((bw8, c24))
        print(f"created_bw8={bw8}")
        print(f"created_c24={c24}")
    if args.validate_image:
        data = args.data.expanduser().resolve()
        if not data.is_file():
            raise FileNotFoundError(f"Dataset YAML does not exist: {data}")
        report_path = (
            args.validation_report.expanduser().resolve()
            if args.validation_report
            else output.with_suffix(".validation.json")
        )
        evaluate(
            yolo=yolo,
            checkpoint=checkpoint,
            generated=generated,
            data_path=data,
            requested_image=args.validate_image,
            split=args.validation_split,
            limit=args.validation_limit,
            resolution=args.resolution,
            confidence=args.confidence,
            inference_floor=args.export_confidence,
            iou_threshold=args.iou,
            report_path=report_path,
        )


if __name__ == "__main__":
    main()
