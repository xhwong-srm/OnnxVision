"""Export LibreYOLO PicoDet weights to the repository ONNX detection contract.

The exported graph includes class-aware NMS and emits normalized ``boxes``
[1,N,4], ``scores`` [1,N], and int64 ``class_ids`` [1,N]. Optional BW8 and
C24 variants embed the preprocessing expected by the .NET ObjectDetector.
"""

from __future__ import annotations

import json
import sys
import warnings
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, compose, helper
from PIL import Image


YOLO_SCRIPTS = Path(__file__).resolve().parents[1] / "yolo"
sys.path.insert(0, str(YOLO_SCRIPTS))
from export_yolo26_detection import (  # noqa: E402
    CONTRACT_VERSION,
    Detection,
    add_initializer,
    canonicalize,
    dataset_names,
    dataset_root,
    finish_wrapper,
    image_files_for_split,
    load_dataset_yaml,
    match_detections,
    parity_metrics,
    print_quality,
    quality_metrics,
    read_ground_truth,
    resize,
)


class PicoDetEmbeddedNMSDetector(nn.Module):
    """Add PicoDet's per-level filtering and class-aware NMS.

    PicoDet's export head returns ``[B, anchors, 4 + classes]`` with decoded
    pixel boxes. Native postprocessing keeps the top 1000 class candidates per
    feature level before class-aware NMS, so reproduce that contract here
    instead of treating the four feature levels as one undifferentiated list.
    """

    def __init__(
        self,
        detector: nn.Module,
        *,
        resolution: int,
        num_classes: int,
        level_counts: tuple[int, ...],
        conf: float,
        iou: float,
        max_det: int,
    ):
        super().__init__()
        self.detector = detector
        self.resolution = int(resolution)
        self.num_classes = int(num_classes)
        self.level_counts = tuple(int(count) for count in level_counts)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.nms_pre = 1000

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from torchvision.ops import nms

        raw = self.detector(x).float()
        levels: list[torch.Tensor] = []
        start = 0
        for count in self.level_counts:
            levels.append(raw[0, start:start + count])
            start += count

        boxes_parts: list[torch.Tensor] = []
        scores_parts: list[torch.Tensor] = []
        classes_parts: list[torch.Tensor] = []
        for level in levels:
            boxes_raw = level[:, :4]
            x1 = boxes_raw[:, 0].clamp(min=0.0, max=float(self.resolution))
            y1 = boxes_raw[:, 1].clamp(min=0.0, max=float(self.resolution))
            x2 = boxes_raw[:, 2].clamp(min=0.0, max=float(self.resolution))
            y2 = boxes_raw[:, 3].clamp(min=0.0, max=float(self.resolution))
            boxes_all = torch.stack((x1, y1, x2, y2), dim=1)
            scores_all = level[:, 4:]
            finite_boxes = torch.isfinite(boxes_all).all(dim=1)
            finite_scores = torch.isfinite(scores_all)
            safe_boxes = torch.where(
                torch.isfinite(boxes_all), boxes_all, torch.zeros_like(boxes_all)
            )
            safe_scores = torch.where(
                finite_boxes[:, None] & finite_scores,
                scores_all,
                scores_all.new_full(scores_all.shape, -1.0),
            )
            flat_scores = safe_scores.reshape(-1)
            top_k = min(level.shape[0] * self.num_classes, self.nms_pre)
            top_scores, top_flat_idx = torch.topk(flat_scores, top_k)
            selected = top_scores > self.conf
            top_scores = top_scores[selected]
            top_flat_idx = top_flat_idx[selected]
            anchor_idx = top_flat_idx // self.num_classes
            class_idx = top_flat_idx - anchor_idx * self.num_classes
            cand_boxes = safe_boxes[anchor_idx]
            cand_scores = top_scores
            cand_cls = class_idx.to(boxes_all.dtype)
            valid_boxes = (cand_boxes[:, 2] > cand_boxes[:, 0]) & (
                cand_boxes[:, 3] > cand_boxes[:, 1]
            )
            boxes_parts.append(cand_boxes[valid_boxes])
            scores_parts.append(cand_scores[valid_boxes])
            classes_parts.append(cand_cls[valid_boxes])

        cand_boxes = torch.cat(boxes_parts, dim=0)
        cand_scores = torch.cat(scores_parts, dim=0)
        cand_cls = torch.cat(classes_parts, dim=0)

        all_boxes = raw[0, :, :4]
        safe_all_boxes = torch.where(
            torch.isfinite(all_boxes), all_boxes, torch.zeros_like(all_boxes)
        )
        lo = safe_all_boxes.min()
        step = (safe_all_boxes.max() - lo).clamp(min=1.0) + 1.0
        nms_boxes = (cand_boxes - lo) + cand_cls[:, None] * step
        keep = nms(nms_boxes, cand_scores, self.iou)
        row = torch.cat(
            (cand_boxes[keep], cand_scores[keep, None], cand_cls[keep, None]), dim=1
        )
        padded = torch.cat((row, row.new_zeros(self.max_det, 6)), dim=0)
        top = torch.topk(padded[:, 4], self.max_det).indices
        return padded[top].reshape(1, self.max_det, 6), raw


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="LibreYOLO PicoDet .pt checkpoint")
    parser.add_argument(
        "--resolution", type=int,
        help="Export image size; defaults to the checkpoint's native PicoDet size",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/picodet-detection.onnx"))
    parser.add_argument("--bw8-output", type=Path)
    parser.add_argument("--c24-output", type=Path)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--confidence", type=float, default=0.001, help="Embedded NMS confidence")
    parser.add_argument("--iou", type=float, default=0.70, help="Embedded NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--simplify", action=BooleanOptionalAction, default=True,
        help="Simplify the intermediate ONNX graph (default: enabled)",
    )
    parser.add_argument("--skip-embedded-preprocessing", action="store_true")
    parser.add_argument("--device", default="cpu", help='Export device, for example "cpu" or "0"')
    parser.add_argument(
        "--data", type=Path,
        help="YOLO data.yaml or dataset directory used for post-export agreement evaluation",
    )
    parser.add_argument("--validation-split", default="test", help="Dataset split to evaluate (default: test)")
    parser.add_argument(
        "--validation-limit",
        type=int,
        default=0,
        help="Maximum dataset images to evaluate; 0 evaluates the complete split",
    )
    parser.add_argument(
        "--validation-confidence",
        type=float,
        default=0.25,
        help="Confidence threshold for quality and PT-vs-ONNX agreement metrics",
    )
    parser.add_argument(
        "--validation-iou",
        type=float,
        default=0.50,
        help="IoU threshold for ground-truth and agreement matching",
    )
    parser.add_argument("--validation-report", type=Path, help="JSON agreement report path")
    return parser.parse_args()


def _rename_tensor(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new
        for index, name in enumerate(node.output):
            if name == old:
                node.output[index] = new
    for output in model.graph.output:
        if output.name == old:
            output.name = new


def resolve_dataset_yaml(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "data.yaml"
    if not resolved.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {resolved}")
    return resolved


def _picodet_preprocess_nodes(
    core: onnx.ModelProto,
    source: str,
    resolution: int,
    *,
    bw8: bool,
) -> tuple[list[onnx.NodeProto], str]:
    """Build PicoDet RGB stretch-resize and ImageNet normalization nodes."""
    prefix = "preprocess/"
    sizes = f"{prefix}sizes"
    resized = f"{prefix}resized"
    float_input = f"{prefix}float_input"
    rgb = f"{prefix}rgb"
    normalized = f"{prefix}normalized"

    target_size = add_initializer(
        core, sizes, np.asarray([1, 3, resolution, resolution], dtype=np.int64)
    )
    mean = add_initializer(
        core,
        f"{prefix}mean",
        np.asarray([[[[123.675]], [[116.28]], [[103.53]]]], dtype=np.float32),
    )
    std = add_initializer(
        core,
        f"{prefix}std",
        np.asarray([[[[58.395]], [[57.12]], [[57.375]]]], dtype=np.float32),
    )

    if bw8:
        source_shape = f"{prefix}source_shape"
        expanded_shape = f"{prefix}expanded_shape"
        expand_indices = add_initializer(
            core, f"{prefix}expand_indices", np.asarray([1], dtype=np.int64)
        )
        expand_updates = add_initializer(
            core, f"{prefix}expand_channels", np.asarray([3], dtype=np.int64)
        )
        nodes = [
            helper.make_node("Shape", [source], [source_shape]),
            helper.make_node(
                "ScatterElements",
                [source_shape, expand_indices, expand_updates],
                [expanded_shape],
                axis=0,
            ),
            helper.make_node("Expand", [source, expanded_shape], [float_input]),
            helper.make_node("Cast", [float_input], [float_input + "/cast"], to=TensorProto.FLOAT),
        ]
        resize_source = float_input + "/cast"
    else:
        channel_indices = add_initializer(
            core, f"{prefix}rgb_indices", np.asarray([2, 1, 0], dtype=np.int64)
        )
        nodes = [
            helper.make_node("Transpose", [source], [f"{prefix}bgr"], perm=[0, 3, 1, 2]),
            helper.make_node("Gather", [f"{prefix}bgr", channel_indices], [rgb], axis=1),
            helper.make_node("Cast", [rgb], [float_input], to=TensorProto.FLOAT),
        ]
        resize_source = float_input

    nodes.extend(
        [
            resize(resize_source, resized, target_size),
            helper.make_node("Sub", [resized, mean], [f"{prefix}centered"]),
            helper.make_node("Div", [f"{prefix}centered", std], [normalized]),
        ]
    )
    return nodes, normalized


def _add_picodet_box_remap(
    core: onnx.ModelProto,
    source: str,
    resolution: int,
    max_det: int,
    source_hw_indices: tuple[int, int],
) -> None:
    """Map stretch-resized canvas boxes to normalized raw-image coordinates."""
    _rename_tensor(core, "core/boxes", "core_boxes")
    _rename_tensor(core, "core/scores", "scores")
    _rename_tensor(core, "core/class_ids", "class_ids")
    prefix = "postprocess/"
    shape = f"{prefix}source_shape"
    dimensions = f"{prefix}dimensions"
    box_pixels = f"{prefix}box_pixels"
    box_normalized = f"{prefix}box_normalized"
    shape_indices = add_initializer(
        core, f"{prefix}shape_indices", np.asarray(source_hw_indices, dtype=np.int64)
    )
    canvas = add_initializer(
        core, f"{prefix}canvas", np.asarray(float(resolution), dtype=np.float32)
    )
    zero = add_initializer(core, f"{prefix}zero", np.asarray(0.0, dtype=np.float32))
    one = add_initializer(core, f"{prefix}one", np.asarray(1.0, dtype=np.float32))
    core.graph.node.extend(
        [
            helper.make_node("Shape", [source], [shape]),
            helper.make_node("Gather", [shape, shape_indices], [dimensions], axis=0),
            helper.make_node("Cast", [dimensions], [dimensions + "/float"], to=TensorProto.FLOAT),
            helper.make_node(
                "Concat",
                [dimensions + "/float", dimensions + "/float"],
                [dimensions + "/xyxy"],
                axis=0,
            ),
            helper.make_node("Mul", ["core_boxes", canvas], [box_pixels]),
            helper.make_node("Div", [box_pixels, dimensions + "/xyxy"], [box_normalized]),
            helper.make_node("Clip", [box_normalized, zero, one], ["boxes"]),
        ]
    )
    outputs = []
    for output in core.graph.output:
        if output.name == "core_boxes":
            outputs.append(helper.make_tensor_value_info(
                "boxes", TensorProto.FLOAT, [1, max_det, 4]
            ))
        else:
            outputs.append(output)
    del core.graph.output[:]
    core.graph.output.extend(outputs)


def prepare_picodet_core(path: Path) -> tuple[onnx.ModelProto, str, dict[str, str]]:
    """Prefix the core graph while preserving connected graph outputs."""
    model = onnx.load(path)
    metadata = {item.key: item.value for item in model.metadata_props}
    core = compose.add_prefix(model, "core/")
    input_name = core.graph.input[0].name
    del core.graph.input[:]
    return core, input_name, metadata


def _wrap_picodet_input(
    core_path: Path,
    output: Path,
    resolution: int,
    max_det: int,
    *,
    bw8: bool,
) -> None:
    core, core_input, metadata = prepare_picodet_core(core_path)
    input_name = "images_bw8_uint8_nchw" if bw8 else "images_c24_uint8_nhwc_bgr"
    input_shape = [1, 1, "height", "width"] if bw8 else [1, "height", "width", 3]
    core.graph.input.append(
        helper.make_tensor_value_info(input_name, TensorProto.UINT8, input_shape)
    )
    nodes, normalized = _picodet_preprocess_nodes(
        core, input_name, resolution, bw8=bw8
    )
    source_hw_indices = (3, 2) if bw8 else (2, 1)
    _add_picodet_box_remap(core, input_name, resolution, max_det, source_hw_indices)
    metadata.update(
        {
            "vision_task": "object_detection",
            "detection_contract": CONTRACT_VERSION,
            "box_format": "xyxy",
            "box_coordinates": "normalized_original",
            "nms_required": "false",
            "resize_mode": "stretch",
            "preprocess": "picodet_rgb_stretch_imagenet",
            "embedded_preprocessing": "true",
            "source_model": "libreyolo-picodet",
        }
    )
    # The returned name is deliberately checked here: a future edit that
    # changes the preprocessing output without updating the core input would
    # otherwise produce a valid but disconnected ONNX graph.
    if normalized != core_input:
        nodes.append(helper.make_node("Identity", [normalized], [core_input]))
    finish_wrapper(core, nodes, output, metadata)


def wrap_picodet_bw8(core_path: Path, output: Path, resolution: int, max_det: int) -> None:
    _wrap_picodet_input(core_path, output, resolution, max_det, bw8=True)


def wrap_picodet_c24(core_path: Path, output: Path, resolution: int, max_det: int) -> None:
    _wrap_picodet_input(core_path, output, resolution, max_det, bw8=False)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def run_native(
    model: Any,
    image_path: Path,
    resolution: int,
    confidence: float,
    nms_iou: float,
    max_det: int = 300,
    grayscale: bool = False,
) -> tuple[Detection, ...]:
    with Image.open(image_path) as image:
        width, height = image.size
        source: Any = image_path
        if grayscale:
            source = Image.fromarray(np.asarray(image.convert("L")), mode="L").convert("RGB")
    result = model.predict(
        source=source,
        imgsz=resolution,
        conf=confidence,
        iou=nms_iou,
        max_det=max_det,
        color_format="rgb",
    )
    if isinstance(result, (list, tuple)):
        result = result[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return ()
    xyxy = _to_numpy(boxes.xyxy)
    scores = _to_numpy(boxes.conf)
    classes = _to_numpy(boxes.cls)
    scale = np.asarray([width, height, width, height], dtype=np.float32)
    return tuple(
        Detection(np.asarray(box, dtype=np.float32) / scale, float(score), int(class_id))
        for box, score, class_id in zip(xyxy, scores, classes)
        if np.isfinite(score) and float(score) >= confidence
    )


def run_onnx(
    session: Any,
    image_path: Path,
    resolution: int,
    confidence: float,
) -> tuple[Detection, ...]:
    from libreyolo.models.picodet.utils import preprocess_numpy

    input_info = session.get_inputs()[0]
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        width, height = image.size
    if input_info.type == "tensor(float)":
        chw, _ = preprocess_numpy(rgb, input_size=resolution)
        value = chw[None]
    elif input_info.type == "tensor(uint8)" and input_info.shape[-1] == 3:
        value = rgb[..., ::-1].copy()[None]
    elif input_info.type == "tensor(uint8)":
        value = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.uint8)[None, None]
    else:
        raise ValueError(f"Unsupported PicoDet ONNX input: {input_info.type} {input_info.shape}")
    outputs = {
        item.name: np.asarray(result)
        for item, result in zip(session.get_outputs(), session.run(None, {input_info.name: value}))
    }
    boxes = np.asarray(outputs["boxes"])[0].astype(np.float32)
    if input_info.type == "tensor(float)":
        # The float core consumes the stretched square tensor. Convert its
        # normalized canvas coordinates back to the original image frame.
        denominator = np.asarray(
            [width, height, width, height],
            dtype=np.float32,
        )
        boxes = boxes * float(resolution) / denominator
    boxes = np.clip(boxes, 0.0, 1.0)
    scores = np.asarray(outputs["scores"])[0]
    classes = np.asarray(outputs["class_ids"])[0]
    return tuple(
        Detection(box, float(score), int(class_id))
        for box, score, class_id in zip(boxes, scores, classes)
        if np.isfinite(score) and float(score) >= confidence
    )


def evaluate(
    model: Any,
    checkpoint: Path,
    generated: list[Path],
    data_path: Path,
    split: str,
    limit: int,
    resolution: int,
    max_det: int,
    inference_floor: float,
    nms_iou: float,
    confidence: float,
    iou_threshold: float,
    report_path: Path,
) -> None:
    import onnxruntime as ort

    document = load_dataset_yaml(data_path)
    names = dataset_names(document)
    if split not in document:
        raise ValueError(f"Dataset YAML does not define split '{split}': {data_path}")
    images = image_files_for_split(dataset_root(data_path, document), document[split])
    if limit:
        images = images[:limit]
    if not images:
        raise ValueError(f"Dataset split '{split}' contains no supported images")
    samples = [read_ground_truth(path) for path in images]
    print(
        f"validation_dataset={data_path}; split={split}; images={len(samples)}; "
        f"confidence={confidence}; iou={iou_threshold}"
    )

    predictions: dict[str, dict[Path, tuple[Detection, ...]]] = {
        "native_pt": {},
        "native_pt_bw8": {},
    }
    references: dict[str, str] = {}
    for index, sample in enumerate(samples, 1):
        predictions["native_pt"][sample.image] = run_native(
            model, sample.image, resolution, inference_floor,
            nms_iou=nms_iou, max_det=max_det,
        )
        predictions["native_pt_bw8"][sample.image] = run_native(
            model, sample.image, resolution, inference_floor,
            nms_iou=nms_iou, max_det=max_det, grayscale=True,
        )
        print(f"validation_progress={index}/{len(samples)} image={sample.image.name}")
    for path in generated:
        session = ort.InferenceSession(str(path.resolve()), providers=["CPUExecutionProvider"])
        backend = path.name
        input_info = session.get_inputs()[0]
        references[backend] = (
            "native_pt_bw8"
            if input_info.type == "tensor(uint8)" and input_info.shape[-1] != 3
            else "native_pt"
        )
        predictions[path.name] = {
            sample.image: run_onnx(session, sample.image, resolution, inference_floor)
            for sample in samples
        }

    quality = {
        backend: quality_metrics(samples, values, names, iou_threshold, confidence)
        for backend, values in predictions.items()
    }
    thresholded = {
        backend: {
            path: tuple(item for item in values[path] if item.score >= confidence)
            for path in values
        }
        for backend, values in predictions.items()
    }
    parity = {
        backend: parity_metrics(thresholded[references[backend]], values, iou_threshold)
        for backend, values in thresholded.items()
        if backend in references
    }
    per_image: list[dict[str, Any]] = []
    for sample in samples:
        item: dict[str, Any] = {
            "image": str(sample.image),
            "annotation": str(sample.label),
            "ground_truth": len(sample.ground_truth),
            "backends": {},
        }
        for backend, values in thresholded.items():
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
                {sample.image: thresholded[references[backend]][sample.image]},
                {sample.image: values[sample.image]},
                iou_threshold,
            )
            for backend, values in thresholded.items()
            if backend in references
        }
        per_image.append(item)
    for backend, values in quality.items():
        print_quality(backend, values)
    for backend, values in parity.items():
        print(
            f"parity/{backend} reference={references[backend]}: matched={values['matched_detections']} "
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
        "models": [str(path) for path in generated],
        "agreement_references": references,
        "settings": {
            "resolution": resolution,
            "inference_floor": inference_floor,
            "confidence": confidence,
            "nms_iou": nms_iou,
            "iou": iou_threshold,
            "native_preprocess": "libreyolo_picodet_rgb_stretch_imagenet",
            "onnx_provider": "CPUExecutionProvider",
        },
        "interpretation": {
            "quality": "Predictions compared with YOLO ground-truth annotations.",
            "parity": "Each ONNX result compared with native PyTorch on the same test image and confidence floor.",
        },
        "quality": quality,
        "native_vs_onnx": parity,
        "per_image": per_image,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"validation_report={report_path}")


def simplify_onnx(path: Path) -> None:
    try:
        from onnxslim import slim
    except ImportError as error:
        raise RuntimeError("onnxslim is required for --simplify") from error
    original = onnx.load(path)
    metadata = {item.key: item.value for item in original.metadata_props}
    try:
        simplified = slim(original)
        if not isinstance(simplified, onnx.ModelProto):
            raise TypeError(f"onnxslim returned {type(simplified).__name__}, expected ModelProto")
        existing_metadata = {item.key: item.value for item in simplified.metadata_props}
        existing_metadata.update(metadata)
        del simplified.metadata_props[:]
        for key, value in existing_metadata.items():
            item = simplified.metadata_props.add()
            item.key, item.value = key, value
        onnx.checker.check_model(simplified)
        temporary = path.with_name(path.name + ".slim.tmp")
        onnx.save(simplified, temporary)
        temporary.replace(path)
    except Exception as error:
        temporary = path.with_name(path.name + ".slim.tmp")
        if temporary.exists():
            temporary.unlink()
        raise RuntimeError(f"Failed to simplify ONNX model {path}: {error}") from error
    print(f"simplified_onnx={path}")


def validate_onnx(path: Path, resolution: int, max_det: int) -> None:
    """Run a deterministic CPU smoke test without provider auto-selection."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("warning: onnxruntime is unavailable; skipping ONNX smoke validation")
        return
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    if input_info.type == "tensor(float)":
        value = np.zeros((1, 3, resolution, resolution), dtype=np.float32)
    elif input_info.type == "tensor(uint8)" and input_info.shape[-1] == 3:
        value = np.zeros((1, resolution, resolution, 3), dtype=np.uint8)
    elif input_info.type == "tensor(uint8)":
        value = np.zeros((1, 1, resolution, resolution), dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported PicoDet ONNX input {input_info.type} {input_info.shape}")
    results = session.run(None, {input_info.name: value})
    output_names = [item.name for item in session.get_outputs()]
    expected = ["boxes", "scores", "class_ids"]
    if output_names != expected:
        raise ValueError(f"Unexpected validated ONNX outputs: {output_names}")
    shapes = [tuple(np.asarray(result).shape) for result in results]
    if shapes != [(1, max_det, 4), (1, max_det), (1, max_det)]:
        raise ValueError(f"Unexpected validated ONNX output shapes: {shapes}")
    if not all(np.isfinite(np.asarray(result, dtype=np.float32)).all() for result in results):
        raise ValueError("ONNX smoke validation produced non-finite values")
    print(f"validated_onnx={path}; provider=CPUExecutionProvider; outputs={shapes}")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.confidence <= 1.0 or not 0.0 < args.iou <= 1.0:
        raise ValueError("--confidence must be in [0,1] and --iou in (0,1]")
    if args.max_det < 1 or args.validation_limit < 0:
        raise ValueError("--max-det must be positive and --validation-limit cannot be negative")
    if not 0.0 <= args.validation_confidence <= 1.0 or not 0.0 < args.validation_iou <= 1.0:
        raise ValueError("--validation-confidence must be in [0,1] and --validation-iou in (0,1]")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    try:
        from libreyolo import LibreYOLO
    except ImportError as error:
        raise RuntimeError("LibreYOLO is not installed. Run: uv add libreyolo") from error

    model = LibreYOLO(str(checkpoint), device=args.device, task="detect")
    if model._get_model_name() != "picodet":
        raise ValueError(
            f"Expected a PicoDet checkpoint, received family {model._get_model_name()!r}"
        )
    resolution = args.resolution or int(model.input_size)
    if resolution < 32 or resolution % 32:
        raise ValueError("--resolution must be at least 32 and divisible by 32")
    model.model.eval()
    model.model.head.export = True
    names = {int(index): str(name) for index, name in model.names.items()}
    device = next(model.model.parameters()).device
    dummy = torch.zeros(1, 3, resolution, resolution, device=device)
    with torch.inference_mode():
        raw_probe = model.model(dummy)
    if raw_probe.ndim != 3 or raw_probe.shape[0] != 1 or raw_probe.shape[2] != 4 + len(names):
        raise ValueError(f"Unexpected PicoDet export tensor shape: {tuple(raw_probe.shape)}")
    strides = tuple(int(stride) for stride in model.model.head.strides)
    level_counts = tuple((resolution // stride) ** 2 for stride in strides)
    if sum(level_counts) != int(raw_probe.shape[1]):
        raise ValueError(
            "PicoDet export level counts do not match the model output: "
            f"levels={level_counts}; output={tuple(raw_probe.shape)}"
        )
    export_model = PicoDetEmbeddedNMSDetector(
        model.model,
        resolution=resolution,
        num_classes=len(names),
        level_counts=level_counts,
        conf=args.confidence,
        iou=args.iou,
        max_det=args.max_det,
    ).eval()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate = output.with_name(output.stem + ".libreyolo.onnx")
    try:
        with warnings.catch_warnings():
            # PyTorch 2.11/torchvision currently emits these upstream
            # deprecations while lowering torchvision NMS; they are unrelated
            # to the exported graph and would otherwise obscure real failures.
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                module=r"torchvision\._meta_registrations",
                message=r".*create_unbacked_symint.*",
            )
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                module=r"copyreg",
                message=r".*LeafSpec.*",
            )
            torch.onnx.export(
                export_model,
                (dummy,),
                str(intermediate),
                input_names=["images"],
                output_names=["output", "raw"],
                opset_version=args.opset,
                dynamo=True,
                optimize=True,
                verify=False,
                external_data=False,
                verbose=False,
            )
    except Exception as modern_error:
        print(
            "warning: torch.export ONNX export failed; retrying the legacy exporter: "
            f"{type(modern_error).__name__}: {modern_error}"
        )
        torch.onnx.export(
            export_model,
            (dummy,),
            str(intermediate),
            input_names=["images"],
            output_names=["output", "raw"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
    if args.simplify:
        simplify_onnx(intermediate)

    intermediate_graph = onnx.load(intermediate)
    if [item.name for item in intermediate_graph.graph.output] != ["output", "raw"]:
        raise ValueError(
            "Unexpected LibreYOLO export outputs: "
            f"{[item.name for item in intermediate_graph.graph.output]}"
        )
    del intermediate_graph.graph.output[1:]
    onnx.save(intermediate_graph, intermediate)
    canonicalize(intermediate, output, resolution, names)
    graph = onnx.load(output)
    metadata = {item.key: item.value for item in graph.metadata_props}
    metadata.update(
        {
            "source_model": "libreyolo-picodet",
            "nms": "true",
            "nms_required": "false",
            "nms_conf": str(args.confidence),
            "nms_iou": str(args.iou),
            "max_det": str(args.max_det),
            "libreyolo_names": json.dumps(names),
            "preprocess": "picodet_rgb_stretch_imagenet",
            "resize_mode": "stretch",
            "box_coordinates": "normalized_canvas",
            "picodet_strides": json.dumps(strides),
            "picodet_nms_pre": "1000",
        }
    )
    del graph.metadata_props[:]
    for key, value in metadata.items():
        item = graph.metadata_props.add()
        item.key, item.value = key, value
    onnx.checker.check_model(graph)
    onnx.save(graph, output)
    if args.simplify:
        simplify_onnx(output)
    validate_onnx(output, resolution, args.max_det)
    print(f"created_onnx={output}")

    generated = [output]
    if not args.skip_embedded_preprocessing:
        bw8 = (args.bw8_output or output.with_name(output.stem + "-bw8.onnx")).resolve()
        c24 = (args.c24_output or output.with_name(output.stem + "-c24.onnx")).resolve()
        wrap_picodet_bw8(output, bw8, resolution, args.max_det)
        wrap_picodet_c24(output, c24, resolution, args.max_det)
        if args.simplify:
            simplify_onnx(bw8)
            simplify_onnx(c24)
        validate_onnx(bw8, resolution, args.max_det)
        validate_onnx(c24, resolution, args.max_det)
        generated.extend((bw8, c24))
        print(f"created_bw8={bw8}")
        print(f"created_c24={c24}")

    if args.data:
        data = resolve_dataset_yaml(args.data)
        # Export mode changes PicoHead.forward() from native level lists to
        # decoded flat outputs. Restore native mode before agreement testing.
        model.model.head.export = False
        validation_model = model
        if device.type != "cpu":
            validation_model = LibreYOLO(str(checkpoint), device="cpu", task="detect")
        report_path = (
            args.validation_report.expanduser().resolve()
            if args.validation_report
            else output.with_suffix(".validation.json")
        )
        evaluate(
            model=validation_model,
            checkpoint=checkpoint,
            generated=generated,
            data_path=data,
            split=args.validation_split,
            limit=args.validation_limit,
            resolution=resolution,
            max_det=args.max_det,
            inference_floor=args.confidence,
            nms_iou=args.iou,
            confidence=args.validation_confidence,
            iou_threshold=args.validation_iou,
            report_path=report_path,
        )


if __name__ == "__main__":
    main()
