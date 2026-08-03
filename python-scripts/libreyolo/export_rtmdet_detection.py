"""Export LibreYOLO RTMDet weights to the repository ONNX detection contract.

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
from onnx import TensorProto, helper
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
    prepare_core,
    print_quality,
    quality_metrics,
    read_ground_truth,
    resize,
)


class RTMDetExportLayout(nn.Module):
    """Adapt RTMDet's [B,N,4+C] output to LibreYOLO's NMS layout."""

    def __init__(self, detector: nn.Module):
        super().__init__()
        self.detector = detector

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.detector(images).permute(0, 2, 1)


class RTMDetEmbeddedNMSDetector(nn.Module):
    """Add fixed-shape class-aware NMS without tracing tensor dimensions.

    ``libreyolo.export.nms.EmbeddedNMSDetector`` is shared by several model
    families and uses Python ``float``/``bool`` conversions of input and
    output shapes. Those conversions are safe for its fixed-size export, but
    produce noisy legacy-tracer warnings and can freeze the wrong value when
    a shape changes. RTMDet already has a fixed export resolution, so keep the
    shape-dependent values as explicit Python constants and use integer
    arithmetic for the flattened class index.
    """

    def __init__(
        self,
        detector: nn.Module,
        *,
        resolution: int,
        num_classes: int,
        candidate_count: int,
        conf: float,
        iou: float,
        max_det: int,
    ):
        super().__init__()
        self.detector = detector
        self.resolution = int(resolution)
        self.num_classes = int(num_classes)
        self.candidate_count = int(candidate_count)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.max_nms = min(self.candidate_count, max(self.max_det, 30000))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from torchvision.ops import nms

        raw = self.detector(x)
        pred = raw[0].transpose(0, 1).float()
        boxes_raw = pred[:, :4]
        x1 = boxes_raw[:, 0].clamp(min=0.0, max=float(self.resolution))
        y1 = boxes_raw[:, 1].clamp(min=0.0, max=float(self.resolution))
        x2 = boxes_raw[:, 2].clamp(min=0.0, max=float(self.resolution))
        y2 = boxes_raw[:, 3].clamp(min=0.0, max=float(self.resolution))
        boxes_all = torch.stack((x1, y1, x2, y2), dim=1)
        scores_all = pred[:, 4:]
        finite_boxes = torch.isfinite(boxes_all).all(dim=1)
        finite_scores = torch.isfinite(scores_all)
        safe_boxes_all = torch.where(
            torch.isfinite(boxes_all), boxes_all, torch.zeros_like(boxes_all)
        )
        safe_scores_all = torch.where(
            finite_boxes[:, None] & finite_scores,
            scores_all,
            scores_all.new_full(scores_all.shape, -1.0),
        )

        flat_scores = safe_scores_all.reshape(-1)
        top_scores, top_flat_idx = torch.topk(flat_scores, self.max_nms)
        selected = top_scores > self.conf
        top_scores = top_scores[selected]
        top_flat_idx = top_flat_idx[selected]
        anchor_idx = top_flat_idx // self.num_classes
        class_idx = top_flat_idx - anchor_idx * self.num_classes
        cand_boxes = safe_boxes_all[anchor_idx]
        cand_scores = top_scores
        cand_cls = class_idx.to(boxes_all.dtype)
        valid_boxes = (cand_boxes[:, 2] > cand_boxes[:, 0]) & (
            cand_boxes[:, 3] > cand_boxes[:, 1]
        )
        cand_boxes = cand_boxes[valid_boxes]
        cand_scores = cand_scores[valid_boxes]
        cand_cls = cand_cls[valid_boxes]

        lo = safe_boxes_all.min()
        step = (safe_boxes_all.max() - lo).clamp(min=1.0) + 1.0
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
    parser.add_argument("--checkpoint", required=True, type=Path, help="LibreYOLO RTMDet .pt checkpoint")
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--output", type=Path, default=Path("artifacts/rtmdet-detection.onnx"))
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
    parser.add_argument("--data", type=Path, help="YOLO data.yaml used for post-export agreement evaluation")
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


def _rtmdet_preprocess_nodes(
    core: onnx.ModelProto,
    source: str,
    resolution: int,
    *,
    bw8: bool,
) -> tuple[list[onnx.NodeProto], str]:
    """Build dynamic-shape RTMDet BGR letterbox and normalization nodes."""
    prefix = "preprocess/"
    source_shape = f"{prefix}source_shape"
    source_hw = f"{prefix}source_hw"
    source_hw_float = f"{prefix}source_hw_float"
    ratio_axes = f"{prefix}ratio_axes"
    scaled_float = f"{prefix}scaled_float"
    scaled_hw = f"{prefix}scaled_hw"
    sizes = f"{prefix}sizes"
    padded = f"{prefix}padded"
    resized = f"{prefix}resized"
    normalized = f"{prefix}normalized"

    shape = helper.make_node("Shape", [source], [source_shape])
    hw_indices = add_initializer(
        core,
        f"{prefix}hw_indices",
        np.asarray([2, 3] if bw8 else [1, 2], dtype=np.int64),
    )
    target_float = add_initializer(
        core, f"{prefix}target_float", np.asarray([resolution, resolution], dtype=np.float32)
    )
    target_int = add_initializer(
        core, f"{prefix}target_int", np.asarray([resolution, resolution], dtype=np.int64)
    )
    ratio_axes_value = add_initializer(
        core, ratio_axes, np.asarray([0], dtype=np.int64)
    )
    zero4 = add_initializer(core, f"{prefix}zero4", np.zeros(4, dtype=np.int64))
    zero2 = add_initializer(core, f"{prefix}zero2", np.zeros(2, dtype=np.int64))
    one = add_initializer(core, f"{prefix}batch", np.asarray([1], dtype=np.int64))
    channels = add_initializer(core, f"{prefix}channels", np.asarray([3], dtype=np.int64))
    pad_value = add_initializer(
        core, f"{prefix}pad_value", np.asarray(114.0, dtype=np.float32)
    )
    mean = add_initializer(
        core,
        f"{prefix}mean",
        np.asarray([[[[103.53]], [[116.28]], [[123.675]]]], dtype=np.float32),
    )
    std = add_initializer(
        core,
        f"{prefix}std",
        np.asarray([[[[57.375]], [[57.12]], [[58.395]]]], dtype=np.float32),
    )

    nodes: list[onnx.NodeProto] = [
        shape,
        helper.make_node("Gather", [source_shape, hw_indices], [source_hw], axis=0),
        helper.make_node("Cast", [source_hw], [source_hw_float], to=TensorProto.FLOAT),
        helper.make_node("Div", [target_float, source_hw_float], [f"{prefix}ratios"]),
        helper.make_node(
            "ReduceMin", [f"{prefix}ratios", ratio_axes_value], [f"{prefix}ratio"], keepdims=0
        ),
        helper.make_node("Mul", [source_hw_float, f"{prefix}ratio"], [scaled_float]),
        helper.make_node("Floor", [scaled_float], [f"{prefix}scaled_floor"]),
        helper.make_node("Cast", [f"{prefix}scaled_floor"], [scaled_hw], to=TensorProto.INT64),
        helper.make_node("Concat", [one, channels, scaled_hw], [sizes], axis=0),
    ]

    if bw8:
        expand_indices = add_initializer(
            core, f"{prefix}expand_indices", np.asarray([1], dtype=np.int64)
        )
        expand_updates = add_initializer(
            core, f"{prefix}expand_channels", np.asarray([3], dtype=np.int64)
        )
        expanded_shape = f"{prefix}expanded_shape"
        nodes.extend(
            [
                helper.make_node(
                    "ScatterElements",
                    [source_shape, expand_indices, expand_updates],
                    [expanded_shape],
                    axis=0,
                ),
                helper.make_node("Expand", [source, expanded_shape], [f"{prefix}expanded"]),
                helper.make_node("Cast", [f"{prefix}expanded"], [f"{prefix}float"], to=TensorProto.FLOAT),
            ]
        )
        resize_source = f"{prefix}float"
    else:
        nodes.extend(
            [
                helper.make_node(
                    "Transpose", [source], [f"{prefix}bgr"], perm=[0, 3, 1, 2]
                ),
                helper.make_node("Cast", [f"{prefix}bgr"], [f"{prefix}float"], to=TensorProto.FLOAT),
            ]
        )
        resize_source = f"{prefix}float"

    nodes.extend(
        [
            resize(resize_source, resized, sizes),
            helper.make_node(
                "Sub", [target_int, scaled_hw], [f"{prefix}padding_hw"]
            ),
            helper.make_node(
                "Concat",
                [zero4, zero2, f"{prefix}padding_hw"],
                [f"{prefix}pads"],
                axis=0,
            ),
            helper.make_node(
                "Pad",
                [resized, f"{prefix}pads", pad_value],
                [padded],
                mode="constant",
            ),
            helper.make_node("Sub", [padded, mean], [f"{prefix}centered"]),
            helper.make_node("Div", [f"{prefix}centered", std], [normalized]),
        ]
    )
    return nodes, normalized


def _add_rtmdet_box_remap(
    core: onnx.ModelProto,
    source: str,
    resolution: int,
    max_det: int,
    ratio_name: str = "preprocess/ratio",
) -> None:
    """Map padded-canvas boxes to normalized coordinates of the raw image."""
    _rename_tensor(core, "boxes", "core_boxes")
    prefix = "postprocess/"
    shape = f"{prefix}source_shape"
    hw = f"{prefix}hw"
    hw_float = f"{prefix}hw_float"
    width_height = f"{prefix}width_height"
    denominator = f"{prefix}denominator"
    box_pixels = f"{prefix}box_pixels"
    box_normalized = f"{prefix}box_normalized"
    boxes = "boxes"
    shape_indices = add_initializer(
        core, f"{prefix}shape_indices", np.asarray([1, 0], dtype=np.int64)
    )
    canvas = add_initializer(
        core, f"{prefix}canvas", np.asarray(float(resolution), dtype=np.float32)
    )
    zero = add_initializer(core, f"{prefix}zero", np.asarray(0.0, dtype=np.float32))
    one = add_initializer(core, f"{prefix}one", np.asarray(1.0, dtype=np.float32))
    core.graph.node.extend(
        [
            helper.make_node("Shape", [source], [shape]),
            helper.make_node("Gather", [shape, shape_indices], [hw], axis=0),
            helper.make_node("Cast", [hw], [hw_float], to=TensorProto.FLOAT),
            helper.make_node(
                "Gather", [hw_float, shape_indices], [width_height], axis=0
            ),
            helper.make_node(
                "Mul", [width_height, ratio_name], [f"{prefix}scaled_width_height"]
            ),
            helper.make_node(
                "Concat",
                [f"{prefix}scaled_width_height", f"{prefix}scaled_width_height"],
                [denominator],
                axis=0,
            ),
            helper.make_node("Mul", ["core_boxes", canvas], [box_pixels]),
            helper.make_node("Div", [box_pixels, denominator], [box_normalized]),
            helper.make_node("Clip", [box_normalized, zero, one], [boxes]),
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


def _wrap_rtmdet_input(
    core_path: Path,
    output: Path,
    resolution: int,
    max_det: int,
    *,
    bw8: bool,
) -> None:
    core, core_input, metadata = prepare_core(core_path)
    input_name = "images_bw8_uint8_nchw" if bw8 else "images_c24_uint8_nhwc_bgr"
    input_shape = [1, 1, "height", "width"] if bw8 else [1, "height", "width", 3]
    core.graph.input.append(
        helper.make_tensor_value_info(input_name, TensorProto.UINT8, input_shape)
    )
    nodes, normalized = _rtmdet_preprocess_nodes(
        core, input_name, resolution, bw8=bw8
    )
    _add_rtmdet_box_remap(core, input_name, resolution, max_det)
    metadata.update(
        {
            "vision_task": "object_detection",
            "detection_contract": CONTRACT_VERSION,
            "box_format": "xyxy",
            "box_coordinates": "normalized_original",
            "nms_required": "false",
            "resize_mode": "letterbox",
            "preprocess": "rtmdet_bgr_letterbox",
            "embedded_preprocessing": "true",
            "source_model": "libreyolo-rtmdet",
        }
    )
    # The returned name is deliberately checked here: a future edit that
    # changes the preprocessing output without updating the core input would
    # otherwise produce a valid but disconnected ONNX graph.
    if normalized != core_input:
        nodes.append(helper.make_node("Identity", [normalized], [core_input]))
    finish_wrapper(core, nodes, output, metadata)


def wrap_rtmdet_bw8(core_path: Path, output: Path, resolution: int, max_det: int) -> None:
    _wrap_rtmdet_input(core_path, output, resolution, max_det, bw8=True)


def wrap_rtmdet_c24(core_path: Path, output: Path, resolution: int, max_det: int) -> None:
    _wrap_rtmdet_input(core_path, output, resolution, max_det, bw8=False)


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
        max_det=300,
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
    from libreyolo.models.rtmdet.utils import preprocess_numpy

    input_info = session.get_inputs()[0]
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        width, height = image.size
    if input_info.type == "tensor(float)":
        chw, ratio = preprocess_numpy(rgb, input_size=resolution)
        value = chw[None]
    elif input_info.type == "tensor(uint8)" and input_info.shape[-1] == 3:
        value = rgb[..., ::-1].copy()[None]
        ratio = None
    elif input_info.type == "tensor(uint8)":
        value = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.uint8)[None, None]
        ratio = None
    else:
        raise ValueError(f"Unsupported RTMDet ONNX input: {input_info.type} {input_info.shape}")
    outputs = {
        item.name: np.asarray(result)
        for item, result in zip(session.get_outputs(), session.run(None, {input_info.name: value}))
    }
    boxes = np.asarray(outputs["boxes"])[0].astype(np.float32)
    if ratio is not None:
        # The float core consumes the padded square tensor. Convert its
        # normalized canvas coordinates back to the original image frame.
        denominator = np.asarray(
            [ratio * width, ratio * height, ratio * width, ratio * height],
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
            model, sample.image, resolution, inference_floor, nms_iou=nms_iou
        )
        predictions["native_pt_bw8"][sample.image] = run_native(
            model, sample.image, resolution, inference_floor, nms_iou=nms_iou, grayscale=True
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
            "native_preprocess": "libreyolo_rtmdet_bgr_letterbox",
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
    except ImportError:
        print("warning: onnxslim is unavailable; skipping simplification")
        return
    simplified = slim(onnx.load(path))
    onnx.checker.check_model(simplified)
    onnx.save(simplified, path)


def validate_core_onnx(path: Path, resolution: int, max_det: int) -> None:
    """Run a deterministic CPU smoke test without provider auto-selection."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("warning: onnxruntime is unavailable; skipping ONNX smoke validation")
        return
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    if input_info.type != "tensor(float)":
        raise ValueError(f"Expected float RTMDet core input, received {input_info.type}")
    value = np.zeros((1, 3, resolution, resolution), dtype=np.float32)
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
    if args.resolution < 32 or args.resolution % 32:
        raise ValueError("--resolution must be at least 32 and divisible by 32")
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
    if model._get_model_name() != "rtmdet":
        raise ValueError(f"Expected an RTMDet checkpoint, received family {model._get_model_name()!r}")
    model.model.eval()
    model.model.head.export = True
    names = {int(index): str(name) for index, name in model.names.items()}
    device = next(model.model.parameters()).device
    dummy = torch.zeros(1, 3, args.resolution, args.resolution, device=device)
    with torch.inference_mode():
        raw_probe = RTMDetExportLayout(model.model)(dummy)
    if raw_probe.ndim != 3 or raw_probe.shape[0] != 1 or raw_probe.shape[1] != 4 + len(names):
        raise ValueError(f"Unexpected RTMDet export tensor shape: {tuple(raw_probe.shape)}")
    candidate_count = int(raw_probe.shape[2])
    export_model = RTMDetEmbeddedNMSDetector(
        RTMDetExportLayout(model.model),
        resolution=args.resolution,
        num_classes=len(names),
        candidate_count=candidate_count,
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
    canonicalize(intermediate, output, args.resolution, names)
    graph = onnx.load(output)
    metadata = {item.key: item.value for item in graph.metadata_props}
    metadata.update(
        {
            "source_model": "libreyolo-rtmdet",
            "nms": "true",
            "nms_required": "false",
            "nms_conf": str(args.confidence),
            "nms_iou": str(args.iou),
            "max_det": str(args.max_det),
            "libreyolo_names": json.dumps(names),
            "preprocess": "rtmdet_bgr_letterbox",
            "resize_mode": "letterbox",
            "box_coordinates": "normalized_padded_canvas",
        }
    )
    del graph.metadata_props[:]
    for key, value in metadata.items():
        item = graph.metadata_props.add()
        item.key, item.value = key, value
    onnx.checker.check_model(graph)
    onnx.save(graph, output)
    validate_core_onnx(output, args.resolution, args.max_det)
    print(f"created_onnx={output}")

    generated = [output]
    if not args.skip_embedded_preprocessing:
        bw8 = (args.bw8_output or output.with_name(output.stem + "-bw8.onnx")).resolve()
        c24 = (args.c24_output or output.with_name(output.stem + "-c24.onnx")).resolve()
        wrap_rtmdet_bw8(output, bw8, args.resolution, args.max_det)
        wrap_rtmdet_c24(output, c24, args.resolution, args.max_det)
        generated.extend((bw8, c24))
        print(f"created_bw8={bw8}")
        print(f"created_c24={c24}")

    if args.data:
        data = args.data.expanduser().resolve()
        if not data.is_file():
            raise FileNotFoundError(f"Dataset YAML does not exist: {data}")
        # Export mode changes RTMDetHead.forward() from native tuple outputs to
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
            resolution=args.resolution,
            inference_floor=args.confidence,
            nms_iou=args.iou,
            confidence=args.validation_confidence,
            iou_threshold=args.validation_iou,
            report_path=report_path,
        )


if __name__ == "__main__":
    main()
