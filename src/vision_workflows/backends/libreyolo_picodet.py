from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from ..domain.errors import ConfigurationError
from ..workflows.context import optional_import


def export_picodet_core(
    model: Any,
    output: Path,
    *,
    image_size: int,
    num_classes: int,
    opset: int,
    simplify: bool,
    confidence: float,
    iou: float,
    max_det: int = 300,
) -> Path:
    """Export PicoDet with its native decode and embedded class-aware NMS.

    LibreYOLO's generic ONNX exporter only permits ``nms=True`` for YOLO9.
    PicoDet's export head already decodes boxes, but it emits the provider raw
    tensor ``[B, anchors, 4 + classes]``.  This adapter reproduces PicoDet's
    per-level top-k filtering and wraps that tensor in the same fixed
    ``[1, max_det, 6]`` end-to-end layout used by the generic contract path.
    """
    if image_size < 32 or image_size % 32:
        raise ConfigurationError(
            f"PicoDet export image size must be at least 32 and divisible by 32; got {image_size}"
        )
    if num_classes <= 0:
        raise ConfigurationError("PicoDet export requires at least one class")
    if not 0.0 <= confidence <= 1.0:
        raise ConfigurationError("PicoDet export confidence must be in [0, 1]")
    if not 0.0 < iou <= 1.0:
        raise ConfigurationError("PicoDet export IoU must be in (0, 1]")
    if max_det <= 0:
        raise ConfigurationError("PicoDet export max_det must be positive")

    torch = optional_import("torch")
    onnx = optional_import("onnx")
    detector = getattr(model, "model", None)
    head = getattr(detector, "head", None)
    if detector is None or head is None or not hasattr(head, "export"):
        raise ConfigurationError("the loaded checkpoint is not an exportable PicoDet model")
    if getattr(model, "_get_model_name", lambda: "")() != "picodet":
        raise ConfigurationError("the loaded checkpoint is not a PicoDet model")

    detector.eval()
    original_export = head.export
    head.export = True
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        device = next(detector.parameters()).device
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        with torch.inference_mode():
            raw_probe = detector(dummy)
        if not isinstance(raw_probe, torch.Tensor):
            raise ConfigurationError(
                "PicoDet export head did not produce its decoded tensor"
            )
        if raw_probe.ndim != 3 or raw_probe.shape[0] != 1 or raw_probe.shape[2] != 4 + num_classes:
            raise ConfigurationError(
                "unexpected PicoDet export tensor shape: "
                f"{tuple(raw_probe.shape)}; expected [1, anchors, {4 + num_classes}]"
            )

        strides = tuple(int(stride) for stride in head.strides)
        level_counts = tuple((image_size // stride) ** 2 for stride in strides)
        if sum(level_counts) != int(raw_probe.shape[1]):
            raise ConfigurationError(
                "PicoDet export level counts do not match the model output: "
                f"levels={level_counts}; output={tuple(raw_probe.shape)}"
            )

        export_model = _build_embedded_nms_detector(
            torch,
            detector,
            image_size=image_size,
            num_classes=num_classes,
            level_counts=level_counts,
            confidence=confidence,
            iou=iou,
            max_det=max_det,
        )
        _export_onnx(
            torch,
            export_model,
            dummy,
            output,
            opset=opset,
        )
    finally:
        head.export = original_export

    graph = onnx.load(str(output))
    if len(graph.graph.output) != 2:
        raise ConfigurationError(
            "PicoDet export must produce an embedded output and one raw auxiliary output"
        )
    # The raw auxiliary tensor is useful to LibreYOLO's own native backend,
    # but is not part of this workflow's public ONNX contract.
    del graph.graph.output[1:]
    onnx.checker.check_model(graph)
    onnx.save(graph, str(output))
    if simplify:
        _simplify_onnx(output, onnx)
    return output


def _build_embedded_nms_detector(
    torch: Any,
    detector: Any,
    *,
    image_size: int,
    num_classes: int,
    level_counts: tuple[int, ...],
    confidence: float,
    iou: float,
    max_det: int,
) -> Any:
    class PicoDetEmbeddedNMSDetector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.detector = detector
            self.image_size = int(image_size)
            self.num_classes = int(num_classes)
            self.level_counts = tuple(int(count) for count in level_counts)
            self.confidence = float(confidence)
            self.iou = float(iou)
            self.max_det = int(max_det)
            self.nms_pre = 1000

        def forward(self, x: Any) -> tuple[Any, Any]:
            from torchvision.ops import nms

            raw = self.detector(x).float()
            boxes_parts: list[Any] = []
            scores_parts: list[Any] = []
            classes_parts: list[Any] = []
            start = 0
            for count in self.level_counts:
                level = raw[0, start : start + count]
                start += count
                boxes_raw = level[:, :4]
                boxes_all = torch.stack(
                    (
                        boxes_raw[:, 0].clamp(min=0.0, max=float(self.image_size)),
                        boxes_raw[:, 1].clamp(min=0.0, max=float(self.image_size)),
                        boxes_raw[:, 2].clamp(min=0.0, max=float(self.image_size)),
                        boxes_raw[:, 3].clamp(min=0.0, max=float(self.image_size)),
                    ),
                    dim=1,
                )
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
                selected = top_scores > self.confidence
                top_scores = top_scores[selected]
                top_flat_idx = top_flat_idx[selected]
                anchor_idx = top_flat_idx // self.num_classes
                class_idx = top_flat_idx - anchor_idx * self.num_classes
                candidate_boxes = safe_boxes[anchor_idx]
                candidate_scores = top_scores
                candidate_classes = class_idx.to(boxes_all.dtype)
                valid_boxes = (candidate_boxes[:, 2] > candidate_boxes[:, 0]) & (
                    candidate_boxes[:, 3] > candidate_boxes[:, 1]
                )
                boxes_parts.append(candidate_boxes[valid_boxes])
                scores_parts.append(candidate_scores[valid_boxes])
                classes_parts.append(candidate_classes[valid_boxes])

            candidate_boxes = torch.cat(boxes_parts, dim=0)
            candidate_scores = torch.cat(scores_parts, dim=0)
            candidate_classes = torch.cat(classes_parts, dim=0)
            all_boxes = raw[0, :, :4]
            safe_all_boxes = torch.where(
                torch.isfinite(all_boxes), all_boxes, torch.zeros_like(all_boxes)
            )
            lower = safe_all_boxes.min()
            class_step = (safe_all_boxes.max() - lower).clamp(min=1.0) + 1.0
            nms_boxes = (candidate_boxes - lower) + candidate_classes[:, None] * class_step
            keep = nms(nms_boxes, candidate_scores, self.iou)
            rows = torch.cat(
                (
                    candidate_boxes[keep],
                    candidate_scores[keep, None],
                    candidate_classes[keep, None],
                ),
                dim=1,
            )
            padded = torch.cat((rows, rows.new_zeros(self.max_det, 6)), dim=0)
            top = torch.topk(padded[:, 4], self.max_det).indices
            return padded[top].reshape(1, self.max_det, 6), raw

    return PicoDetEmbeddedNMSDetector().eval()


def _export_onnx(torch: Any, export_model: Any, dummy: Any, output: Path, *, opset: int) -> None:
    with warnings.catch_warnings():
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
        try:
            torch.onnx.export(
                export_model,
                (dummy,),
                str(output),
                input_names=["images"],
                output_names=["output", "raw"],
                opset_version=opset,
                dynamo=True,
                optimize=True,
                verify=False,
                external_data=False,
                verbose=False,
            )
        except Exception as modern_error:
            warnings.warn(
                "PicoDet modern ONNX export failed; retrying the legacy exporter: "
                f"{type(modern_error).__name__}: {modern_error}",
                RuntimeWarning,
                stacklevel=2,
            )
            torch.onnx.export(
                export_model,
                (dummy,),
                str(output),
                input_names=["images"],
                output_names=["output", "raw"],
                opset_version=opset,
                do_constant_folding=True,
                dynamo=False,
            )


def _simplify_onnx(path: Path, onnx: Any) -> None:
    onnxslim = optional_import("onnxslim")
    original = onnx.load(str(path))
    metadata = {item.key: item.value for item in original.metadata_props}
    simplified = onnxslim.slim(original)
    if not isinstance(simplified, onnx.ModelProto):
        raise ConfigurationError(
            f"onnxslim returned {type(simplified).__name__}, expected ModelProto"
        )
    existing = {item.key: item.value for item in simplified.metadata_props}
    existing.update(metadata)
    del simplified.metadata_props[:]
    for key, value in existing.items():
        item = simplified.metadata_props.add()
        item.key = key
        item.value = value
    onnx.checker.check_model(simplified)
    temporary = path.with_name(path.name + ".slim.tmp")
    try:
        onnx.save(simplified, str(temporary))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
