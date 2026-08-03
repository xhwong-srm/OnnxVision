"""Export a MobileNetV4-backed RetinaNet checkpoint to the local ONNX contract.

The float model emits decoded candidate detections:

* ``boxes``: normalized XYXY float boxes, ``[batch, candidates, 4]``
* ``scores``: sigmoid class scores, ``[batch, candidates]``
* ``class_ids``: zero-based int64 class IDs, ``[batch, candidates]``

NMS is intentionally left to the repository's C# runtime and is declared by
``nms_required=true`` metadata. ``--embedded-preprocessing`` additionally
creates the uint8 BW8/C24 models consumed by ``OnnxVision.ObjectDetector``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, compose, helper, numpy_helper
from PIL import Image
from torch import nn
from torchvision.models.detection.image_list import ImageList

from export_timm_classification import add_preprocess_initializers, resize_node
from train_timm_detection import (
    DEFAULT_ANCHOR_RATIOS,
    DEFAULT_ANCHOR_SIZES,
    build_detector,
)


class RetinaNetCandidateModel(nn.Module):
    """Export-safe fixed-resolution RetinaNet decoding without Python NMS."""

    def __init__(self, detector: nn.Module, anchors: torch.Tensor, imgsz: int, num_classes: int):
        super().__init__()
        self.backbone = detector.backbone
        self.head = detector.head
        self.register_buffer("anchors", anchors)
        self.imgsz = int(imgsz)
        self.num_classes = int(num_classes)

    def forward(self, images: torch.Tensor):
        feature_dict = self.backbone(images)
        outputs = self.head(list(feature_dict.values()))
        deltas = outputs["bbox_regression"]
        logits = outputs["cls_logits"]
        anchors = self.anchors.to(dtype=deltas.dtype)

        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        center_x = anchors[:, 0] + 0.5 * widths
        center_y = anchors[:, 1] + 0.5 * heights
        dx = deltas[..., 0]
        dy = deltas[..., 1]
        dw = torch.clamp(deltas[..., 2], max=4.135166556742356)
        dh = torch.clamp(deltas[..., 3], max=4.135166556742356)
        predicted_center_x = dx * widths + center_x
        predicted_center_y = dy * heights + center_y
        predicted_width = torch.exp(dw) * widths
        predicted_height = torch.exp(dh) * heights
        decoded = torch.stack(
            (
                predicted_center_x - 0.5 * predicted_width,
                predicted_center_y - 0.5 * predicted_height,
                predicted_center_x + 0.5 * predicted_width,
                predicted_center_y + 0.5 * predicted_height,
            ),
            dim=-1,
        )
        boxes = decoded / float(self.imgsz)
        boxes = boxes.clamp(0.0, 1.0)

        batch_size, anchor_count, _ = boxes.shape
        boxes = boxes.unsqueeze(2).expand(-1, -1, self.num_classes, -1).reshape(batch_size, -1, 4)
        scores = torch.sigmoid(logits).reshape(batch_size, -1)
        class_ids = torch.arange(self.num_classes, device=images.device, dtype=torch.int64)
        class_ids = class_ids.reshape(1, 1, -1).expand(batch_size, anchor_count, -1).reshape(batch_size, -1)
        return boxes, scores, class_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Detection checkpoint (.pt) produced by the trainer")
    parser.add_argument("--output", type=Path, help="Float ONNX output path; defaults beside the checkpoint")
    parser.add_argument("--model-name", help="Override the model name stored in the checkpoint")
    parser.add_argument("--imgsz", type=int, help="Override the fixed square training input size")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--device", default="auto", help='"auto", "cpu", or "cuda[:N]"')
    parser.add_argument("--dynamic", "--dynamic-batch", dest="dynamic_batch", action="store_true",
                        help="Allow a dynamic batch dimension; spatial dimensions remain fixed")
    parser.add_argument("--half", action="store_true", help="Export the float core as FP16")
    parser.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True,
                        help="Simplify the float ONNX graph with onnxslim when installed")
    parser.add_argument("--embedded-preprocessing", action="store_true",
                        help="Create the raw uint8 BW8/C24 deployment models")
    parser.add_argument("--bw8-output", type=Path)
    parser.add_argument("--c24-output", type=Path)
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip the ONNX Runtime numerical smoke test")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({requested}) but CUDA is unavailable")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested!r}")
    return device


def load_training_checkpoint(path: Path, model_name_override: str | None, imgsz_override: int | None, device: torch.device):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected a checkpoint produced by train_timm_detection.py")
    if checkpoint.get("task") != "object_detection":
        raise ValueError("Checkpoint task is not object_detection; do not pass a classification checkpoint")
    classes = checkpoint.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("Checkpoint must contain a non-empty classes list")
    saved_config = checkpoint.get("model_config")
    data_config = checkpoint.get("data_config")
    if not isinstance(saved_config, dict) or not isinstance(data_config, dict):
        raise ValueError("Checkpoint is missing model_config or data_config")
    model_name = model_name_override or str(checkpoint.get("model_name") or saved_config.get("model_name"))
    imgsz = int(imgsz_override or saved_config.get("imgsz", 384))
    saved_config = dict(saved_config)
    saved_config["model_name"] = model_name
    saved_config["imgsz"] = imgsz
    model, _, _ = build_detector(
        model_name=model_name,
        num_classes=len(classes),
        imgsz=imgsz,
        fpn_channels=int(saved_config.get("fpn_channels", 256)),
        anchor_sizes=tuple(int(value) for value in saved_config.get("anchor_sizes", DEFAULT_ANCHOR_SIZES)),
        anchor_ratios=tuple(float(value) for value in saved_config.get("anchor_ratios", DEFAULT_ANCHOR_RATIOS)),
        pretrained=False,
        image_mean=[float(value) for value in data_config["mean"]],
        image_std=[float(value) for value in data_config["std"]],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, list(classes), data_config, saved_config, checkpoint


def make_candidate_model(detector: nn.Module, imgsz: int, num_classes: int) -> RetinaNetCandidateModel:
    detector_device = next(detector.parameters()).device
    with torch.inference_mode():
        example = torch.zeros(1, 3, imgsz, imgsz, device=detector_device)
        feature_dict = detector.backbone(example)
        feature_maps = list(feature_dict.values())
        image_list = ImageList(example, [(imgsz, imgsz)])
        anchors = detector.anchor_generator(image_list, feature_maps)[0].detach().cpu()
    expected = sum(feature.shape[-2] * feature.shape[-1] for feature in feature_maps)
    expected *= detector.anchor_generator.num_anchors_per_location()[0]
    if anchors.shape[0] != expected:
        raise ValueError(f"Anchor count mismatch: generated {anchors.shape[0]}, expected {expected}")
    return RetinaNetCandidateModel(detector, anchors, imgsz, num_classes).eval()


def set_metadata(model: onnx.ModelProto, values: dict[str, str]) -> None:
    existing = {item.key: item.value for item in model.metadata_props}
    existing.update(values)
    del model.metadata_props[:]
    for key, value in sorted(existing.items()):
        item = model.metadata_props.add()
        item.key = key
        item.value = value


def export_onnx(args: argparse.Namespace, device: torch.device):
    checkpoint_path = args.model.expanduser().resolve()
    detector, classes, data_config, model_config, checkpoint = load_training_checkpoint(
        checkpoint_path, args.model_name, args.imgsz, device
    )
    imgsz = int(args.imgsz or model_config.get("imgsz", 384))
    export_model = make_candidate_model(detector, imgsz, len(classes)).to(device).eval()
    input_tensor = torch.zeros(1, 3, imgsz, imgsz, device=device)
    if args.half:
        export_model = export_model.half()
        input_tensor = input_tensor.half()
    output = (args.output or checkpoint_path.with_suffix(".onnx")).expanduser().resolve()
    dynamic_shapes = None
    if args.dynamic_batch:
        dynamic_shapes = {"images": {0: torch.export.Dim.DYNAMIC}}
    print(f"Exporting {checkpoint_path.name} to {output} on {device}; imgsz={imgsz}")
    with torch.inference_mode():
        torch.onnx.export(
            export_model,
            (input_tensor,),
            output,
            input_names=["images"],
            output_names=["boxes", "scores", "class_ids"],
            opset_version=args.opset,
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            external_data=False,
            optimize=True,
            verify=True,
            verbose=False,
        )
    if args.simplify:
        try:
            import onnxslim
            onnxslim.slim(str(output), str(output))
        except ImportError:
            print("onnxslim_not_installed=simplification_skipped")
    model_proto = onnx.load(output)
    set_metadata(model_proto, {
        "vision_task": "object_detection",
        "detection_contract": "onnx-vision-detection-v1",
        "nms_required": "true",
        "names": json.dumps({index: name for index, name in enumerate(classes)}, separators=(",", ":")),
        "model_name": str(model_config.get("model_name", checkpoint.get("model_name", ""))),
        "data_config": json.dumps(data_config, default=str, separators=(",", ":")),
        "model_config": json.dumps(model_config, default=str, separators=(",", ":")),
        "candidate_count": str(int(export_model.anchors.shape[0]) * len(classes)),
    })
    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, output)
    print(f"created_onnx={output}")
    return output, imgsz, data_config, classes, export_model, input_tensor


def prepare_core(model_path: Path):
    model = onnx.load(model_path)
    if len(model.graph.input) != 1:
        raise ValueError("Expected exactly one float model input")
    if not any(item.domain == "" and item.version >= 18 for item in model.opset_import):
        raise ValueError("Embedded preprocessing requires an ONNX opset of 18 or newer")
    original_input = model.graph.input[0]
    core_dtype = original_input.type.tensor_type.elem_type
    core = compose.add_prefix(model, "core/")
    core_input = core.graph.input[0].name
    del core.graph.input[:]
    return core, core_input, core_dtype


def finish_detection_wrapper(
    core: onnx.ModelProto,
    preprocessing: list[onnx.NodeProto],
    output_path: Path,
    description: str,
    metadata: dict[str, str],
) -> None:
    core_nodes = list(core.graph.node)
    del core.graph.node[:]
    core.graph.node.extend(preprocessing + core_nodes)
    for output in core.graph.output:
        old_name = output.name
        short_name = old_name.rsplit("/", 1)[-1]
        if short_name not in {"boxes", "scores", "class_ids"}:
            raise ValueError(f"Unexpected detection output name after prefixing: {old_name}")
        core.graph.node.append(
            helper.make_node("Identity", [old_name], [short_name], name=f"outputs/{short_name}")
        )
        output.name = short_name
    core.graph.doc_string = description
    set_metadata(core, metadata)
    onnx.checker.check_model(core)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(core, output_path)


def normalized_interpolation(config: dict) -> str:
    value = str(config.get("interpolation", "bilinear")).lower().rsplit(".", 1)[-1]
    value = {"bilinear": "linear", "bicubic": "cubic", "nearest": "nearest"}.get(value, value)
    if value not in {"linear", "cubic", "nearest"}:
        raise ValueError(f"Unsupported interpolation in checkpoint data_config: {value!r}")
    return value


def core_input_cast(core_input: str, core_dtype: int, source: str) -> list[onnx.NodeProto]:
    if core_dtype == TensorProto.FLOAT:
        return [helper.make_node("Identity", [source], [core_input], name="preprocess/to_core")]
    if core_dtype == TensorProto.FLOAT16:
        return [helper.make_node("Cast", [source], [core_input], to=TensorProto.FLOAT16, name="preprocess/to_core_fp16")]
    raise ValueError(f"Unsupported exported core input type: TensorProto {core_dtype}")


def wrapper_metadata(classes: list[str], data_config: dict, model_config: dict) -> dict[str, str]:
    return {
        "vision_task": "object_detection",
        "detection_contract": "onnx-vision-detection-v1",
        "nms_required": "true",
        "names": json.dumps({index: name for index, name in enumerate(classes)}, separators=(",", ":")),
        "data_config": json.dumps(data_config, default=str, separators=(",", ":")),
        "model_config": json.dumps(model_config, default=str, separators=(",", ":")),
    }


def wrap_bw8(model_path: Path, output_path: Path, imgsz: int, config: dict, classes: list[str], model_config: dict):
    core, core_input, core_dtype = prepare_core(model_path)
    core.graph.input.append(helper.make_tensor_value_info("images_bw8_uint8_nchw", TensorProto.UINT8, [1, 1, "height", "width"]))
    sizes = add_preprocess_initializers(core, config["mean"], config["std"], 1, imgsz)
    core.graph.initializer.append(numpy_helper.from_array(np.asarray([1, 3, imgsz, imgsz], dtype=np.int64), "preprocess/rgb_shape"))
    interpolation = normalized_interpolation(config)
    nodes = [
        resize_node("images_bw8_uint8_nchw", "preprocess/resized_uint8", sizes, interpolation),
        helper.make_node("Cast", ["preprocess/resized_uint8"], ["preprocess/resized_float"], to=TensorProto.FLOAT, name="preprocess/cast"),
        helper.make_node("Div", ["preprocess/resized_float", "preprocess/pixel_scale"], ["preprocess/scaled"], name="preprocess/scale"),
        helper.make_node("Expand", ["preprocess/scaled", "preprocess/rgb_shape"], ["preprocess/rgb"], name="preprocess/gray_to_rgb"),
        helper.make_node("Sub", ["preprocess/rgb", "preprocess/mean"], ["preprocess/centered"], name="preprocess/mean"),
        helper.make_node("Div", ["preprocess/centered", "preprocess/std"], ["preprocess/normalized"], name="preprocess/std"),
    ]
    nodes.extend(core_input_cast(core_input, core_dtype, "preprocess/normalized"))
    finish_detection_wrapper(core, nodes, output_path, "RetinaNet detector with BW8 preprocessing.", wrapper_metadata(classes, config, model_config))


def wrap_c24(model_path: Path, output_path: Path, imgsz: int, config: dict, classes: list[str], model_config: dict):
    core, core_input, core_dtype = prepare_core(model_path)
    core.graph.input.append(helper.make_tensor_value_info("images_c24_uint8_nhwc_bgr", TensorProto.UINT8, [1, "height", "width", 3]))
    sizes = add_preprocess_initializers(core, config["mean"], config["std"], 3, imgsz)
    core.graph.initializer.append(numpy_helper.from_array(np.asarray([2, 1, 0], dtype=np.int64), "preprocess/bgr_to_rgb_indices"))
    interpolation = normalized_interpolation(config)
    nodes = [
        helper.make_node("Transpose", ["images_c24_uint8_nhwc_bgr"], ["preprocess/bgr_nchw"], perm=[0, 3, 1, 2], name="preprocess/transpose"),
        helper.make_node("Gather", ["preprocess/bgr_nchw", "preprocess/bgr_to_rgb_indices"], ["preprocess/rgb_nchw"], axis=1, name="preprocess/bgr_to_rgb"),
        resize_node("preprocess/rgb_nchw", "preprocess/resized_uint8", sizes, interpolation),
        helper.make_node("Cast", ["preprocess/resized_uint8"], ["preprocess/resized_float"], to=TensorProto.FLOAT, name="preprocess/cast"),
        helper.make_node("Div", ["preprocess/resized_float", "preprocess/pixel_scale"], ["preprocess/scaled"], name="preprocess/scale"),
        helper.make_node("Sub", ["preprocess/scaled", "preprocess/mean"], ["preprocess/centered"], name="preprocess/mean"),
        helper.make_node("Div", ["preprocess/centered", "preprocess/std"], ["preprocess/normalized"], name="preprocess/std"),
    ]
    nodes.extend(core_input_cast(core_input, core_dtype, "preprocess/normalized"))
    finish_detection_wrapper(core, nodes, output_path, "RetinaNet detector with C24 preprocessing.", wrapper_metadata(classes, config, model_config))


def validate_onnx(model_path: Path, export_model: nn.Module, input_tensor: torch.Tensor) -> None:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    reference = [value.detach().cpu().numpy() for value in export_model(input_tensor)]
    actual = session.run(None, {input_name: input_tensor.detach().cpu().numpy()})
    if len(actual) != 3:
        raise ValueError(f"Expected three ONNX outputs, received {len(actual)}")
    for name, expected, received in zip(("boxes", "scores", "class_ids"), reference, actual):
        if expected.shape != received.shape:
            raise ValueError(f"{name} shape mismatch: PyTorch {expected.shape} vs ONNX {received.shape}")
        if name == "class_ids":
            if not np.array_equal(expected, received):
                raise ValueError("class_ids mismatch between PyTorch and ONNX")
        else:
            error = float(np.max(np.abs(expected.astype(np.float64) - received.astype(np.float64))))
            print(f"{name}_maximum_absolute_error={error:.9g}")
            if error > 2e-3:
                raise ValueError(f"{name} differs too much between PyTorch and ONNX: {error}")
    print(f"onnxruntime_validation=passed providers={session.get_providers()}")


def main() -> None:
    args = parse_args()
    if args.imgsz is not None and (args.imgsz < 32 or args.imgsz % 32):
        raise ValueError("--imgsz must be at least 32 and divisible by 32")
    device = resolve_device(args.device)
    print(f"device={device}")
    output, imgsz, config, classes, export_model, input_tensor = export_onnx(args, device)
    if not args.skip_validation:
        validate_onnx(output, export_model.float().eval(), input_tensor.float())
    if not args.embedded_preprocessing:
        return
    model_config = json.loads(next(item.value for item in onnx.load(output).metadata_props if item.key == "model_config"))
    if not bool(model_config.get("stretch_to_input_size", True)):
        raise ValueError(
            "Embedded preprocessing currently uses square stretching; export a checkpoint trained with "
            "--stretch-to-input-size for BW8/C24 deployment."
        )
    bw8_path = (args.bw8_output or output.with_name(f"{output.stem}-embedded-preprocess-bw8.onnx")).expanduser().resolve()
    c24_path = (args.c24_output or output.with_name(f"{output.stem}-embedded-preprocess-c24.onnx")).expanduser().resolve()
    wrap_bw8(output, bw8_path, imgsz, config, classes, model_config)
    wrap_c24(output, c24_path, imgsz, config, classes, model_config)
    print(f"created_bw8={bw8_path}")
    print(f"created_c24={c24_path}")


if __name__ == "__main__":
    main()
