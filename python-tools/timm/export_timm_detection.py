"""Export the v2 NMS-free query detector to the local ONNX contract.

The float model emits a fixed-size detection set:

* ``boxes``: normalized XYXY float boxes, ``[batch, queries, 4]``
* ``scores``: best foreground softmax score, ``[batch, queries]``
* ``class_ids``: zero-based int64 class IDs, ``[batch, queries]``

Each query is trained by one-to-one assignment, so the model has no anchor
decode or runtime NMS stage.  The exporter records ``nms_required=false`` in
ONNX metadata.  ``--embedded-preprocessing`` additionally creates the uint8
BW8/C24 models consumed by ``OnnxVision.ObjectDetector``.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, compose, helper, numpy_helper
from PIL import Image
from torch import nn

from export_timm_classification import add_preprocess_initializers, resize_node
from train_timm_detection import (
    DEFAULT_BBOX_LOSS_WEIGHT,
    DEFAULT_DECODER_LAYERS,
    DEFAULT_FPN_CHANNELS,
    DEFAULT_GIOU_LOSS_WEIGHT,
    DEFAULT_NO_OBJECT_WEIGHT,
    DEFAULT_NUM_QUERIES,
    DEFAULT_ATTENTION_HEADS,
    CocoDetectionDataset,
    build_detector,
)


DEFAULT_BOXES_ATOL = 2e-3
DEFAULT_SCORES_ATOL = 1e-2


class NmsFreeOutputModel(nn.Module):
    """Expose only the fixed detection set for ONNX deployment."""

    def __init__(self, detector: nn.Module):
        super().__init__()
        self.detector = detector

    def forward(self, normalized_images: torch.Tensor):
        return self.detector.export_outputs(normalized_images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="NMS-free checkpoint (.pt) produced by the v2 trainer")
    parser.add_argument("--output", type=Path, help="Float ONNX output path; defaults beside the checkpoint")
    parser.add_argument("--model-name", help="Override the backbone name stored in the checkpoint")
    parser.add_argument("--imgsz", type=int, help="Override the fixed square input size")
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
    parser.add_argument(
        "--data",
        type=Path,
        help="COCO dataset root; validate its test split against the native and exported models",
    )
    parser.add_argument("--max-test-images", type=int, help="Limit dataset parity validation to this many test images")
    parser.add_argument("--boxes-atol", type=float, default=DEFAULT_BOXES_ATOL)
    parser.add_argument("--scores-atol", type=float, default=DEFAULT_SCORES_ATOL)
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


def load_training_checkpoint(
    path: Path,
    model_name_override: str | None,
    imgsz_override: int | None,
    device: torch.device,
):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected a checkpoint produced by train_timm_detection_v2.py")
    if checkpoint.get("task") != "object_detection":
        raise ValueError("Checkpoint task is not object_detection; do not pass a classification checkpoint")
    if checkpoint.get("architecture") != "nms_free_query":
        raise ValueError(
            "This exporter accepts only architecture=nms_free_query checkpoints; "
            "use export_timm_detection.py for the RetinaNet baseline"
        )
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
        fpn_channels=int(saved_config.get("fpn_channels", DEFAULT_FPN_CHANNELS)),
        num_queries=int(saved_config.get("num_queries", DEFAULT_NUM_QUERIES)),
        decoder_layers=int(saved_config.get("decoder_layers", DEFAULT_DECODER_LAYERS)),
        attention_heads=int(saved_config.get("attention_heads", DEFAULT_ATTENTION_HEADS)),
        pretrained=False,
        image_mean=[float(value) for value in data_config["mean"]],
        image_std=[float(value) for value in data_config["std"]],
        bbox_loss_weight=float(saved_config.get("bbox_loss_weight", DEFAULT_BBOX_LOSS_WEIGHT)),
        giou_loss_weight=float(saved_config.get("giou_loss_weight", DEFAULT_GIOU_LOSS_WEIGHT)),
        no_object_weight=float(saved_config.get("no_object_weight", DEFAULT_NO_OBJECT_WEIGHT)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, list(classes), data_config, saved_config, checkpoint


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
    data_config = dict(data_config)
    data_config["input_size"] = [3, imgsz, imgsz]
    data_config["resize_mode"] = "stretch_to_input_size"
    export_model = NmsFreeOutputModel(detector).to(device).eval()
    input_tensor = torch.zeros(1, 3, imgsz, imgsz, device=device)
    if args.half:
        export_model = export_model.half()
        input_tensor = input_tensor.half()
    output = (args.output or checkpoint_path.with_suffix(".onnx")).expanduser().resolve()
    dynamic_shapes = None
    if args.dynamic_batch:
        dynamic_shapes = {"normalized_images": {0: torch.export.Dim.DYNAMIC}}
    print(f"Exporting {checkpoint_path.name} to {output} on {device}; imgsz={imgsz}; nms_required=false")
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
        "nms_required": "false",
        "postprocess": "threshold_and_class_argmax_only",
        "head": "nms_free_query_set_prediction",
        "names": json.dumps({index: name for index, name in enumerate(classes)}, separators=(",", ":")),
        "model_name": str(model_config.get("model_name", checkpoint.get("model_name", ""))),
        "data_config": json.dumps(data_config, default=str, separators=(",", ":")),
        "model_config": json.dumps(model_config, default=str, separators=(",", ":")),
        "candidate_count": str(int(model_config.get("num_queries", DEFAULT_NUM_QUERIES))),
    })
    onnx.checker.check_model(model_proto)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_proto, output)
    print(f"created_onnx={output}")
    return output, imgsz, data_config, classes, model_config, export_model, input_tensor


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
        "nms_required": "false",
        "postprocess": "threshold_and_class_argmax_only",
        "head": "nms_free_query_set_prediction",
        "names": json.dumps({index: name for index, name in enumerate(classes)}, separators=(",", ":")),
        "data_config": json.dumps(data_config, default=str, separators=(",", ":")),
        "model_config": json.dumps(model_config, default=str, separators=(",", ":")),
        "candidate_count": str(int(model_config.get("num_queries", DEFAULT_NUM_QUERIES))),
    }


def wrap_bw8(model_path: Path, output_path: Path, imgsz: int, config: dict, classes: list[str], model_config: dict):
    core, core_input, core_dtype = prepare_core(model_path)
    core.graph.input.append(helper.make_tensor_value_info("images_bw8_uint8_nchw", TensorProto.UINT8, [1, 1, "height", "width"]))
    sizes = add_preprocess_initializers(core, config["mean"], config["std"], 1, imgsz)
    core.graph.initializer.append(numpy_helper.from_array(np.asarray([1, 3, imgsz, imgsz], dtype=np.int64), "preprocess/rgb_shape"))
    nodes = [
        resize_node("images_bw8_uint8_nchw", "preprocess/resized_uint8", sizes, normalized_interpolation(config)),
        helper.make_node("Cast", ["preprocess/resized_uint8"], ["preprocess/resized_float"], to=TensorProto.FLOAT, name="preprocess/cast"),
        helper.make_node("Div", ["preprocess/resized_float", "preprocess/pixel_scale"], ["preprocess/scaled"], name="preprocess/scale"),
        helper.make_node("Expand", ["preprocess/scaled", "preprocess/rgb_shape"], ["preprocess/rgb"], name="preprocess/gray_to_rgb"),
        helper.make_node("Sub", ["preprocess/rgb", "preprocess/mean"], ["preprocess/centered"], name="preprocess/mean"),
        helper.make_node("Div", ["preprocess/centered", "preprocess/std"], ["preprocess/normalized"], name="preprocess/std"),
    ]
    nodes.extend(core_input_cast(core_input, core_dtype, "preprocess/normalized"))
    finish_detection_wrapper(core, nodes, output_path, "NMS-free query detector with BW8 preprocessing.", wrapper_metadata(classes, config, model_config))


def wrap_c24(model_path: Path, output_path: Path, imgsz: int, config: dict, classes: list[str], model_config: dict):
    core, core_input, core_dtype = prepare_core(model_path)
    core.graph.input.append(helper.make_tensor_value_info("images_c24_uint8_nhwc_bgr", TensorProto.UINT8, [1, "height", "width", 3]))
    sizes = add_preprocess_initializers(core, config["mean"], config["std"], 3, imgsz)
    core.graph.initializer.append(numpy_helper.from_array(np.asarray([2, 1, 0], dtype=np.int64), "preprocess/bgr_to_rgb_indices"))
    nodes = [
        helper.make_node("Transpose", ["images_c24_uint8_nhwc_bgr"], ["preprocess/bgr_nchw"], perm=[0, 3, 1, 2], name="preprocess/transpose"),
        helper.make_node("Gather", ["preprocess/bgr_nchw", "preprocess/bgr_to_rgb_indices"], ["preprocess/rgb_nchw"], axis=1, name="preprocess/bgr_to_rgb"),
        resize_node("preprocess/rgb_nchw", "preprocess/resized_uint8", sizes, normalized_interpolation(config)),
        helper.make_node("Cast", ["preprocess/resized_uint8"], ["preprocess/resized_float"], to=TensorProto.FLOAT, name="preprocess/cast"),
        helper.make_node("Div", ["preprocess/resized_float", "preprocess/pixel_scale"], ["preprocess/scaled"], name="preprocess/scale"),
        helper.make_node("Sub", ["preprocess/scaled", "preprocess/mean"], ["preprocess/centered"], name="preprocess/mean"),
        helper.make_node("Div", ["preprocess/centered", "preprocess/std"], ["preprocess/normalized"], name="preprocess/std"),
    ]
    nodes.extend(core_input_cast(core_input, core_dtype, "preprocess/normalized"))
    finish_detection_wrapper(core, nodes, output_path, "NMS-free query detector with C24 preprocessing.", wrapper_metadata(classes, config, model_config))


def decode_detector_outputs(
    class_logits: torch.Tensor,
    boxes: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(class_logits, dim=-1)[..., :num_classes]
    scores, class_ids = probabilities.max(dim=-1)
    return boxes, scores, class_ids.to(torch.int64)


def as_numpy_outputs(outputs) -> list[np.ndarray]:
    return [value.detach().cpu().numpy() for value in outputs]


def compare_detection_outputs(
    reference: list[np.ndarray],
    actual: list[np.ndarray],
    label: str,
    boxes_atol: float,
    scores_atol: float,
) -> dict[str, float | int]:
    if len(actual) != 3:
        raise ValueError(f"{label}: expected three outputs, received {len(actual)}")
    names = ("boxes", "scores", "class_ids")
    if len(reference) != len(names):
        raise ValueError(f"{label}: expected three reference outputs, received {len(reference)}")
    errors: dict[str, float | int] = {}
    for name, expected, received in zip(names, reference, actual):
        if expected.shape != received.shape:
            raise ValueError(f"{label}: {name} shape mismatch: {expected.shape} vs {received.shape}")
        if name == "class_ids":
            mismatch_count = int(np.count_nonzero(expected != received))
            errors["class_id_mismatches"] = mismatch_count
            if mismatch_count:
                raise ValueError(f"{label}: class_ids differ at {mismatch_count} positions")
            continue
        error = float(np.max(np.abs(expected.astype(np.float64) - received.astype(np.float64))))
        errors[f"{name}_maximum_absolute_error"] = error
        tolerance = boxes_atol if name == "boxes" else scores_atol
        if error > tolerance:
            raise ValueError(f"{label}: {name} differs too much: {error} > {tolerance}")
    return errors


def validate_onnx(
    model_path: Path,
    export_model: nn.Module,
    input_tensor: torch.Tensor,
    boxes_atol: float,
    scores_atol: float,
) -> None:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    reference = as_numpy_outputs(export_model(input_tensor))
    actual = session.run(None, {input_name: input_tensor.detach().cpu().numpy()})
    errors = compare_detection_outputs(reference, actual, "float_onnx", boxes_atol, scores_atol)
    print(f"boxes_maximum_absolute_error={errors['boxes_maximum_absolute_error']:.9g}")
    print(f"scores_maximum_absolute_error={errors['scores_maximum_absolute_error']:.9g}")
    print(f"onnxruntime_validation=passed providers={session.get_providers()}; nms_required=false")


def load_validation_images(dataset: CocoDetectionDataset, imgsz: int):
    for record in dataset.images:
        with Image.open(record["path"]) as source:
            rgb = source.convert("RGB").copy()
            gray = source.convert("L").copy()
        resized_rgb = rgb.resize((imgsz, imgsz), Image.Resampling.BILINEAR)
        rgb_array = np.asarray(resized_rgb, dtype=np.uint8).copy()
        raw_rgb = torch.from_numpy(rgb_array).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        c24 = np.asarray(rgb, dtype=np.uint8).copy()[:, :, ::-1][None, ...]
        bw8 = np.asarray(gray, dtype=np.uint8).copy()[None, None, ...]
        yield record, rgb, gray, raw_rgb, c24, bw8


def run_session(session: ort.InferenceSession, input_array: np.ndarray) -> dict[str, np.ndarray]:
    input_name = session.get_inputs()[0].name
    values = session.run(None, {input_name: input_array})
    return {metadata.name: value for metadata, value in zip(session.get_outputs(), values)}


def add_preprocess_validation_output(model_path: Path, output_path: Path, imgsz: int) -> None:
    model = onnx.load(model_path)
    normalized_name = "preprocess/normalized"
    node_outputs = {name for node in model.graph.node for name in node.output}
    if normalized_name not in node_outputs:
        raise ValueError(f"{model_path} does not expose the expected {normalized_name!r} tensor")
    if normalized_name not in {output.name for output in model.graph.output}:
        model.graph.output.append(
            helper.make_tensor_value_info(normalized_name, TensorProto.FLOAT, [1, 3, imgsz, imgsz])
        )
    onnx.checker.check_model(model)
    onnx.save(model, output_path)


def update_validation_summary(
    summaries: dict[str, dict[str, float | int]],
    label: str,
    errors: dict[str, float | int],
) -> None:
    summary = summaries.setdefault(
        label,
        {
            "images": 0,
            "boxes_maximum_absolute_error": 0.0,
            "scores_maximum_absolute_error": 0.0,
            "class_id_mismatches": 0,
        },
    )
    summary["images"] += 1
    summary["boxes_maximum_absolute_error"] = max(
        float(summary["boxes_maximum_absolute_error"]),
        float(errors["boxes_maximum_absolute_error"]),
    )
    summary["scores_maximum_absolute_error"] = max(
        float(summary["scores_maximum_absolute_error"]),
        float(errors["scores_maximum_absolute_error"]),
    )
    summary["class_id_mismatches"] += int(errors["class_id_mismatches"])


def validate_test_dataset(
    data_path: Path,
    imgsz: int,
    model_config: dict,
    classes: list[str],
    export_model: nn.Module,
    float_path: Path,
    bw8_path: Path | None,
    c24_path: Path | None,
    max_test_images: int | None,
    device: torch.device,
    boxes_atol: float,
    scores_atol: float,
) -> None:
    if not bool(model_config.get("stretch_to_input_size", True)):
        raise ValueError("Dataset parity validation requires stretch_to_input_size=true")
    dataset = CocoDetectionDataset(data_path, "test", imgsz, False, True, max_test_images)
    if dataset.class_names != classes:
        raise ValueError(f"Test categories differ from checkpoint classes: {dataset.class_names} != {classes}")
    if not len(dataset):
        raise ValueError("Test dataset is empty")

    native_model = export_model.float().eval()
    native_detector = native_model.detector
    float_session = ort.InferenceSession(str(float_path), providers=["CPUExecutionProvider"])
    wrapper_sessions: list[tuple[str, ort.InferenceSession]] = []
    with tempfile.TemporaryDirectory(prefix="nms-free-query-validation-") as temporary:
        for name, path in (("bw8", bw8_path), ("c24", c24_path)):
            if path is None:
                continue
            validation_path = Path(temporary) / f"{name}-with-preprocess-output.onnx"
            add_preprocess_validation_output(path, validation_path, imgsz)
            wrapper_sessions.append((name, ort.InferenceSession(str(validation_path), providers=["CPUExecutionProvider"])))

        mean = torch.tensor([float(value) for value in native_detector.image_mean.flatten()], dtype=torch.float32)
        std = torch.tensor([float(value) for value in native_detector.image_std.flatten()], dtype=torch.float32)
        summaries: dict[str, dict[str, float | int]] = {}
        print(f"test_dataset_validation=starting images={len(dataset)}")
        for index, (record, rgb, gray, raw_rgb, c24, bw8) in enumerate(load_validation_images(dataset, imgsz), start=1):
            image_on_device = raw_rgb.to(device)
            with torch.inference_mode():
                native_logits, native_boxes = native_detector(image_on_device)
                native_outputs = as_numpy_outputs(
                    decode_detector_outputs(native_logits, native_boxes, len(classes))
                )
            normalized = ((raw_rgb - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)).numpy()
            float_outputs = run_session(float_session, normalized)
            errors = compare_detection_outputs(
                native_outputs,
                [float_outputs[name] for name in ("boxes", "scores", "class_ids")],
                f"test[{index}] native_vs_float",
                boxes_atol,
                scores_atol,
            )
            update_validation_summary(summaries, "native_vs_float", errors)

            for wrapper_name, wrapper_session in wrapper_sessions:
                raw_input = bw8 if wrapper_name == "bw8" else c24
                wrapper_outputs = run_session(wrapper_session, raw_input)
                normalized_from_wrapper = wrapper_outputs["preprocess/normalized"]
                with torch.inference_mode():
                    native_wrapper_outputs = as_numpy_outputs(
                        native_model(torch.from_numpy(normalized_from_wrapper).to(device))
                    )
                errors = compare_detection_outputs(
                    native_wrapper_outputs,
                    [wrapper_outputs[name] for name in ("boxes", "scores", "class_ids")],
                    f"test[{index}] native_vs_{wrapper_name}",
                    boxes_atol,
                    scores_atol,
                )
                update_validation_summary(summaries, f"native_vs_{wrapper_name}", errors)
            if index == 1 or index == len(dataset):
                print(f"test_dataset_validation=progress {index}/{len(dataset)} path={record['path'].name}")
    print(f"test_dataset_validation=passed images={len(dataset)}")
    for label, summary in summaries.items():
        print(
            f"{label}_images={summary['images']} "
            f"boxes_maximum_absolute_error={float(summary['boxes_maximum_absolute_error']):.9g} "
            f"scores_maximum_absolute_error={float(summary['scores_maximum_absolute_error']):.9g} "
            f"class_id_mismatches={summary['class_id_mismatches']}"
        )


def main() -> None:
    args = parse_args()
    if args.imgsz is not None and (args.imgsz < 32 or args.imgsz % 32):
        raise ValueError("--imgsz must be at least 32 and divisible by 32")
    if args.max_test_images is not None and args.max_test_images < 1:
        raise ValueError("--max-test-images must be positive")
    if args.boxes_atol <= 0 or args.scores_atol <= 0:
        raise ValueError("--boxes-atol and --scores-atol must be positive")
    device = resolve_device(args.device)
    print(f"device={device}")
    output, imgsz, config, classes, model_config, export_model, input_tensor = export_onnx(args, device)
    if not args.skip_validation:
        validate_onnx(
            output,
            export_model.float().eval(),
            input_tensor.float(),
            args.boxes_atol,
            args.scores_atol,
        )
    bw8_path = None
    c24_path = None
    if args.embedded_preprocessing:
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
    if args.data is not None:
        validate_test_dataset(
            args.data.expanduser().resolve(),
            imgsz,
            model_config,
            classes,
            export_model,
            output,
            bw8_path,
            c24_path,
            args.max_test_images,
            device,
            args.boxes_atol,
            args.scores_atol,
        )


if __name__ == "__main__":
    main()
