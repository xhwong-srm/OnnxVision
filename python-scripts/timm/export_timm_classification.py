"""Export a timm classification checkpoint from train_mobilenetv3_classification.py to ONNX.

The normal model input is a float RGB NCHW tensor after the training
script's resize, ToTensor, and Normalize operations.  With
``--embedded-preprocessing`` the script also creates raw-image models for
embedded callers:

* BW8: uint8 grayscale NCHW
* C24: uint8 BGR NHWC

The embedded models perform resize, RGB conversion where required, scaling,
and the model's mean/std normalization inside ONNX.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch
from onnx import TensorProto, compose, helper, numpy_helper
from PIL import Image
from torchvision import transforms


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class ProbabilityModel(torch.nn.Module):
    """Expose classifier probabilities instead of raw logits for deployment."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(images), dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Checkpoint (.pt) produced by the training script")
    parser.add_argument("--model-name", help="Override the model_name stored in the checkpoint")
    parser.add_argument("--imgsz", type=int, help="Override the square training input size")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset (default: 18)")
    parser.add_argument("--dynamic", action="store_true", help="Allow dynamic batch and image dimensions")
    parser.add_argument("--half", action="store_true", help="Export the classifier core as FP16")
    parser.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True,
                        help="Simplify with onnxslim when installed (default: enabled)")
    parser.add_argument("--embedded-preprocessing", action="store_true",
                        help="Also create raw uint8 BW8 and C24 ONNX models")
    parser.add_argument("--dataset", type=Path, help="Class-folder dataset used to validate wrappers")
    parser.add_argument("--bw8-output", type=Path)
    parser.add_argument("--c24-output", type=Path)
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def load_training_checkpoint(path: Path, model_name_override: str | None):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected a checkpoint produced by the timm classification training script")
    model_name = model_name_override or checkpoint.get("model_name")
    classes = checkpoint.get("classes")
    if not model_name or not classes:
        raise ValueError("Checkpoint must contain model_name and non-empty classes")
    config = checkpoint.get("data_config")
    if not config:
        raise ValueError("Checkpoint is missing data_config; retrain or add the training metadata")
    config = dict(config)
    config["model_name"] = model_name
    interpolation = str(config.get("interpolation", "bilinear")).lower()
    config["onnx_mode"] = {"bilinear": "linear", "bicubic": "cubic", "nearest": "nearest"}.get(interpolation)
    if config["onnx_mode"] is None:
        raise ValueError(f"Unsupported data_config interpolation: {interpolation!r}")
    model = timm.create_model(model_name, pretrained=False, num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, list(classes), config


def model_input_size(config: dict, override: int | None) -> int:
    input_size = config.get("input_size")
    if not input_size or len(input_size) != 3 or input_size[0] != 3:
        raise ValueError(f"Unsupported data_config input_size: {input_size!r}")
    height, width = input_size[-2:]
    if height != width and override is None:
        raise ValueError("Only square export inputs are supported; provide --imgsz for a square size")
    return override or height


def export_onnx(args: argparse.Namespace):
    checkpoint_path = args.model.expanduser().resolve()
    model, classes, config = load_training_checkpoint(checkpoint_path, args.model_name)
    imgsz = model_input_size(config, args.imgsz)
    output = checkpoint_path.with_suffix(".onnx")
    input_tensor = torch.zeros(1, 3, imgsz, imgsz)
    if args.half:
        model = model.half()
        input_tensor = input_tensor.half()
    export_model = ProbabilityModel(model)
    dynamic_axes = {"images": {0: "batch"}, "probabilities": {0: "batch"}} if args.dynamic else None
    if args.dynamic:
        dynamic_axes["images"].update({2: "height", 3: "width"})
    print(f"Exporting {checkpoint_path.name} ({model.__class__.__name__}) to ONNX...")
    torch.onnx.export(export_model, input_tensor, output, input_names=["images"], output_names=["probabilities"],
                      opset_version=args.opset, dynamic_axes=dynamic_axes,
                      do_constant_folding=True, dynamo=False)
    if args.simplify:
        try:
            import onnxslim
            onnxslim.slim(str(output), str(output))
        except ImportError:
            print("onnxslim_not_installed=simplification_skipped")
    model_proto = onnx.load(output)
    metadata = {"names": json.dumps({i: name for i, name in enumerate(classes)}),
                "model_name": str(config.get("model_name", "")),
                "data_config": json.dumps(config, default=str)}
    for key, value in metadata.items():
        entry = model_proto.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(model_proto, output)
    print(f"created_onnx={output}")
    return output, imgsz, config


def prepare_core(model_path: Path) -> tuple[onnx.ModelProto, str]:
    model = onnx.load(model_path)
    if len(model.graph.input) != 1:
        raise ValueError("Expected exactly one model input")
    if not any(item.domain == "" and item.version >= 18 for item in model.opset_import):
        raise ValueError("Embedded preprocessing requires an ONNX opset of 18 or newer")
    core = compose.add_prefix(model, "core/")
    core_input = core.graph.input[0].name
    del core.graph.input[:]
    return core, core_input


def add_preprocess_initializers(core, mean, std, channels, imgsz):
    sizes = "preprocess/target_sizes"
    core.graph.initializer.extend([
        numpy_helper.from_array(np.asarray([1, channels, imgsz, imgsz], dtype=np.int64), sizes),
        numpy_helper.from_array(np.asarray([[[[255.0]]]], dtype=np.float32), "preprocess/pixel_scale"),
        numpy_helper.from_array(np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1), "preprocess/mean"),
        numpy_helper.from_array(np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1), "preprocess/std"),
    ])
    return sizes


def finish_wrapper(core, preprocessing, output_path: Path, description: str):
    nodes = list(core.graph.node)
    del core.graph.node[:]
    core.graph.node.extend(preprocessing + nodes)
    core.graph.doc_string = description
    onnx.checker.check_model(core)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(core, output_path)


def resize_node(source, result, sizes, mode):
    attributes = {"mode": mode, "coordinate_transformation_mode": "half_pixel", "antialias": 1}
    return helper.make_node("Resize", [source, "", "", sizes], [result], name="preprocess/resize", **attributes)


def wrap_bw8(model_path, output_path, imgsz, config):
    core, core_input = prepare_core(model_path)
    core.graph.input.append(helper.make_tensor_value_info("images_bw8_uint8_nchw", TensorProto.UINT8, [1, 1, "height", "width"]))
    sizes = add_preprocess_initializers(core, config["mean"], config["std"], 1, imgsz)
    core.graph.initializer.append(numpy_helper.from_array(np.asarray([1, 3, imgsz, imgsz], dtype=np.int64), "preprocess/rgb_shape"))
    nodes = [resize_node("images_bw8_uint8_nchw", "preprocess/resized_uint8", sizes, config["onnx_mode"]),
             helper.make_node("Cast", ["preprocess/resized_uint8"], ["preprocess/resized_float"], to=TensorProto.FLOAT, name="preprocess/cast"),
             helper.make_node("Div", ["preprocess/resized_float", "preprocess/pixel_scale"], ["preprocess/scaled"], name="preprocess/scale"),
             helper.make_node("Expand", ["preprocess/scaled", "preprocess/rgb_shape"], ["preprocess/rgb"], name="preprocess/gray_to_rgb"),
             helper.make_node("Sub", ["preprocess/rgb", "preprocess/mean"], ["preprocess/centered"], name="preprocess/mean"),
             helper.make_node("Div", ["preprocess/centered", "preprocess/std"], [core_input], name="preprocess/std")]
    finish_wrapper(core, nodes, output_path, "timm image classifier with BW8 preprocessing.")


def wrap_c24(model_path, output_path, imgsz, config):
    core, core_input = prepare_core(model_path)
    core.graph.input.append(helper.make_tensor_value_info("images_c24_uint8_nhwc_bgr", TensorProto.UINT8, [1, "height", "width", 3]))
    sizes = add_preprocess_initializers(core, config["mean"], config["std"], 3, imgsz)
    core.graph.initializer.append(numpy_helper.from_array(np.asarray([2, 1, 0], dtype=np.int64), "preprocess/bgr_to_rgb_indices"))
    nodes = [helper.make_node("Transpose", ["images_c24_uint8_nhwc_bgr"], ["preprocess/bgr_nchw"], perm=[0, 3, 1, 2], name="preprocess/transpose"),
             helper.make_node("Gather", ["preprocess/bgr_nchw", "preprocess/bgr_to_rgb_indices"], ["preprocess/rgb_nchw"], axis=1, name="preprocess/bgr_to_rgb"),
             resize_node("preprocess/rgb_nchw", "preprocess/resized_uint8", sizes, config["onnx_mode"]),
             helper.make_node("Cast", ["preprocess/resized_uint8"], ["preprocess/resized_float"], to=TensorProto.FLOAT, name="preprocess/cast"),
             helper.make_node("Div", ["preprocess/resized_float", "preprocess/pixel_scale"], ["preprocess/scaled"], name="preprocess/scale"),
             helper.make_node("Sub", ["preprocess/scaled", "preprocess/mean"], ["preprocess/centered"], name="preprocess/mean"),
             helper.make_node("Div", ["preprocess/centered", "preprocess/std"], [core_input], name="preprocess/std")]
    finish_wrapper(core, nodes, output_path, "timm image classifier with C24 preprocessing.")


def image_paths(dataset: Path):
    return sorted(path for path in dataset.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def validate_wrappers(model_path, bw8_path, c24_path, dataset, imgsz, config):
    images = image_paths(dataset)
    if not images:
        raise ValueError(f"No images found below {dataset}")
    sessions = {name: ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                for name, path in (("reference", model_path), ("bw8", bw8_path), ("c24", c24_path))}
    input_names = {name: session.get_inputs()[0].name for name, session in sessions.items()}
    interpolation = getattr(transforms.InterpolationMode, config["interpolation"].upper())
    preprocessing = transforms.Compose([transforms.Resize((imgsz, imgsz), interpolation=interpolation, antialias=True), transforms.ToTensor(), transforms.Normalize(config["mean"], config["std"])])
    classes = [path.name for path in sorted(path for path in dataset.iterdir() if path.is_dir())]
    class_index = {name: index for index, name in enumerate(classes)}
    correct = {name: 0 for name in sessions}
    agreements = {name: 0 for name in ("bw8", "c24", "bw8_c24")}
    for path in images:
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            gray_rgb = source.convert("L").convert("RGB")
            tensors = {"reference": preprocessing(rgb).unsqueeze(0).numpy(),
                       "bw8": np.asarray(source.convert("L"), dtype=np.uint8)[None, None, ...],
                       "c24": np.asarray(rgb, dtype=np.uint8)[..., ::-1].copy()[None, ...]}
            scores = {name: sessions[name].run(None, {input_names[name]: value})[0][0] for name, value in tensors.items()}
        predictions = {name: int(np.argmax(value)) for name, value in scores.items()}
        expected = class_index[path.parent.name]
        for name, prediction in predictions.items():
            correct[name] += prediction == expected
        agreements["bw8"] += predictions["bw8"] == int(np.argmax(sessions["reference"].run(None, {input_names["reference"]: preprocessing(gray_rgb).unsqueeze(0).numpy()})[0][0]))
        agreements["c24"] += predictions["c24"] == predictions["reference"]
        agreements["bw8_c24"] += predictions["bw8"] == predictions["c24"]
    total = len(images)
    print(f"validation_images={total}")
    for name, count in correct.items():
        print(f"{name}_accuracy={count / total:.6f} ({count}/{total})")
    for name, count in agreements.items():
        print(f"{name}_agreement={count / total:.6f} ({count}/{total})")


def main() -> None:
    args = parse_args()
    if not args.embedded_preprocessing and (args.dataset or args.bw8_output or args.c24_output or args.skip_validation):
        raise ValueError("Dataset and wrapper options require --embedded-preprocessing")
    onnx_path, imgsz, config = export_onnx(args)
    if not args.embedded_preprocessing:
        return
    bw8_path = (args.bw8_output or onnx_path.with_name(f"{onnx_path.stem}-embedded-preprocess-bw8.onnx")).resolve()
    c24_path = (args.c24_output or onnx_path.with_name(f"{onnx_path.stem}-embedded-preprocess-c24.onnx")).resolve()
    wrap_bw8(onnx_path, bw8_path, imgsz, config)
    wrap_c24(onnx_path, c24_path, imgsz, config)
    print(f"created_bw8={bw8_path}")
    print(f"created_c24={c24_path}")
    if args.dataset and not args.skip_validation:
        validate_wrappers(onnx_path, bw8_path, c24_path, args.dataset.resolve(), imgsz, config)


if __name__ == "__main__":
    main()
