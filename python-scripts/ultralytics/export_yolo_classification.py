"""Export a YOLO classification checkpoint to ONNX.

The regular export accepts the float RGB NCHW tensor used by the original
preprocessing path.  With ``--embedded-preprocessing`` this script also emits
two uint8 models for the two camera input families:

* BW8: grayscale NCHW
* C24: BGR NHWC

The wrappers resize and normalize inside ONNX, so the caller can pass raw
image memory without duplicating preprocessing in its host application.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, compose, helper, numpy_helper
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="Path to the trained .pt checkpoint, typically weights/best.pt",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Square input image size used during export (default: 224)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="Optional ONNX opset version; uses the Ultralytics default when omitted",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Allow dynamic input batch/image dimensions on the regular ONNX model",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export FP16 where supported",
    )
    parser.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Simplify the ONNX graph (default: enabled)",
    )
    parser.add_argument(
        "--embedded-preprocessing",
        action="store_true",
        help="Also create raw uint8 BW8 and C24 ONNX wrappers",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Optional class-folder dataset used to validate embedded wrappers",
    )
    parser.add_argument("--bw8-output", type=Path, help="BW8 wrapper output path")
    parser.add_argument("--c24-output", type=Path, help="C24 wrapper output path")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Create embedded wrappers without dataset validation",
    )
    return parser.parse_args()


def export_onnx(args: argparse.Namespace) -> Path:
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if model_path.suffix.lower() != ".pt":
        raise ValueError(f"Expected an Ultralytics .pt checkpoint: {model_path}")

    model = YOLO(str(model_path), task="classify")
    options: dict[str, object] = {
        "format": "onnx",
        "imgsz": args.imgsz,
        "dynamic": args.dynamic,
        "half": args.half,
        "simplify": args.simplify,
    }
    if args.opset is not None:
        options["opset"] = args.opset

    print(f"Exporting {model_path.name} to ONNX...")
    output = Path(model.export(**options)).resolve()
    print(f"created_onnx={output}")
    return output


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


def add_initializers(
    core: onnx.ModelProto, channels: int, imgsz: int
) -> tuple[str, str]:
    sizes_name = "preprocess/target_sizes"
    scale_name = "preprocess/pixel_scale"
    core.graph.initializer.extend(
        [
            numpy_helper.from_array(
                np.asarray([1, channels, imgsz, imgsz], dtype=np.int64), sizes_name
            ),
            numpy_helper.from_array(np.asarray(255.0, dtype=np.float32), scale_name),
        ]
    )
    return sizes_name, scale_name


def finish_wrapper(
    core: onnx.ModelProto,
    preprocessing: list[onnx.NodeProto],
    output_path: Path,
    description: str,
) -> None:
    original_nodes = list(core.graph.node)
    del core.graph.node[:]
    core.graph.node.extend(preprocessing + original_nodes)
    core.graph.doc_string = description
    onnx.checker.check_model(core)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(core, output_path)


def wrap_bw8(model_path: Path, output_path: Path, imgsz: int) -> None:
    core, core_input = prepare_core(model_path)
    core.graph.input.append(
        helper.make_tensor_value_info(
            "images_bw8_uint8_nchw", TensorProto.UINT8, [1, 1, "height", "width"]
        )
    )
    sizes_name, scale_name = add_initializers(core, 1, imgsz)
    rgb_shape_name = "preprocess/rgb_shape"
    core.graph.initializer.append(
        numpy_helper.from_array(
            np.asarray([1, 3, imgsz, imgsz], dtype=np.int64), rgb_shape_name
        )
    )
    preprocessing = [
        helper.make_node(
            "Resize",
            ["images_bw8_uint8_nchw", "", "", sizes_name],
            ["preprocess/resized_uint8"],
            mode="linear",
            coordinate_transformation_mode="half_pixel",
            antialias=1,
            name="preprocess/resize",
        ),
        helper.make_node(
            "Cast", ["preprocess/resized_uint8"], ["preprocess/resized_float"],
            to=TensorProto.FLOAT, name="preprocess/cast"
        ),
        helper.make_node(
            "Div", ["preprocess/resized_float", scale_name],
            ["preprocess/normalized_gray"], name="preprocess/normalize"
        ),
        helper.make_node(
            "Expand", ["preprocess/normalized_gray", rgb_shape_name],
            [core_input], name="preprocess/gray_to_rgb"
        ),
    ]
    finish_wrapper(core, preprocessing, output_path, "YOLO classifier with standard BW8 preprocessing.")


def wrap_c24(model_path: Path, output_path: Path, imgsz: int) -> None:
    core, core_input = prepare_core(model_path)
    core.graph.input.append(
        helper.make_tensor_value_info(
            "images_c24_uint8_nhwc_bgr", TensorProto.UINT8, [1, "height", "width", 3]
        )
    )
    sizes_name, scale_name = add_initializers(core, 3, imgsz)
    channel_order_name = "preprocess/bgr_to_rgb_indices"
    core.graph.initializer.append(
        numpy_helper.from_array(np.asarray([2, 1, 0], dtype=np.int64), channel_order_name)
    )
    preprocessing = [
        helper.make_node("Transpose", ["images_c24_uint8_nhwc_bgr"],
                         ["preprocess/images_uint8_nchw_bgr"], perm=[0, 3, 1, 2],
                         name="preprocess/transpose"),
        helper.make_node("Gather", ["preprocess/images_uint8_nchw_bgr", channel_order_name],
                         ["preprocess/images_uint8_nchw_rgb"], axis=1,
                         name="preprocess/bgr_to_rgb"),
        helper.make_node("Resize", ["preprocess/images_uint8_nchw_rgb", "", "", sizes_name],
                         ["preprocess/resized_uint8"], mode="linear",
                         coordinate_transformation_mode="half_pixel", antialias=1,
                         name="preprocess/resize"),
        helper.make_node("Cast", ["preprocess/resized_uint8"], ["preprocess/resized_float"],
                         to=TensorProto.FLOAT, name="preprocess/cast"),
        helper.make_node("Div", ["preprocess/resized_float", scale_name], [core_input],
                         name="preprocess/normalize"),
    ]
    finish_wrapper(core, preprocessing, output_path, "YOLO classifier with standard C24 preprocessing.")


def image_paths(dataset: Path) -> list[Path]:
    return sorted(path for path in dataset.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def validate_wrappers(
    model_path: Path, bw8_path: Path, c24_path: Path, dataset: Path, imgsz: int
) -> None:
    images = image_paths(dataset)
    if not images:
        raise ValueError(f"No images found below {dataset}")
    sessions = {
        "reference": ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"]),
        "bw8": ort.InferenceSession(str(bw8_path), providers=["CPUExecutionProvider"]),
        "c24": ort.InferenceSession(str(c24_path), providers=["CPUExecutionProvider"]),
    }
    input_names = {name: session.get_inputs()[0].name for name, session in sessions.items()}
    preprocessing = transforms.Compose([
        transforms.Resize((imgsz, imgsz), antialias=True), transforms.ToTensor()
    ])
    class_names = [path.name for path in sorted(path for path in dataset.iterdir() if path.is_dir())]
    class_index = {name: index for index, name in enumerate(class_names)}
    correct = {name: 0 for name in sessions}
    agreements = {"bw8": 0, "c24": 0, "bw8_c24": 0}
    maximum_score_delta = {name: 0.0 for name in agreements}
    for path in images:
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            tensors = {
                "reference": preprocessing(rgb).unsqueeze(0).numpy(),
                "bw8": np.asarray(source.convert("L"), dtype=np.uint8)[None, None, ...],
                "c24": np.asarray(rgb, dtype=np.uint8)[..., ::-1].copy()[None, ...],
            }
        scores = {name: sessions[name].run(None, {input_names[name]: tensor})[0][0]
                  for name, tensor in tensors.items()}
        predictions = {name: int(np.argmax(value)) for name, value in scores.items()}
        expected = class_index[path.parent.name]
        for name, prediction in predictions.items():
            correct[name] += prediction == expected
        agreements["bw8"] += predictions["bw8"] == predictions["reference"]
        agreements["c24"] += predictions["c24"] == predictions["reference"]
        agreements["bw8_c24"] += predictions["bw8"] == predictions["c24"]
        for name in ("bw8", "c24"):
            maximum_score_delta[name] = max(maximum_score_delta[name],
                float(np.max(np.abs(scores[name] - scores["reference"]))))
        maximum_score_delta["bw8_c24"] = max(maximum_score_delta["bw8_c24"],
            float(np.max(np.abs(scores["bw8"] - scores["c24"]))))
    total = len(images)
    print(f"validation_images={total}")
    for name in sessions:
        print(f"{name}_accuracy={correct[name] / total:.6f} ({correct[name]}/{total})")
    for name, count in agreements.items():
        print(f"{name}_agreement={count / total:.6f} ({count}/{total})")
        print(f"{name}_maximum_absolute_score_delta={maximum_score_delta[name]:.9f}")


def main() -> None:
    args = parse_args()
    if not args.embedded_preprocessing:
        if args.dataset or args.bw8_output or args.c24_output or args.skip_validation:
            raise ValueError(
                "--dataset, wrapper output, and --skip-validation options require "
                "--embedded-preprocessing"
            )
        export_onnx(args)
        return

    onnx_path = export_onnx(args)

    bw8_path = (args.bw8_output or onnx_path.with_name(
        f"{onnx_path.stem}-embedded-preprocess-bw8.onnx")).resolve()
    c24_path = (args.c24_output or onnx_path.with_name(
        f"{onnx_path.stem}-embedded-preprocess-c24.onnx")).resolve()
    wrap_bw8(onnx_path, bw8_path, args.imgsz)
    wrap_c24(onnx_path, c24_path, args.imgsz)
    print(f"created_bw8={bw8_path}")
    print(f"created_c24={c24_path}")
    print("bw8_input=images_bw8_uint8_nchw uint8 NCHW dynamic-height-width grayscale")
    print("c24_input=images_c24_uint8_nhwc_bgr uint8 NHWC dynamic-height-width BGR")
    if args.dataset and not args.skip_validation:
        validate_wrappers(onnx_path, bw8_path, c24_path, args.dataset.resolve(), args.imgsz)


if __name__ == "__main__":
    main()
