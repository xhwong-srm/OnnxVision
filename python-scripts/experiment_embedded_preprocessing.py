"""Export and validate BW8 and C24 preprocessing wrappers for a YOLO classifier.

The BW8 wrapper accepts a decoded grayscale uint8 image in NCHW layout. The
C24 wrapper accepts the raw BGR-interleaved memory layout exposed by an Open
eVision EImageC24/EROIC24 image in NHWC layout. Both graphs perform resize,
float conversion, [0, 1] scaling, and conversion to the classifier's RGB NCHW
input before executing the same original classifier core.
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


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Original float NCHW ONNX classifier")
    parser.add_argument("dataset", type=Path, help="Class-folder dataset to compare")
    parser.add_argument("--bw8-output", type=Path, help="BW8 wrapped model output path")
    parser.add_argument("--c24-output", type=Path, help="C24 wrapped model output path")
    parser.add_argument("--imgsz", type=int, default=224)
    return parser.parse_args()


def prepare_core(model_path: Path) -> tuple[onnx.ModelProto, str]:
    model = onnx.load(model_path)
    if len(model.graph.input) != 1:
        raise ValueError("Expected exactly one model input")
    if not any(item.domain == "" and item.version >= 18 for item in model.opset_import):
        raise ValueError("Antialiased ONNX Resize requires opset 18 or newer")

    core = compose.add_prefix(model, "core/")
    core_input = core.graph.input[0].name
    del core.graph.input[:]
    return core, core_input


def add_common_initializers(
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
    sizes_name, scale_name = add_common_initializers(core, 1, imgsz)
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
            "Cast",
            ["preprocess/resized_uint8"],
            ["preprocess/resized_float"],
            to=TensorProto.FLOAT,
            name="preprocess/cast",
        ),
        helper.make_node(
            "Div",
            ["preprocess/resized_float", scale_name],
            ["preprocess/normalized_gray"],
            name="preprocess/normalize",
        ),
        helper.make_node(
            "Expand",
            ["preprocess/normalized_gray", rgb_shape_name],
            [core_input],
            name="preprocess/gray_to_rgb",
        ),
    ]
    finish_wrapper(
        core,
        preprocessing,
        output_path,
        "YOLO classifier with Open eVision BW8 uint8 NCHW preprocessing.",
    )


def wrap_c24(model_path: Path, output_path: Path, imgsz: int) -> None:
    core, core_input = prepare_core(model_path)
    core.graph.input.append(
        helper.make_tensor_value_info(
            "images_c24_uint8_nhwc_bgr", TensorProto.UINT8, [1, "height", "width", 3]
        )
    )
    sizes_name, scale_name = add_common_initializers(core, 3, imgsz)
    channel_order_name = "preprocess/bgr_to_rgb_indices"
    core.graph.initializer.append(
        numpy_helper.from_array(
            np.asarray([2, 1, 0], dtype=np.int64), channel_order_name
        )
    )

    preprocessing = [
        helper.make_node(
            "Transpose",
            ["images_c24_uint8_nhwc_bgr"],
            ["preprocess/images_uint8_nchw_bgr"],
            perm=[0, 3, 1, 2],
            name="preprocess/transpose",
        ),
        helper.make_node(
            "Gather",
            ["preprocess/images_uint8_nchw_bgr", channel_order_name],
            ["preprocess/images_uint8_nchw_rgb"],
            axis=1,
            name="preprocess/bgr_to_rgb",
        ),
        helper.make_node(
            "Resize",
            ["preprocess/images_uint8_nchw_rgb", "", "", sizes_name],
            ["preprocess/resized_uint8"],
            mode="linear",
            coordinate_transformation_mode="half_pixel",
            antialias=1,
            name="preprocess/resize",
        ),
        helper.make_node(
            "Cast",
            ["preprocess/resized_uint8"],
            ["preprocess/resized_float"],
            to=TensorProto.FLOAT,
            name="preprocess/cast",
        ),
        helper.make_node(
            "Div",
            ["preprocess/resized_float", scale_name],
            [core_input],
            name="preprocess/normalize",
        ),
    ]
    finish_wrapper(
        core,
        preprocessing,
        output_path,
        "YOLO classifier with Open eVision C24 uint8 NHWC BGR preprocessing.",
    )


def image_paths(dataset: Path) -> list[Path]:
    return sorted(
        path for path in dataset.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    dataset = args.dataset.resolve()
    bw8_output = (
        args.bw8_output.resolve()
        if args.bw8_output
        else model_path.with_name(f"{model_path.stem}-embedded-preprocess-bw8.onnx")
    )
    c24_output = (
        args.c24_output.resolve()
        if args.c24_output
        else model_path.with_name(f"{model_path.stem}-embedded-preprocess-c24.onnx")
    )
    images = image_paths(dataset)
    if not images:
        raise SystemExit(f"No images found below {dataset}")

    wrap_bw8(model_path, bw8_output, args.imgsz)
    wrap_c24(model_path, c24_output, args.imgsz)

    sessions = {
        "reference": ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"]),
        "bw8": ort.InferenceSession(str(bw8_output), providers=["CPUExecutionProvider"]),
        "c24": ort.InferenceSession(str(c24_output), providers=["CPUExecutionProvider"]),
    }
    input_names = {name: session.get_inputs()[0].name for name, session in sessions.items()}
    preprocessing = transforms.Compose(
        [transforms.Resize((args.imgsz, args.imgsz), antialias=True), transforms.ToTensor()]
    )

    class_names = [path.name for path in sorted(path for path in dataset.iterdir() if path.is_dir())]
    class_index = {name: index for index, name in enumerate(class_names)}
    correct = {name: 0 for name in sessions}
    agreements = {"bw8": 0, "c24": 0, "bw8_c24": 0}
    maximum_score_delta = {"bw8": 0.0, "c24": 0.0, "bw8_c24": 0.0}
    changed: list[str] = []

    for path in images:
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            gray = source.convert("L")
            tensors = {
                "reference": preprocessing(rgb).unsqueeze(0).numpy(),
                "bw8": np.asarray(gray, dtype=np.uint8)[None, None, ...],
                # EImageC24 uses Windows-compatible BGR byte order in memory.
                "c24": np.asarray(rgb, dtype=np.uint8)[..., ::-1].copy()[None, ...],
            }

        scores = {
            name: sessions[name].run(None, {input_names[name]: tensor})[0][0]
            for name, tensor in tensors.items()
        }
        predictions = {name: int(np.argmax(value)) for name, value in scores.items()}
        expected = class_index[path.parent.name]
        for name, prediction in predictions.items():
            correct[name] += prediction == expected

        agreements["bw8"] += predictions["bw8"] == predictions["reference"]
        agreements["c24"] += predictions["c24"] == predictions["reference"]
        agreements["bw8_c24"] += predictions["bw8"] == predictions["c24"]
        maximum_score_delta["bw8"] = max(
            maximum_score_delta["bw8"],
            float(np.max(np.abs(scores["bw8"] - scores["reference"]))),
        )
        maximum_score_delta["c24"] = max(
            maximum_score_delta["c24"],
            float(np.max(np.abs(scores["c24"] - scores["reference"]))),
        )
        maximum_score_delta["bw8_c24"] = max(
            maximum_score_delta["bw8_c24"],
            float(np.max(np.abs(scores["bw8"] - scores["c24"]))),
        )
        if len(set(predictions.values())) != 1:
            changed.append(
                f"{path.relative_to(dataset)}: "
                + " ".join(f"{name}={class_names[index]}" for name, index in predictions.items())
            )

    total = len(images)
    print(f"created_bw8={bw8_output}")
    print(f"created_c24={c24_output}")
    print("bw8_input=images_bw8_uint8_nchw uint8 NCHW dynamic-height-width grayscale")
    print("c24_input=images_c24_uint8_nhwc_bgr uint8 NHWC dynamic-height-width BGR")
    print(f"images={total}")
    for name in sessions:
        print(f"{name}_accuracy={correct[name] / total:.6f} ({correct[name]}/{total})")
    for name, count in agreements.items():
        print(f"{name}_agreement={count / total:.6f} ({count}/{total})")
        print(f"{name}_maximum_absolute_score_delta={maximum_score_delta[name]:.9f}")
    for item in changed:
        print(f"changed={item}")


if __name__ == "__main__":
    main()
