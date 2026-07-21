"""Wrap and validate a YOLO classification ONNX model with image preprocessing.

The wrapped model accepts a decoded RGB uint8 image in NHWC layout. Resize,
NHWC-to-NCHW conversion, float conversion, and [0, 1] scaling are performed by
the ONNX graph before the original classifier.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, compose, helper, numpy_helper
from PIL import Image
from torchvision import transforms


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Original float NCHW ONNX classifier")
    parser.add_argument("dataset", type=Path, help="Class-folder dataset to compare")
    parser.add_argument("--output", type=Path, help="Wrapped model output path")
    parser.add_argument("--imgsz", type=int, default=224)
    return parser.parse_args()


def wrap_model(model_path: Path, output_path: Path, imgsz: int) -> None:
    model = onnx.load(model_path)
    if len(model.graph.input) != 1:
        raise ValueError("Expected exactly one model input")
    if not any(item.domain == "" and item.version >= 18 for item in model.opset_import):
        raise ValueError("Antialiased ONNX Resize requires opset 18 or newer")

    core = compose.add_prefix(model, "core/")
    core_input = core.graph.input[0].name
    del core.graph.input[:]
    core.graph.input.append(
        helper.make_tensor_value_info(
            "images_uint8_nhwc", TensorProto.UINT8, [1, "height", "width", 3]
        )
    )

    sizes = numpy_helper.from_array(
        np.asarray([1, 3, imgsz, imgsz], dtype=np.int64), "preprocess/target_sizes"
    )
    scale = numpy_helper.from_array(
        np.asarray(255.0, dtype=np.float32), "preprocess/pixel_scale"
    )
    core.graph.initializer.extend([sizes, scale])

    preprocessing = [
        helper.make_node(
            "Transpose",
            ["images_uint8_nhwc"],
            ["preprocess/images_uint8_nchw"],
            perm=[0, 3, 1, 2],
            name="preprocess/transpose",
        ),
        helper.make_node(
            "Resize",
            ["preprocess/images_uint8_nchw", "", "", "preprocess/target_sizes"],
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
            ["preprocess/resized_float", "preprocess/pixel_scale"],
            [core_input],
            name="preprocess/normalize",
        ),
    ]
    original_nodes = list(core.graph.node)
    del core.graph.node[:]
    core.graph.node.extend(preprocessing + original_nodes)
    core.graph.doc_string = (
        "YOLO classification model wrapped with uint8 NHWC RGB preprocessing: "
        "antialiased linear resize, NCHW conversion, and [0,1] scaling."
    )

    onnx.checker.check_model(core)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(core, output_path)


def image_paths(dataset: Path) -> list[Path]:
    return sorted(
        path for path in dataset.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    dataset = args.dataset.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else model_path.with_name(f"{model_path.stem}-embedded-preprocess.onnx")
    )
    images = image_paths(dataset)
    if not images:
        raise SystemExit(f"No images found below {dataset}")

    wrap_model(model_path, output_path, args.imgsz)
    reference_session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    wrapped_session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    reference_input = reference_session.get_inputs()[0].name
    wrapped_input = wrapped_session.get_inputs()[0].name
    preprocessing = transforms.Compose(
        [transforms.Resize((args.imgsz, args.imgsz), antialias=True), transforms.ToTensor()]
    )

    class_names = [path.name for path in sorted(path for path in dataset.iterdir() if path.is_dir())]
    class_index = {name: index for index, name in enumerate(class_names)}
    agreements = reference_correct = wrapped_correct = 0
    maximum_score_delta = 0.0
    changed: list[str] = []

    for path in images:
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            reference_tensor = preprocessing(rgb).unsqueeze(0).numpy()
            wrapped_tensor = np.asarray(rgb, dtype=np.uint8)[None, ...]

        reference_scores = reference_session.run(None, {reference_input: reference_tensor})[0][0]
        wrapped_scores = wrapped_session.run(None, {wrapped_input: wrapped_tensor})[0][0]
        reference_prediction = int(np.argmax(reference_scores))
        wrapped_prediction = int(np.argmax(wrapped_scores))
        expected = class_index[path.parent.name]
        reference_correct += reference_prediction == expected
        wrapped_correct += wrapped_prediction == expected
        agreements += reference_prediction == wrapped_prediction
        maximum_score_delta = max(
            maximum_score_delta, float(np.max(np.abs(reference_scores - wrapped_scores)))
        )
        if reference_prediction != wrapped_prediction:
            changed.append(
                f"{path.relative_to(dataset)}: reference={class_names[reference_prediction]} "
                f"wrapped={class_names[wrapped_prediction]}"
            )

    total = len(images)
    print(f"created={output_path}")
    print(f"input={wrapped_input} uint8 NHWC dynamic-height-width RGB")
    print(f"images={total}")
    print(f"prediction_agreement={agreements / total:.6f} ({agreements}/{total})")
    print(f"reference_accuracy={reference_correct / total:.6f} ({reference_correct}/{total})")
    print(f"wrapped_accuracy={wrapped_correct / total:.6f} ({wrapped_correct}/{total})")
    print(f"maximum_absolute_score_delta={maximum_score_delta:.9f}")
    for item in changed:
        print(f"changed={item}")


if __name__ == "__main__":
    main()
