"""Export YOLO26 detection weights to ONNX and the repository detection contract.

The contract outputs normalized ``boxes`` [1,N,4] in xyxy form, ``scores``
[1,N], and zero-based int64 ``class_ids`` [1,N]. Optional BW8 and C24 models
embed stretch-resize and uint8 preprocessing for the .NET ObjectDetector.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

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
        "--simplify", action=BooleanOptionalAction, default=True,
        help="Let Ultralytics simplify the exported graph (default: enabled)",
    )
    parser.add_argument("--skip-embedded-preprocessing", action="store_true")
    parser.add_argument("--validate-image", type=Path)
    return parser.parse_args()


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


def validate(path: Path, image_path: Path) -> None:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
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
    result = {
        item.name: value
        for item, value in zip(session.get_outputs(), session.run(None, {input_info.name: value}))
    }
    top = int(np.argmax(result["scores"][0]))
    print(f"validated={path}")
    print(f"top_score={float(result['scores'][0, top]):.6f}")
    print(f"top_class_id={int(result['class_ids'][0, top])}")
    print(f"top_box={result['boxes'][0, top].tolist()}")


def main() -> None:
    args = parse_args()
    if args.resolution < 32 or args.resolution % 32:
        raise ValueError("--resolution must be at least 32 and divisible by 32")
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
            simplify=args.simplify, opset=args.opset,
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
        for path in generated:
            validate(path, args.validate_image.resolve())


if __name__ == "__main__":
    main()
