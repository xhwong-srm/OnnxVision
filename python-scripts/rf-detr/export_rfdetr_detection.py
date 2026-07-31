"""Export RF-DETR detection models to the OnnxVision detection contract.

The exported BW8 and C24 models accept raw production images and expose a
model-neutral candidate interface:

* ``boxes``: float32 ``[1,N,4]`` normalized XYXY boxes
* ``scores``: float32 ``[1,N]`` confidence probabilities
* ``class_ids``: int64 ``[1,N]`` zero-based indices into ``names`` metadata

RF-DETR-specific preprocessing, sparse COCO IDs, sigmoid, top-k selection,
and CXCYWH-to-XYXY conversion are contained in the ONNX graph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, compose, helper, numpy_helper
from PIL import Image
from rfdetr import RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES


CONTRACT_VERSION = "onnx-vision-detection-v1"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Optional RF-DETR Nano checkpoint")
    parser.add_argument("--output", type=Path, default=Path("artifacts/rfdetr-nano-detection.onnx"))
    parser.add_argument("--bw8-output", type=Path)
    parser.add_argument("--c24-output", type=Path)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--skip-embedded-preprocessing", action="store_true")
    parser.add_argument("--validate-image", type=Path)
    return parser.parse_args()


def set_metadata(model: onnx.ModelProto, values: dict[str, str]) -> None:
    del model.metadata_props[:]
    for key, value in values.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value


def add_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> str:
    model.graph.initializer.append(numpy_helper.from_array(value, name))
    return name


def canonicalize_outputs(
    official_path: Path,
    output_path: Path,
    class_names: list[str],
    source_class_ids: list[int],
    max_detections: int,
) -> None:
    model = onnx.load(official_path)
    if len(model.graph.output) != 2:
        raise ValueError("Expected RF-DETR outputs (dets, labels)")
    boxes_input, logits_input = (item.name for item in model.graph.output)
    del model.graph.output[:]

    prefix = "contract/"
    class_indices = add_initializer(
        model, prefix + "source_class_ids", np.asarray(source_class_ids, dtype=np.int64)
    )
    topk = min(max_detections, 300 * len(class_names))
    topk_value = add_initializer(model, prefix + "topk", np.asarray([topk], dtype=np.int64))
    divisor = add_initializer(
        model, prefix + "class_count", np.asarray(len(class_names), dtype=np.int64)
    )
    axes = add_initializer(model, prefix + "unsqueeze_axes", np.asarray([2], dtype=np.int64))
    two = add_initializer(model, prefix + "two", np.asarray(2.0, dtype=np.float32))
    clip_min = add_initializer(model, prefix + "clip_min", np.asarray(0.0, dtype=np.float32))
    clip_max = add_initializer(model, prefix + "clip_max", np.asarray(1.0, dtype=np.float32))

    nodes = [
        helper.make_node("Gather", [logits_input, class_indices], [prefix + "logits"], axis=2),
        helper.make_node("Sigmoid", [prefix + "logits"], [prefix + "probabilities"]),
        helper.make_node("Flatten", [prefix + "probabilities"], [prefix + "flat_scores"], axis=1),
        helper.make_node(
            "TopK",
            [prefix + "flat_scores", topk_value],
            ["scores", prefix + "flat_indices"],
            axis=1,
            largest=1,
            sorted=1,
        ),
        helper.make_node("Mod", [prefix + "flat_indices", divisor], ["class_ids"], fmod=0),
        helper.make_node("Div", [prefix + "flat_indices", divisor], [prefix + "query_ids"]),
        helper.make_node("Split", [boxes_input], [
            prefix + "cx", prefix + "cy", prefix + "width", prefix + "height"
        ], axis=2, num_outputs=4),
        helper.make_node("Div", [prefix + "width", two], [prefix + "half_width"]),
        helper.make_node("Div", [prefix + "height", two], [prefix + "half_height"]),
        helper.make_node("Sub", [prefix + "cx", prefix + "half_width"], [prefix + "x1"]),
        helper.make_node("Sub", [prefix + "cy", prefix + "half_height"], [prefix + "y1"]),
        helper.make_node("Add", [prefix + "cx", prefix + "half_width"], [prefix + "x2"]),
        helper.make_node("Add", [prefix + "cy", prefix + "half_height"], [prefix + "y2"]),
        helper.make_node("Concat", [
            prefix + "x1", prefix + "y1", prefix + "x2", prefix + "y2"
        ], [prefix + "all_xyxy"], axis=2),
        helper.make_node("Unsqueeze", [prefix + "query_ids", axes], [prefix + "query_ids_3d"]),
        helper.make_node("Concat", [prefix + "query_ids_3d"] * 4, [prefix + "box_indices"], axis=2),
        helper.make_node(
            "GatherElements", [prefix + "all_xyxy", prefix + "box_indices"], [prefix + "selected_boxes"], axis=1
        ),
        helper.make_node("Clip", [prefix + "selected_boxes", clip_min, clip_max], ["boxes"]),
    ]
    model.graph.node.extend(nodes)
    model.graph.output.extend([
        helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, topk, 4]),
        helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, topk]),
        helper.make_tensor_value_info("class_ids", TensorProto.INT64, [1, topk]),
    ])
    metadata = {
        "vision_task": "object_detection",
        "detection_contract": CONTRACT_VERSION,
        "box_format": "xyxy",
        "box_coordinates": "normalized",
        "names": json.dumps({index: name for index, name in enumerate(class_names)}),
        "source_model": "rfdetr-nano",
        "candidate_count": str(topk),
    }
    set_metadata(model, metadata)
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)


def prepare_core(path: Path) -> tuple[onnx.ModelProto, str, dict[str, str]]:
    model = onnx.load(path)
    metadata = {item.key: item.value for item in model.metadata_props}
    core = compose.add_prefix(model, "core/", rename_outputs=False)
    input_name = core.graph.input[0].name
    del core.graph.input[:]
    return core, input_name, metadata


def add_preprocessing_initializers(model: onnx.ModelProto, channels: int, resolution: int) -> str:
    sizes = add_initializer(
        model, "preprocess/target_sizes", np.asarray([1, channels, resolution, resolution], dtype=np.int64)
    )
    add_initializer(model, "preprocess/pixel_scale", np.asarray([[[[255.0]]]], dtype=np.float32))
    add_initializer(model, "preprocess/mean", np.asarray(MEAN, dtype=np.float32).reshape(1, 3, 1, 1))
    add_initializer(model, "preprocess/std", np.asarray(STD, dtype=np.float32).reshape(1, 3, 1, 1))
    return sizes


def resize_node(source: str, result: str, sizes: str) -> onnx.NodeProto:
    return helper.make_node(
        "Resize", [source, "", "", sizes], [result],
        mode="linear", coordinate_transformation_mode="half_pixel", antialias=1
    )


def finish_wrapper(
    core: onnx.ModelProto,
    nodes: list[onnx.NodeProto],
    output_path: Path,
    metadata: dict[str, str],
    description: str,
) -> None:
    original_nodes = list(core.graph.node)
    del core.graph.node[:]
    core.graph.node.extend(nodes + original_nodes)
    core.graph.doc_string = description
    set_metadata(core, metadata)
    onnx.checker.check_model(core)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(core, output_path)


def wrap_bw8(core_path: Path, output_path: Path, resolution: int) -> None:
    core, core_input, metadata = prepare_core(core_path)
    core.graph.input.append(helper.make_tensor_value_info(
        "images_bw8_uint8_nchw", TensorProto.UINT8, [1, 1, "height", "width"]
    ))
    sizes = add_preprocessing_initializers(core, 1, resolution)
    rgb_shape = add_initializer(
        core, "preprocess/rgb_shape", np.asarray([1, 3, resolution, resolution], dtype=np.int64)
    )
    nodes = [
        resize_node("images_bw8_uint8_nchw", "preprocess/resized_uint8", sizes),
        helper.make_node("Cast", ["preprocess/resized_uint8"], ["preprocess/resized_float"], to=TensorProto.FLOAT),
        helper.make_node("Div", ["preprocess/resized_float", "preprocess/pixel_scale"], ["preprocess/scaled"]),
        helper.make_node("Expand", ["preprocess/scaled", rgb_shape], ["preprocess/rgb"]),
        helper.make_node("Sub", ["preprocess/rgb", "preprocess/mean"], ["preprocess/centered"]),
        helper.make_node("Div", ["preprocess/centered", "preprocess/std"], [core_input]),
    ]
    finish_wrapper(core, nodes, output_path, metadata, "Object detector with raw BW8 preprocessing.")


def wrap_c24(core_path: Path, output_path: Path, resolution: int) -> None:
    core, core_input, metadata = prepare_core(core_path)
    core.graph.input.append(helper.make_tensor_value_info(
        "images_c24_uint8_nhwc_bgr", TensorProto.UINT8, [1, "height", "width", 3]
    ))
    sizes = add_preprocessing_initializers(core, 3, resolution)
    indices = add_initializer(
        core, "preprocess/bgr_to_rgb_indices", np.asarray([2, 1, 0], dtype=np.int64)
    )
    nodes = [
        helper.make_node("Transpose", ["images_c24_uint8_nhwc_bgr"], ["preprocess/bgr_nchw"], perm=[0, 3, 1, 2]),
        helper.make_node("Gather", ["preprocess/bgr_nchw", indices], ["preprocess/rgb_nchw"], axis=1),
        resize_node("preprocess/rgb_nchw", "preprocess/resized_uint8", sizes),
        helper.make_node("Cast", ["preprocess/resized_uint8"], ["preprocess/resized_float"], to=TensorProto.FLOAT),
        helper.make_node("Div", ["preprocess/resized_float", "preprocess/pixel_scale"], ["preprocess/scaled"]),
        helper.make_node("Sub", ["preprocess/scaled", "preprocess/mean"], ["preprocess/centered"]),
        helper.make_node("Div", ["preprocess/centered", "preprocess/std"], [core_input]),
    ]
    finish_wrapper(core, nodes, output_path, metadata, "Object detector with raw C24 BGR preprocessing.")


def validate_model(path: Path, image_path: Path) -> None:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_metadata = session.get_inputs()[0]
    with Image.open(image_path) as image:
        if input_metadata.type == "tensor(uint8)" and input_metadata.shape[1] == 1:
            value = np.asarray(image.convert("L"), dtype=np.uint8)[None, None]
        elif input_metadata.type == "tensor(uint8)":
            value = np.asarray(image.convert("RGB"), dtype=np.uint8)[..., ::-1].copy()[None]
        else:
            rgb = image.convert("RGB").resize((384, 384))
            value = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
            value = (value - np.asarray(MEAN, dtype=np.float32)[None, :, None, None])
            value /= np.asarray(STD, dtype=np.float32)[None, :, None, None]
    outputs = {item.name: value for item, value in zip(session.get_outputs(), session.run(None, {input_metadata.name: value}))}
    print(f"validated={path}")
    print(f"top_score={float(outputs['scores'][0, 0]):.6f}")
    print(f"top_class_id={int(outputs['class_ids'][0, 0])}")
    print(f"top_box={outputs['boxes'][0, 0].tolist()}")


def main() -> None:
    args = parse_args()
    if args.max_detections <= 0:
        raise ValueError("--max-detections must be positive")
    kwargs = {}
    if args.checkpoint:
        kwargs["pretrain_weights"] = str(args.checkpoint.expanduser().resolve())
    model = RFDETRNano(**kwargs)
    class_names = list(model.class_names)
    source_class_ids = list(COCO_CLASSES) if class_names == list(COCO_CLASSES.values()) else list(range(len(class_names)))
    resolution = int(model.model.resolution)

    output = args.output.expanduser().resolve()
    official_dir = output.parent / f".{output.stem}-official"
    official_path = Path(model.export(
        output_dir=str(official_dir), format="onnx", opset_version=args.opset, fp16=False, verbose=False
    )).resolve()
    canonicalize_outputs(official_path, output, class_names, source_class_ids, args.max_detections)
    print(f"created_onnx={output}")
    if not args.skip_embedded_preprocessing:
        bw8 = (args.bw8_output or output.with_name(output.stem + "-bw8.onnx")).resolve()
        c24 = (args.c24_output or output.with_name(output.stem + "-c24.onnx")).resolve()
        wrap_bw8(output, bw8, resolution)
        wrap_c24(output, c24, resolution)
        print(f"created_bw8={bw8}")
        print(f"created_c24={c24}")
        if args.validate_image:
            validate_model(bw8, args.validate_image.resolve())
            validate_model(c24, args.validate_image.resolve())
    elif args.validate_image:
        validate_model(output, args.validate_image.resolve())


if __name__ == "__main__":
    main()
