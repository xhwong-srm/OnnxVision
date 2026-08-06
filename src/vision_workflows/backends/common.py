from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..domain.errors import BackendUnavailableError, ConfigurationError
from ..domain.results import ArtifactRef
from ..workflows.context import optional_import
from ..workflows.runs import artifact


CONTRACT_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
DETECTION_CONTRACT_NAME = "onnx-vision-object-detection"
CLASSIFICATION_CONTRACT_NAME = "onnx-vision-classification"
CONTRACT_VERSION = "2.0.0"


def embedded_input_contract() -> dict[str, Any]:
    """Return the shared raw-image input contract used by both ONNX tasks."""
    return {
        "batch": {"axis": 0, "minimum": 1, "dynamic": True},
        "variants": {
            "bw8": {
                "name": "images_bw8_uint8_nchw",
                "dtype": "uint8",
                "layout": "NCHW",
                "pixel_format": "BW8",
                "shape": ["B", 1, "H", "W"],
                "preprocessing": "embedded",
            },
            "c24": {
                "name": "images_c24_uint8_nhwc_bgr",
                "dtype": "uint8",
                "layout": "NHWC",
                "pixel_format": "C24_BGR",
                "shape": ["B", "H", "W", 3],
                "preprocessing": "embedded",
            },
        },
    }


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ConfigurationError(f"{label} does not exist: {path}")
    return path


def optional_module(module: str):
    try:
        return optional_import(module)
    except BackendUnavailableError:
        raise


def detection_contract(
    names: list[str] | tuple[str, ...],
    *,
    nms_required: bool = False,
    input_variant: str | None = None,
) -> dict[str, Any]:
    contract = {
        "name": DETECTION_CONTRACT_NAME,
        "version": CONTRACT_VERSION,
        "task": "object_detection",
        "inputs": embedded_input_contract(),
        "outputs": {
            "boxes": {
                "dtype": "float32",
                "shape": ["B", "Q", 4],
                "coordinate_format": "xyxy",
                "coordinate_space": "normalized_0_1",
            },
            "scores": {"dtype": "float32", "shape": ["B", "Q"]},
            "class_ids": {"dtype": "int64", "shape": ["B", "Q"]},
        },
        "names": {str(index): name for index, name in enumerate(names)},
        "nms_required": nms_required,
    }
    if input_variant is not None:
        _validate_input_variant(input_variant)
        contract["input_variant"] = input_variant
    return contract


def classification_contract(
    names: list[str] | tuple[str, ...], *, input_variant: str | None = None
) -> dict[str, Any]:
    contract = {
        "name": CLASSIFICATION_CONTRACT_NAME,
        "version": CONTRACT_VERSION,
        "task": "classification",
        "inputs": embedded_input_contract(),
        "outputs": {"probabilities": {"dtype": "float32", "shape": ["B", "C"]}},
        "names": {str(index): name for index, name in enumerate(names)},
    }
    if input_variant is not None:
        _validate_input_variant(input_variant)
        contract["input_variant"] = input_variant
    return contract


def _validate_input_variant(value: str) -> None:
    if value not in {"bw8", "c24"}:
        raise ValueError(f"unsupported embedded input variant: {value}")


def metadata_for_contract(contract: dict[str, Any]) -> dict[str, Any]:
    version = str(contract["version"])
    if not CONTRACT_VERSION_PATTERN.fullmatch(version):
        raise ValueError(f"contract version must use major.minor.micro format: {version}")

    metadata = {
        "vision_task": contract["task"],
        "contract_name": contract["name"],
        "contract_version": version,
        "names": contract["names"],
        "inputs": contract["inputs"],
        "outputs": contract["outputs"],
    }
    if "input_variant" in contract:
        metadata["input_variant"] = contract["input_variant"]
    if contract["task"] == "object_detection":
        metadata["nms_required"] = bool(contract["nms_required"])
    return metadata


def class_names_from_model(model: Any) -> tuple[str, ...]:
    raw_names = getattr(model, "names", None)
    if raw_names is None:
        raw_names = getattr(getattr(model, "model", None), "names", None)
    if isinstance(raw_names, Mapping):
        raw_names = [raw_names[key] for key in sorted(raw_names, key=lambda value: int(value))]
    if not isinstance(raw_names, (list, tuple)):
        return ()
    return tuple(str(name).strip() for name in raw_names if str(name).strip())


def validate_onnx(path: Path, contract: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    onnx = optional_module("onnx")
    model = onnx.load(str(require_file(path, "ONNX artifact")))
    onnx.checker.check_model(model)
    metadata = {item.key: item.value for item in model.metadata_props}
    checks = [{"name": "onnx_checker", "status": "passed"}]
    expected = metadata_for_contract(contract)
    for key in ("vision_task", "contract_name", "contract_version", "input_variant"):
        if key not in expected:
            continue
        if metadata.get(key) != expected[key]:
            checks.append({"name": key, "status": "failed", "expected": expected[key], "actual": metadata.get(key)})

    for key in ("inputs", "outputs"):
        try:
            actual_value = json.loads(metadata.get(key, ""))
        except json.JSONDecodeError:
            actual_value = None
        if actual_value != expected[key]:
            checks.append({"name": key, "status": "failed", "expected": expected[key], "actual": actual_value})

    if contract["task"] == "object_detection":
        nms_required = metadata.get("nms_required")
        if nms_required not in {"true", "false"}:
            checks.append({"name": "nms_required", "status": "failed", "expected": "true or false", "actual": nms_required})

    expected_names = contract.get("names") or {}
    try:
        actual_names = json.loads(metadata.get("names", ""))
    except json.JSONDecodeError:
        actual_names = None
    if not isinstance(actual_names, dict) or not actual_names:
        checks.append({"name": "names", "status": "failed", "expected": "non-empty class mapping", "actual": actual_names})
    elif expected_names and actual_names != expected_names:
        checks.append({"name": "names", "status": "failed", "expected": expected_names, "actual": actual_names})

    _append_tensor_checks(checks, model, contract, metadata)
    return tuple(checks)


def _append_tensor_checks(
    checks: list[dict[str, Any]], model: Any, contract: dict[str, Any], metadata: dict[str, str]
) -> None:
    graph = model.graph
    variants = contract["inputs"]["variants"]
    selected_variant = contract.get("input_variant") or metadata.get("input_variant")
    if selected_variant is not None and selected_variant not in variants:
        checks.append({"name": "input_variant", "status": "failed", "expected": tuple(variants), "actual": selected_variant})
        return

    if len(graph.input) != 1:
        checks.append({"name": "input_tensor", "status": "failed", "expected": "exactly one input tensor", "actual": len(graph.input)})
    else:
        actual_input = graph.input[0]
        candidates = ((selected_variant, variants[selected_variant]),) if selected_variant else tuple(variants.items())
        if not any(_input_matches(actual_input, variant, contract["inputs"]["batch"]) for _, variant in candidates):
            checks.append({"name": "input_tensor", "status": "failed", "expected": candidates, "actual": _tensor_description(actual_input)})

    output_by_name = {value.name: value for value in graph.output}
    for name, specification in contract["outputs"].items():
        actual_output = output_by_name.get(name)
        if actual_output is None or not _output_matches(actual_output, specification, contract, name):
            checks.append({"name": f"output_tensor:{name}", "status": "failed", "expected": specification, "actual": _tensor_description(actual_output) if actual_output else None})


def _tensor_description(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    tensor = value.type.tensor_type
    dimensions = []
    for dimension in tensor.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(None)
    return {"name": value.name, "dtype": int(tensor.elem_type), "shape": dimensions}


def _tensor_dtype(value: Any) -> str:
    import onnx

    return {
        "float": "float32",
        "double": "float64",
        "int64": "int64",
        "uint8": "uint8",
    }.get(onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type).casefold(), "unknown")


def _dimension_matches(actual: Any, expected: Any, *, dynamic: bool = False) -> bool:
    if dynamic:
        return not actual.HasField("dim_value")
    if isinstance(expected, int):
        return actual.HasField("dim_value") and int(actual.dim_value) == expected
    return True


def _input_matches(value: Any, variant: dict[str, Any], batch: dict[str, Any]) -> bool:
    tensor = value.type.tensor_type
    expected_shape = variant["shape"]
    if value.name != variant["name"] or _tensor_dtype(value) != variant["dtype"]:
        return False
    if len(tensor.shape.dim) != len(expected_shape):
        return False
    for index, expected in enumerate(expected_shape):
        if not _dimension_matches(tensor.shape.dim[index], expected, dynamic=index == batch["axis"] and batch["dynamic"]):
            return False
    return True


def _output_matches(value: Any, specification: dict[str, Any], contract: dict[str, Any], name: str) -> bool:
    tensor = value.type.tensor_type
    expected_shape = specification["shape"]
    if _tensor_dtype(value) != specification["dtype"] or len(tensor.shape.dim) != len(expected_shape):
        return False
    for index, expected in enumerate(expected_shape):
        if expected == "B":
            if not _dimension_matches(tensor.shape.dim[index], expected, dynamic=True):
                return False
        elif expected == "C" and contract["names"]:
            if not _dimension_matches(tensor.shape.dim[index], len(contract["names"])):
                return False
        elif not _dimension_matches(tensor.shape.dim[index], expected):
            return False
    return True


def embedded_output_paths(output: Path) -> dict[str, Path]:
    output = output.expanduser().resolve()
    if output.suffix.casefold() != ".onnx":
        raise ConfigurationError("ONNX export output must be a .onnx file path; variants are emitted beside it")
    return {
        "bw8": output.with_name(f"{output.stem}-bw8.onnx"),
        "c24": output.with_name(f"{output.stem}-c24.onnx"),
    }


def standardize_detection_core(core: Path, output: Path, *, image_size: int) -> Path:
    """Convert a supported end-to-end detector output to the shared contract."""
    onnx = optional_module("onnx")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    model = onnx.load(str(require_file(core, "ONNX core artifact")))
    graph = model.graph
    if len(graph.output) == 3 and {value.name for value in graph.output} == {
        "boxes", "scores", "class_ids"
    }:
        output_path = output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.checker.check_model(model)
        onnx.save(model, str(output_path))
        return output_path

    if len(graph.output) not in {1, 2}:
        raise ConfigurationError(
            "detection ONNX export must emit boxes/scores/class_ids or an end-to-end [B,Q,6] tensor first"
        )
    raw = graph.output[0]
    raw_type = raw.type.tensor_type
    raw_shape = raw_type.shape.dim
    if len(raw_shape) != 3 or not raw_shape[2].HasField("dim_value") or int(raw_shape[2].dim_value) != 6:
        raise ConfigurationError(
            "unsupported detection ONNX output; expected an end-to-end [B,Q,6] tensor first "
            "or explicit boxes/scores/class_ids outputs"
        )
    if raw_type.elem_type != onnx.TensorProto.FLOAT:
        raise ConfigurationError("end-to-end detection output must use float32 values")

    from onnx import TensorProto, helper

    graph.initializer.extend([
        helper.make_tensor("contract/slice_starts", TensorProto.INT64, [1], [0]),
        helper.make_tensor("contract/boxes_ends", TensorProto.INT64, [1], [4]),
        helper.make_tensor("contract/score_starts", TensorProto.INT64, [1], [4]),
        helper.make_tensor("contract/class_starts", TensorProto.INT64, [1], [5]),
        helper.make_tensor("contract/slice_axes", TensorProto.INT64, [1], [2]),
        helper.make_tensor("contract/score_ends", TensorProto.INT64, [1], [5]),
        helper.make_tensor("contract/class_ends", TensorProto.INT64, [1], [6]),
        helper.make_tensor("contract/squeeze_axes", TensorProto.INT64, [1], [2]),
        helper.make_tensor(
            "contract/box_scale", TensorProto.FLOAT, [1, 1, 4], [float(image_size)] * 4
        ),
    ])
    graph.node.extend([
        helper.make_node(
            "Slice",
            [raw.name, "contract/slice_starts", "contract/boxes_ends", "contract/slice_axes"],
            ["contract/boxes_pixels"],
            name="contract/boxes",
        ),
        helper.make_node(
            "Div",
            ["contract/boxes_pixels", "contract/box_scale"],
            ["boxes"],
            name="contract/boxes_normalized",
        ),
        helper.make_node(
            "Slice",
            [raw.name, "contract/score_starts", "contract/score_ends", "contract/slice_axes"],
            ["contract/scores_column"],
            name="contract/scores_slice",
        ),
        helper.make_node(
            "Squeeze",
            ["contract/scores_column", "contract/squeeze_axes"],
            ["scores"],
            name="contract/scores",
        ),
        helper.make_node(
            "Slice",
            [raw.name, "contract/class_starts", "contract/class_ends", "contract/slice_axes"],
            ["contract/class_column"],
            name="contract/class_slice",
        ),
        helper.make_node(
            "Squeeze",
            ["contract/class_column", "contract/squeeze_axes"],
            ["contract/class_values"],
            name="contract/class_squeeze",
        ),
        helper.make_node(
            "Cast",
            ["contract/class_values"],
            ["class_ids"],
            to=TensorProto.INT64,
            name="contract/class_ids",
        ),
    ])
    batch_dimension = _dimension_value(raw_shape[0], "B")
    query_dimension = _dimension_value(raw_shape[1], "Q")
    del graph.output[:]
    graph.output.extend([
        helper.make_tensor_value_info(
            "boxes", TensorProto.FLOAT, [batch_dimension, query_dimension, 4]
        ),
        helper.make_tensor_value_info(
            "scores", TensorProto.FLOAT, [batch_dimension, query_dimension]
        ),
        helper.make_tensor_value_info(
            "class_ids", TensorProto.INT64, [batch_dimension, query_dimension]
        ),
    ])
    graph.doc_string = "Standardized end-to-end detection output with dynamic batch."
    output_path = output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    return output_path


def _dimension_value(dimension: Any, fallback: str) -> int | str | None:
    if dimension.HasField("dim_value"):
        return int(dimension.dim_value)
    if dimension.HasField("dim_param"):
        return dimension.dim_param
    return fallback


def wrap_embedded_variants(
    core: Path,
    outputs: dict[str, Path],
    *,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    apply_softmax: bool = False,
    output_names: tuple[str, ...] | None = None,
) -> tuple[Path, ...]:
    """Wrap a float RGB NCHW ONNX core with dynamic-batch BW8 and C24 inputs."""
    onnx = optional_module("onnx")
    np = optional_module("numpy")
    core_model = require_file(core, "ONNX core artifact")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("embedded preprocessing requires three mean and std values")
    for variant, output in outputs.items():
        if variant not in {"bw8", "c24"}:
            raise ValueError(f"unsupported embedded input variant: {variant}")
        model = onnx.load(str(core_model))
        if len(model.graph.input) != 1:
            raise ValueError("ONNX core must have exactly one input")
        if not any(item.domain == "" and item.version >= 18 for item in model.opset_import):
            raise ValueError("embedded preprocessing requires ONNX opset 18 or newer")
        core_graph = onnx.compose.add_prefix(model, "core/")
        core_input = core_graph.graph.input[0].name
        del core_graph.graph.input[:]
        original_nodes = list(core_graph.graph.node)
        del core_graph.graph.node[:]
        preprocessing = _embedded_preprocessing_nodes(
            onnx,
            np,
            core_graph,
            core_input,
            variant,
            image_size,
            mean,
            std,
        )
        postprocessing: list[Any] = []
        if output_names is not None and len(output_names) != len(core_graph.graph.output):
            raise ValueError("output_names must match the ONNX core output count")
        if apply_softmax and len(core_graph.graph.output) != 1:
            raise ValueError("classification core must have exactly one output for softmax wrapping")
        for output_index, output_value in enumerate(core_graph.graph.output):
            original_output = output_value.name
            public_output = output_names[output_index] if output_names is not None else original_output.removeprefix("core/")
            if apply_softmax:
                public_output = "probabilities"
                postprocessing.append(onnx.helper.make_node("Softmax", [original_output], [public_output], axis=1, name="contract/softmax"))
            else:
                postprocessing.append(onnx.helper.make_node("Identity", [original_output], [public_output], name=f"contract/{public_output}"))
            output_value.name = public_output
        core_graph.graph.node.extend(preprocessing + original_nodes + postprocessing)
        core_graph.graph.doc_string = f"Embedded {variant} preprocessing with dynamic batch."
        output_path = output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.checker.check_model(core_graph)
        onnx.save(core_graph, str(output_path))
    return tuple(outputs[variant].expanduser().resolve() for variant in ("bw8", "c24"))


def _embedded_preprocessing_nodes(
    onnx: Any,
    np: Any,
    model: Any,
    core_input: str,
    variant: str,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> list[Any]:
    from onnx import TensorProto, helper, numpy_helper

    if variant == "bw8":
        input_name = "images_bw8_uint8_nchw"
        model.graph.input.append(helper.make_tensor_value_info(input_name, TensorProto.UINT8, ["B", 1, "H", "W"]))
        resized_source = input_name
        input_channels = 1
        nodes: list[Any] = []
    else:
        input_name = "images_c24_uint8_nhwc_bgr"
        model.graph.input.append(helper.make_tensor_value_info(input_name, TensorProto.UINT8, ["B", "H", "W", 3]))
        nodes = [
            helper.make_node("Transpose", [input_name], ["preprocess/bgr_nchw"], perm=[0, 3, 1, 2], name="preprocess/transpose"),
            helper.make_node(
                "Gather",
                ["preprocess/bgr_nchw", "preprocess/bgr_to_rgb_indices"],
                ["preprocess/rgb_nchw"],
                axis=1,
                name="preprocess/bgr_to_rgb",
            ),
        ]
        model.graph.initializer.append(numpy_helper.from_array(np.asarray([2, 1, 0], dtype=np.int64), "preprocess/bgr_to_rgb_indices"))
        resized_source = "preprocess/rgb_nchw"
        input_channels = 3

    sizes_name, size_nodes = _add_dynamic_sizes(model, np, helper, resized_source, input_channels, image_size, "preprocess/resize_sizes")
    nodes.extend(size_nodes)
    nodes.extend([
        helper.make_node(
            "Resize",
            [resized_source, "", "", sizes_name],
            ["preprocess/resized_uint8"],
            mode="linear",
            coordinate_transformation_mode="half_pixel",
            antialias=1,
            name="preprocess/resize",
        ),
        helper.make_node("Cast", ["preprocess/resized_uint8"], ["preprocess/resized_float"], to=TensorProto.FLOAT, name="preprocess/cast"),
        helper.make_node(
            "Div",
            ["preprocess/resized_float", "preprocess/pixel_scale"],
            ["preprocess/scaled"],
            name="preprocess/scale",
        ),
    ])
    model.graph.initializer.append(numpy_helper.from_array(np.asarray([[[[255.0]]]], dtype=np.float32), "preprocess/pixel_scale"))

    if variant == "bw8":
        rgb_sizes_name, rgb_size_nodes = _add_dynamic_sizes(model, np, helper, "preprocess/resized_float", 3, image_size, "preprocess/rgb_sizes")
        nodes.extend(rgb_size_nodes)
        nodes.append(helper.make_node("Expand", ["preprocess/scaled", rgb_sizes_name], ["preprocess/rgb"], name="preprocess/gray_to_rgb"))
        normalized_source = "preprocess/rgb"
    else:
        normalized_source = "preprocess/scaled"

    model.graph.initializer.extend([
        numpy_helper.from_array(np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1), "preprocess/mean"),
        numpy_helper.from_array(np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1), "preprocess/std"),
    ])
    nodes.extend([
        helper.make_node("Sub", [normalized_source, "preprocess/mean"], ["preprocess/centered"], name="preprocess/mean"),
        helper.make_node("Div", ["preprocess/centered", "preprocess/std"], [core_input], name="preprocess/std"),
    ])
    return nodes


def _add_dynamic_sizes(model: Any, np: Any, helper: Any, source: str, channels: int, image_size: int, name: str) -> tuple[str, list[Any]]:
    shape_name = f"{name}/shape"
    batch_name = f"{name}/batch"
    tail_name = f"{name}/tail"
    index_name = f"{name}/batch_index"
    sizes_name = name
    model.graph.initializer.extend([
        helper.make_tensor(index_name, 7, [1], [0]),
        helper.make_tensor(tail_name, 7, [3], [channels, image_size, image_size]),
    ])
    return sizes_name, [
        helper.make_node("Shape", [source], [shape_name], name=f"{name}/shape_node"),
        helper.make_node("Gather", [shape_name, index_name], [batch_name], axis=0, name=f"{name}/batch_node"),
        helper.make_node("Concat", [batch_name, tail_name], [sizes_name], axis=0, name=f"{name}/concat"),
    ]


def set_onnx_metadata(path: Path, values: dict[str, Any]) -> None:
    onnx = optional_module("onnx")
    model = onnx.load(str(path))
    existing = {item.key: item for item in model.metadata_props}
    for key, value in values.items():
        if key in existing:
            existing[key].value = json.dumps(value, default=str) if not isinstance(value, str) else value
        else:
            item = model.metadata_props.add()
            item.key = key
            item.value = json.dumps(value, default=str) if not isinstance(value, str) else value
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def artifacts_for(paths: list[tuple[Path, str]]) -> tuple[ArtifactRef, ...]:
    return tuple(artifact(path, kind) for path, kind in paths if path.exists())
