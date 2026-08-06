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


def batch_contract(batch_size: int | None = None) -> dict[str, Any]:
    if batch_size is None:
        return {"axis": 0, "mode": "dynamic", "minimum": 1}
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return {"axis": 0, "mode": "fixed", "size": batch_size}


def embedded_input_contract(batch_size: int | None = None) -> dict[str, Any]:
    """Return the shared raw-image input contract used by both ONNX tasks."""
    return {
        "batch": batch_contract(batch_size),
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
    batch_size: int | None = None,
) -> dict[str, Any]:
    contract = {
        "name": DETECTION_CONTRACT_NAME,
        "version": CONTRACT_VERSION,
        "task": "object_detection",
        "inputs": embedded_input_contract(batch_size),
        "outputs": {
            "boxes": {
                "dtype": "float32",
                "shape": ["B", "Q", 4],
                "coordinate_format": "xyxy",
                "coordinate_space": "normalized_0_1",
            },
            "scores": {
                "dtype": "float32",
                "shape": ["B", "Q"],
                "range": [0.0, 1.0],
                "padding_value": 0.0,
                "valid_when": "score_gt_0",
            },
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
    names: list[str] | tuple[str, ...], *, input_variant: str | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    contract = {
        "name": CLASSIFICATION_CONTRACT_NAME,
        "version": CONTRACT_VERSION,
        "task": "classification",
        "inputs": embedded_input_contract(batch_size),
        "outputs": {
            "probabilities": {
                "dtype": "float32",
                "shape": ["B", "C"],
                "semantics": "categorical_probabilities",
                "range": [0.0, 1.0],
                "row_sum": 1.0,
            }
        },
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
    if not is_compatible_contract_version(version):
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


def is_compatible_contract_version(version: str) -> bool:
    if not CONTRACT_VERSION_PATTERN.fullmatch(version):
        return False
    major, _, _ = (int(item) for item in version.split("."))
    return major == int(CONTRACT_VERSION.split(".", 1)[0])


def _contains_contract(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains_contract(actual[key], value)
            for key, value in expected.items()
        )
    return actual == expected


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
    contract = _resolved_validation_contract(contract, metadata)
    expected = metadata_for_contract(contract)
    actual_variant = metadata.get("input_variant")
    expected_variant = expected.get("input_variant")
    if actual_variant not in {"bw8", "c24"}:
        checks.append({
            "name": "input_variant",
            "status": "failed",
            "expected": expected_variant or ("bw8", "c24"),
            "actual": actual_variant,
        })
    elif expected_variant is not None and actual_variant != expected_variant:
        checks.append({"name": "input_variant", "status": "failed", "expected": expected_variant, "actual": actual_variant})
    for key in ("vision_task", "contract_name"):
        if key not in expected:
            continue
        if metadata.get(key) != expected[key]:
            checks.append({"name": key, "status": "failed", "expected": expected[key], "actual": metadata.get(key)})

    for key in ("inputs", "outputs"):
        try:
            actual_value = json.loads(metadata.get(key, ""))
        except json.JSONDecodeError:
            actual_value = None
        if not _contains_contract(actual_value, expected[key]):
            checks.append({"name": key, "status": "failed", "expected": expected[key], "actual": actual_value})

    version = metadata.get("contract_version", "")
    if not is_compatible_contract_version(version):
        checks.append({"name": "contract_version", "status": "failed", "expected": "compatible 2.x.y", "actual": version})

    if contract["task"] == "object_detection":
        nms_required = metadata.get("nms_required")
        expected_nms = str(bool(contract["nms_required"])).lower()
        if nms_required != expected_nms:
            checks.append({"name": "nms_required", "status": "failed", "expected": expected_nms, "actual": nms_required})

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
    if not any(check.get("status") == "failed" for check in checks):
        _append_runtime_checks(checks, path, contract)
    return tuple(checks)


def _resolved_validation_contract(
    requested: dict[str, Any], metadata: dict[str, str]
) -> dict[str, Any]:
    if requested.get("names"):
        return requested
    try:
        names_mapping = json.loads(metadata.get("names", ""))
        inputs = json.loads(metadata.get("inputs", ""))
    except json.JSONDecodeError:
        return requested
    if not isinstance(names_mapping, dict) or not isinstance(inputs, dict):
        return requested
    try:
        names = [names_mapping[str(index)] for index in range(len(names_mapping))]
    except (KeyError, TypeError):
        return requested
    batch = inputs.get("batch", {})
    if batch.get("mode") == "fixed" and isinstance(batch.get("size"), int):
        batch_size = int(batch["size"])
    else:
        batch_size = None
    input_variant = metadata.get("input_variant")
    if requested.get("task") == "classification":
        return classification_contract(
            names, input_variant=input_variant, batch_size=batch_size
        )
    nms_value = metadata.get("nms_required", "false").casefold() == "true"
    return detection_contract(
        names,
        nms_required=nms_value,
        input_variant=input_variant,
        batch_size=batch_size,
    )


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
    if set(output_by_name) != set(contract["outputs"]):
        checks.append({
            "name": "output_tensors",
            "status": "failed",
            "expected": tuple(contract["outputs"]),
            "actual": tuple(output_by_name),
        })
    for name, specification in contract["outputs"].items():
        actual_output = output_by_name.get(name)
        if actual_output is None or not _output_matches(actual_output, specification, contract, name):
            checks.append({"name": f"output_tensor:{name}", "status": "failed", "expected": specification, "actual": _tensor_description(actual_output) if actual_output else None})


def _append_runtime_checks(
    checks: list[dict[str, Any]], path: Path, contract: dict[str, Any]
) -> None:
    try:
        ort = optional_module("onnxruntime")
        np = optional_module("numpy")
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        input_info = session.get_inputs()[0]
        batch = contract["inputs"]["batch"]
        batches = (1, 2) if batch["mode"] == "dynamic" else (int(batch["size"]),)
        variant = contract["input_variant"]
        for batch_size in batches:
            shape = (
                (batch_size, 1, 8, 8)
                if variant == "bw8"
                else (batch_size, 8, 8, 3)
            )
            value = np.zeros(shape, dtype=np.uint8)
            outputs = session.run(None, {input_info.name: value})
            _validate_runtime_outputs(outputs, contract, batch_size, np)
        checks.append({"name": "onnxruntime_contract", "status": "passed"})
    except Exception as error:
        checks.append({
            "name": "onnxruntime_contract",
            "status": "failed",
            "actual": str(error),
        })


def _validate_runtime_outputs(
    outputs: list[Any], contract: dict[str, Any], batch_size: int, np: Any
) -> None:
    if contract["task"] == "classification":
        probabilities = outputs[0]
        if probabilities.shape != (batch_size, len(contract["names"])):
            raise ValueError(f"classification output shape mismatch: {probabilities.shape}")
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("classification probabilities must be finite and within [0,1]")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=0.001, rtol=0.0):
            raise ValueError("classification probability rows must sum to one")
        return

    boxes, scores, class_ids = outputs
    if boxes.ndim != 3 or scores.ndim != 2 or class_ids.ndim != 2:
        raise ValueError("detection output ranks are invalid")
    if boxes.shape[:2] != scores.shape or class_ids.shape != scores.shape or boxes.shape[0] != batch_size:
        raise ValueError("detection output shapes do not agree")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("detection scores must be finite and within [0,1]")
    valid = scores > 0
    valid_boxes = boxes[valid]
    valid_ids = class_ids[valid]
    if valid_boxes.size and (
        not np.isfinite(valid_boxes).all()
        or (valid_boxes < 0).any()
        or (valid_boxes > 1).any()
        or (valid_boxes[:, 0] > valid_boxes[:, 2]).any()
        or (valid_boxes[:, 1] > valid_boxes[:, 3]).any()
    ):
        raise ValueError("valid detection boxes must be ordered normalized XYXY values")
    if valid_ids.size and ((valid_ids < 0).any() or (valid_ids >= len(contract["names"])).any()):
        raise ValueError("valid detection class IDs are outside the class mapping")


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


def _batch_dimension_matches(actual: Any, batch: dict[str, Any]) -> bool:
    mode = batch.get("mode")
    if mode == "dynamic":
        return not actual.HasField("dim_value") and batch.get("minimum") == 1
    if mode == "fixed":
        size = batch.get("size")
        return isinstance(size, int) and size > 0 and _dimension_matches(actual, size)
    return False


def _input_matches(value: Any, variant: dict[str, Any], batch: dict[str, Any]) -> bool:
    tensor = value.type.tensor_type
    expected_shape = variant["shape"]
    if value.name != variant["name"] or _tensor_dtype(value) != variant["dtype"]:
        return False
    if len(tensor.shape.dim) != len(expected_shape):
        return False
    for index, expected in enumerate(expected_shape):
        if index == batch.get("axis"):
            if not _batch_dimension_matches(tensor.shape.dim[index], batch):
                return False
        elif not _dimension_matches(tensor.shape.dim[index], expected):
            return False
    return True


def _output_matches(value: Any, specification: dict[str, Any], contract: dict[str, Any], name: str) -> bool:
    tensor = value.type.tensor_type
    expected_shape = specification["shape"]
    if _tensor_dtype(value) != specification["dtype"] or len(tensor.shape.dim) != len(expected_shape):
        return False
    for index, expected in enumerate(expected_shape):
        if expected == "B":
            if not _batch_dimension_matches(tensor.shape.dim[index], contract["inputs"]["batch"]):
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


def standardize_detection_core(
    core: Path,
    output: Path,
    *,
    image_size: int,
    source_box_format: str,
    source_box_space: str,
) -> Path:
    """Convert declared provider outputs to normalized XYXY contract tensors."""
    onnx = optional_module("onnx")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if source_box_format not in {"xyxy", "cxcywh"}:
        raise ValueError("source_box_format must be xyxy or cxcywh")
    if source_box_space not in {"pixels", "normalized_0_1"}:
        raise ValueError("source_box_space must be pixels or normalized_0_1")
    model = onnx.load(str(require_file(core, "ONNX core artifact")))
    graph = model.graph
    from onnx import TensorProto, helper

    if len(graph.output) == 3 and {value.name for value in graph.output} == {
        "boxes", "scores", "class_ids"
    }:
        output_by_name = {value.name: value for value in graph.output}
        source_boxes = output_by_name["boxes"]
        source_scores = output_by_name["scores"]
        source_class_ids = output_by_name["class_ids"]
        _rename_graph_value(graph, "boxes", "contract/source_boxes")
        boxes_name = "contract/source_boxes"
        scores_name = source_scores.name
        class_ids_name = source_class_ids.name
        batch_dimension = _dimension_value(source_boxes.type.tensor_type.shape.dim[0], "B")
        query_dimension = _dimension_value(source_boxes.type.tensor_type.shape.dim[1], "Q")
    else:
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
        graph.initializer.extend([
            helper.make_tensor("contract/slice_starts", TensorProto.INT64, [1], [0]),
            helper.make_tensor("contract/boxes_ends", TensorProto.INT64, [1], [4]),
            helper.make_tensor("contract/score_starts", TensorProto.INT64, [1], [4]),
            helper.make_tensor("contract/class_starts", TensorProto.INT64, [1], [5]),
            helper.make_tensor("contract/slice_axes", TensorProto.INT64, [1], [2]),
            helper.make_tensor("contract/score_ends", TensorProto.INT64, [1], [5]),
            helper.make_tensor("contract/class_ends", TensorProto.INT64, [1], [6]),
            helper.make_tensor("contract/squeeze_axes", TensorProto.INT64, [1], [2]),
        ])
        boxes_name = "contract/source_boxes"
        scores_name = "scores"
        class_ids_name = "class_ids"
        graph.node.extend([
            helper.make_node(
                "Slice",
                [raw.name, "contract/slice_starts", "contract/boxes_ends", "contract/slice_axes"],
                [boxes_name],
                name="contract/boxes_slice",
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
            [scores_name],
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
            [class_ids_name],
            to=TensorProto.INT64,
            name="contract/class_ids",
            ),
        ])
        batch_dimension = _dimension_value(raw_shape[0], "B")
        query_dimension = _dimension_value(raw_shape[1], "Q")

    normalized_boxes, box_nodes = _box_conversion_nodes(
        helper,
        boxes_name,
        source_box_format=source_box_format,
        source_box_space=source_box_space,
    )
    graph.initializer.extend([
        helper.make_tensor("contract/half", TensorProto.FLOAT, [1], [0.5]),
        helper.make_tensor(
            "contract/box_scale", TensorProto.FLOAT, [1, 1, 4], [float(image_size)] * 4
        ),
        helper.make_tensor("contract/clip_min", TensorProto.FLOAT, [], [0.0]),
        helper.make_tensor("contract/clip_max", TensorProto.FLOAT, [], [1.0]),
    ])
    graph.node.extend(box_nodes)
    graph.node.append(helper.make_node(
        "Clip",
        [normalized_boxes, "contract/clip_min", "contract/clip_max"],
        ["boxes"],
        name="contract/boxes_clip",
    ))
    del graph.output[:]
    graph.output.extend([
        helper.make_tensor_value_info(
            "boxes", TensorProto.FLOAT, [batch_dimension, query_dimension, 4]
        ),
        helper.make_tensor_value_info(
            scores_name, TensorProto.FLOAT, [batch_dimension, query_dimension]
        ),
        helper.make_tensor_value_info(
            class_ids_name, TensorProto.INT64, [batch_dimension, query_dimension]
        ),
    ])
    graph.doc_string = "Standardized end-to-end detection outputs."
    output_path = output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    return output_path


def _rename_graph_value(graph: Any, old: str, new: str) -> None:
    for node in graph.node:
        for values in (node.input, node.output):
            for index, value in enumerate(values):
                if value == old:
                    values[index] = new
    for collection in (graph.input, graph.output, graph.value_info):
        for value in collection:
            if value.name == old:
                value.name = new


def _box_conversion_nodes(
    helper: Any,
    boxes: str,
    *,
    source_box_format: str,
    source_box_space: str,
) -> tuple[str, list[Any]]:
    nodes: list[Any] = []
    converted = boxes
    if source_box_format == "cxcywh":
        nodes.extend([
            helper.make_node("Split", [converted], [
                "contract/cx", "contract/cy", "contract/width", "contract/height"
            ], axis=2, num_outputs=4, name="contract/split_cxcywh"),
            helper.make_node("Mul", ["contract/width", "contract/half"], ["contract/half_width"], name="contract/half_width"),
            helper.make_node("Mul", ["contract/height", "contract/half"], ["contract/half_height"], name="contract/half_height"),
            helper.make_node("Sub", ["contract/cx", "contract/half_width"], ["contract/x1"], name="contract/x1"),
            helper.make_node("Sub", ["contract/cy", "contract/half_height"], ["contract/y1"], name="contract/y1"),
            helper.make_node("Add", ["contract/cx", "contract/half_width"], ["contract/x2"], name="contract/x2"),
            helper.make_node("Add", ["contract/cy", "contract/half_height"], ["contract/y2"], name="contract/y2"),
            helper.make_node("Concat", ["contract/x1", "contract/y1", "contract/x2", "contract/y2"], ["contract/boxes_xyxy"], axis=2, name="contract/boxes_xyxy"),
        ])
        converted = "contract/boxes_xyxy"
    if source_box_space == "pixels":
        nodes.append(helper.make_node(
            "Div", [converted, "contract/box_scale"], ["contract/boxes_normalized"], name="contract/boxes_normalized"
        ))
        converted = "contract/boxes_normalized"
    return converted, nodes


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
    pixel_scale: float = 255.0,
    resize_antialias: bool = True,
    batch_size: int | None = None,
    apply_softmax: bool = False,
    output_names: tuple[str, ...] | None = None,
) -> tuple[Path, ...]:
    """Wrap a float RGB NCHW ONNX core with BW8 and C24 inputs.

    ``pixel_scale=255`` is the usual uint8-to-[0,1] path. Models such as
    PicoDet that consume ImageNet-normalized pixel values can set it to 1 and
    provide pixel-space ``mean``/``std`` values instead.
    """
    onnx = optional_module("onnx")
    np = optional_module("numpy")
    core_model = require_file(core, "ONNX core artifact")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("embedded preprocessing requires three mean and std values")
    if pixel_scale <= 0:
        raise ValueError("embedded preprocessing pixel_scale must be positive")
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
            pixel_scale,
            resize_antialias,
            batch_size,
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
        mode = "dynamic" if batch_size is None else f"fixed batch {batch_size}"
        core_graph.graph.doc_string = f"Embedded {variant} preprocessing with {mode}."
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
    pixel_scale: float,
    resize_antialias: bool,
    batch_size: int | None,
) -> list[Any]:
    from onnx import TensorProto, helper, numpy_helper

    if variant == "bw8":
        input_name = "images_bw8_uint8_nchw"
        batch_dimension = "B" if batch_size is None else batch_size
        model.graph.input.append(helper.make_tensor_value_info(input_name, TensorProto.UINT8, [batch_dimension, 1, "H", "W"]))
        resized_source = input_name
        input_channels = 1
        nodes: list[Any] = []
    else:
        input_name = "images_c24_uint8_nhwc_bgr"
        batch_dimension = "B" if batch_size is None else batch_size
        model.graph.input.append(helper.make_tensor_value_info(input_name, TensorProto.UINT8, [batch_dimension, "H", "W", 3]))
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
            antialias=int(resize_antialias),
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
    model.graph.initializer.append(numpy_helper.from_array(np.asarray([[[[pixel_scale]]]], dtype=np.float32), "preprocess/pixel_scale"))

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
