from __future__ import annotations

import pytest

from vision_workflows.backends.common import (
    CLASSIFICATION_CONTRACT_NAME,
    CONTRACT_VERSION,
    DETECTION_CONTRACT_NAME,
    classification_contract,
    detection_contract,
    is_compatible_contract_version,
    metadata_for_contract,
    embedded_output_paths,
    set_onnx_metadata,
    standardize_detection_core,
    validate_onnx,
    wrap_embedded_variants,
)


def test_classification_metadata_uses_explicit_task_and_semver_contract() -> None:
    assert CLASSIFICATION_CONTRACT_NAME == "onnx-vision-classification"
    contract = classification_contract(("normal", "flipped"))

    assert contract["name"] == CLASSIFICATION_CONTRACT_NAME
    assert contract["version"] == CONTRACT_VERSION == "2.0.0"
    metadata = metadata_for_contract(contract)
    assert metadata == {
        "vision_task": "classification",
        "contract_name": CLASSIFICATION_CONTRACT_NAME,
        "contract_version": "2.0.0",
        "names": {"0": "normal", "1": "flipped"},
        "inputs": contract["inputs"],
        "outputs": contract["outputs"],
    }
    assert contract["inputs"]["batch"] == {"axis": 0, "mode": "dynamic", "minimum": 1}
    assert contract["inputs"]["variants"]["bw8"]["shape"] == ["B", 1, "H", "W"]
    assert contract["inputs"]["variants"]["c24"]["shape"] == ["B", "H", "W", 3]
    assert contract["outputs"]["probabilities"]["shape"] == ["B", "C"]


def test_detection_metadata_includes_nms_contract_fields() -> None:
    assert DETECTION_CONTRACT_NAME == "onnx-vision-object-detection"
    contract = detection_contract(("seal",), nms_required=True)

    assert contract["name"] == DETECTION_CONTRACT_NAME
    assert contract["task"] == "object_detection"
    assert metadata_for_contract(contract) == {
        "vision_task": "object_detection",
        "contract_name": DETECTION_CONTRACT_NAME,
        "contract_version": "2.0.0",
        "names": {"0": "seal"},
        "inputs": contract["inputs"],
        "outputs": contract["outputs"],
        "nms_required": True,
    }
    assert contract["outputs"] == {
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
    }


def test_variant_contract_identifies_the_single_artifact_input() -> None:
    contract = classification_contract(("normal",), input_variant="c24")

    assert contract["input_variant"] == "c24"
    assert metadata_for_contract(contract)["input_variant"] == "c24"


def test_metadata_rejects_non_semver_contract_versions() -> None:
    contract = classification_contract(("normal",))
    contract["version"] = "classification-v1"

    with pytest.raises(ValueError, match="major.minor.micro"):
        metadata_for_contract(contract)


def test_v2_semver_accepts_minor_and_micro_but_rejects_other_majors() -> None:
    assert is_compatible_contract_version("2.0.0")
    assert is_compatible_contract_version("2.99.123")
    assert not is_compatible_contract_version("1.9.9")
    assert not is_compatible_contract_version("3.0.0")
    assert not is_compatible_contract_version("2.0")


def test_fixed_batch_contract_is_explicit() -> None:
    contract = classification_contract(("normal",), batch_size=4)
    assert contract["inputs"]["batch"] == {"axis": 0, "mode": "fixed", "size": 4}


def test_standardize_detection_core_splits_end_to_end_batch_output(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Identity", ["images"], ["output0"]),
            onnx.helper.make_node("Identity", ["images"], ["raw"]),
        ],
        "end_to_end",
        [onnx.helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, ["B", 2, 6])],
        [
            onnx.helper.make_tensor_value_info("output0", onnx.TensorProto.FLOAT, ["B", 2, 6]),
            onnx.helper.make_tensor_value_info("raw", onnx.TensorProto.FLOAT, ["B", 2, 6]),
        ],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
    source = tmp_path / "raw.onnx"
    target = tmp_path / "standardized.onnx"
    onnx.save(model, source)

    standardize_detection_core(
        source,
        target,
        image_size=8,
        source_box_format="xyxy",
        source_box_space="pixels",
    )
    standardized = onnx.load(target)

    assert [value.name for value in standardized.graph.output] == ["boxes", "scores", "class_ids"]
    assert [len(value.type.tensor_type.shape.dim) for value in standardized.graph.output] == [3, 2, 2]

    ort = pytest.importorskip("onnxruntime")
    np = pytest.importorskip("numpy")
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    values = np.asarray([[[0.0, 0.0, 8.0, 8.0, 0.75, 1.0], [1.0, 2.0, 4.0, 6.0, 0.25, 0.0]]], dtype=np.float32)
    boxes, scores, class_ids = session.run(None, {"images": values})
    assert boxes.tolist() == [[[0.0, 0.0, 1.0, 1.0], [0.125, 0.25, 0.5, 0.75]]]
    assert scores.tolist() == [[0.75, 0.25]]
    assert class_ids.tolist() == [[1, 0]]


def test_standardize_detection_core_converts_normalized_cxcywh(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["images"], ["output0"])],
        "end_to_end",
        [onnx.helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 1, 6])],
        [onnx.helper.make_tensor_value_info("output0", onnx.TensorProto.FLOAT, [1, 1, 6])],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
    source = tmp_path / "raw-cxcywh.onnx"
    target = tmp_path / "standardized-cxcywh.onnx"
    onnx.save(model, source)

    standardize_detection_core(
        source,
        target,
        image_size=8,
        source_box_format="cxcywh",
        source_box_space="normalized_0_1",
    )

    ort = pytest.importorskip("onnxruntime")
    np = pytest.importorskip("numpy")
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    values = np.asarray([[[0.5, 0.5, 0.5, 0.25, 0.8, 0.0]]], dtype=np.float32)
    boxes, _, _ = session.run(None, {"images": values})
    assert boxes.tolist() == [[[0.25, 0.375, 0.75, 0.625]]]


def test_standardize_detection_core_normalizes_explicit_box_output(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Identity", ["source_boxes"], ["boxes"]),
            onnx.helper.make_node("Identity", ["source_scores"], ["scores"]),
            onnx.helper.make_node("Identity", ["source_ids"], ["class_ids"]),
        ],
        "explicit_detection",
        [
            onnx.helper.make_tensor_value_info("source_boxes", onnx.TensorProto.FLOAT, [1, 1, 4]),
            onnx.helper.make_tensor_value_info("source_scores", onnx.TensorProto.FLOAT, [1, 1]),
            onnx.helper.make_tensor_value_info("source_ids", onnx.TensorProto.INT64, [1, 1]),
        ],
        [
            onnx.helper.make_tensor_value_info("boxes", onnx.TensorProto.FLOAT, [1, 1, 4]),
            onnx.helper.make_tensor_value_info("scores", onnx.TensorProto.FLOAT, [1, 1]),
            onnx.helper.make_tensor_value_info("class_ids", onnx.TensorProto.INT64, [1, 1]),
        ],
    )
    source = tmp_path / "explicit.onnx"
    target = tmp_path / "explicit-standardized.onnx"
    onnx.save(onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)]), source)
    standardize_detection_core(
        source, target, image_size=8,
        source_box_format="xyxy", source_box_space="pixels",
    )

    ort = pytest.importorskip("onnxruntime")
    np = pytest.importorskip("numpy")
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    boxes, scores, class_ids = session.run(None, {
        "source_boxes": np.asarray([[[1.0, 2.0, 4.0, 6.0]]], dtype=np.float32),
        "source_scores": np.asarray([[0.75]], dtype=np.float32),
        "source_ids": np.asarray([[0]], dtype=np.int64),
    })
    assert boxes.tolist() == [[[0.125, 0.25, 0.5, 0.75]]]
    assert scores.tolist() == [[0.75]]
    assert class_ids.tolist() == [[0]]


@pytest.mark.parametrize("batch_size", [None, 2])
def test_classification_contract_validates_dynamic_and_fixed_runtime(
    tmp_path, batch_size: int | None
) -> None:
    onnx = pytest.importorskip("onnx")
    np = pytest.importorskip("numpy")
    batch_dimension = "B" if batch_size is None else batch_size
    weights = onnx.numpy_helper.from_array(
        np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32),
        "weights",
    )
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("GlobalAveragePool", ["images"], ["pooled"]),
            onnx.helper.make_node("Flatten", ["pooled"], ["features"], axis=1),
            onnx.helper.make_node("MatMul", ["features", "weights"], ["logits"]),
        ],
        "classification",
        [onnx.helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [batch_dimension, 3, 8, 8])],
        [onnx.helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [batch_dimension, 2])],
        [weights],
    )
    core = tmp_path / "core.onnx"
    onnx.save(onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)]), core)
    outputs = embedded_output_paths(tmp_path / "model.onnx")
    wrap_embedded_variants(
        core,
        outputs,
        image_size=8,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        batch_size=batch_size,
        apply_softmax=True,
        output_names=("probabilities",),
    )
    for variant, path in outputs.items():
        contract = classification_contract(
            ("normal", "flipped"), input_variant=variant, batch_size=batch_size
        )
        set_onnx_metadata(path, metadata_for_contract(contract))
        checks = validate_onnx(path, contract)
        assert not [check for check in checks if check["status"] == "failed"]
        assert any(check["name"] == "onnxruntime_contract" for check in checks)
