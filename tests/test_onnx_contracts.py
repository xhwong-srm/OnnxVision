from __future__ import annotations

import pytest

from vision_workflows.backends.common import (
    CLASSIFICATION_CONTRACT_NAME,
    CONTRACT_VERSION,
    DETECTION_CONTRACT_NAME,
    classification_contract,
    detection_contract,
    metadata_for_contract,
    standardize_detection_core,
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
    assert contract["inputs"]["batch"] == {"axis": 0, "minimum": 1, "dynamic": True}
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
        "scores": {"dtype": "float32", "shape": ["B", "Q"]},
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

    standardize_detection_core(source, target, image_size=8)
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
