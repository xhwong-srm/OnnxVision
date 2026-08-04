from __future__ import annotations

import pytest

from vision_workflows.backends.common import (
    CLASSIFICATION_CONTRACT_NAME,
    CONTRACT_VERSION,
    DETECTION_CONTRACT_NAME,
    classification_contract,
    detection_contract,
    metadata_for_contract,
)


def test_classification_metadata_uses_explicit_task_and_semver_contract() -> None:
    assert CLASSIFICATION_CONTRACT_NAME == "onnx-vision-classification"
    contract = classification_contract(("normal", "flipped"))

    assert contract["name"] == CLASSIFICATION_CONTRACT_NAME
    assert contract["version"] == CONTRACT_VERSION == "1.0.0"
    assert metadata_for_contract(contract) == {
        "vision_task": "classification",
        "contract_name": CLASSIFICATION_CONTRACT_NAME,
        "contract_version": "1.0.0",
        "names": {"0": "normal", "1": "flipped"},
    }


def test_detection_metadata_includes_nms_contract_fields() -> None:
    assert DETECTION_CONTRACT_NAME == "onnx-vision-object-detection"
    contract = detection_contract(("seal",), nms_required=True)

    assert contract["name"] == DETECTION_CONTRACT_NAME
    assert contract["task"] == "object_detection"
    assert metadata_for_contract(contract) == {
        "vision_task": "object_detection",
        "contract_name": DETECTION_CONTRACT_NAME,
        "contract_version": "1.0.0",
        "names": {"0": "seal"},
        "nms_required": True,
    }


def test_metadata_rejects_non_semver_contract_versions() -> None:
    contract = classification_contract(("normal",))
    contract["version"] = "classification-v1"

    with pytest.raises(ValueError, match="major.minor.micro"):
        metadata_for_contract(contract)
