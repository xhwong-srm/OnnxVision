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
CONTRACT_VERSION = "1.0.0"


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


def detection_contract(names: list[str] | tuple[str, ...], *, nms_required: bool = False) -> dict[str, Any]:
    return {
        "name": DETECTION_CONTRACT_NAME,
        "version": CONTRACT_VERSION,
        "task": "object_detection",
        "inputs": {"float": {"dtype": "float32", "layout": "NCHW"}},
        "outputs": {"boxes": "float32[N,4]", "scores": "float32[N]", "class_ids": "int64[N]"},
        "names": {str(index): name for index, name in enumerate(names)},
        "nms_required": nms_required,
    }


def classification_contract(names: list[str] | tuple[str, ...]) -> dict[str, Any]:
    return {
        "name": CLASSIFICATION_CONTRACT_NAME,
        "version": CONTRACT_VERSION,
        "task": "classification",
        "inputs": {"images": {"dtype": "float32", "layout": "NCHW"}},
        "outputs": {"probabilities": "float32[N,C]"},
        "names": {str(index): name for index, name in enumerate(names)},
    }


def metadata_for_contract(contract: dict[str, Any]) -> dict[str, Any]:
    version = str(contract["version"])
    if not CONTRACT_VERSION_PATTERN.fullmatch(version):
        raise ValueError(f"contract version must use major.minor.micro format: {version}")

    metadata = {
        "vision_task": contract["task"],
        "contract_name": contract["name"],
        "contract_version": version,
        "names": contract["names"],
    }
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
    for key in ("vision_task", "contract_name", "contract_version"):
        if metadata.get(key) != expected[key]:
            checks.append({"name": key, "status": "failed", "expected": expected[key], "actual": metadata.get(key)})

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
    return tuple(checks)


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
