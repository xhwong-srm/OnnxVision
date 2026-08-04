from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..domain.errors import BackendUnavailableError, ConfigurationError
from ..domain.results import ArtifactRef
from ..workflows.context import optional_import
from ..workflows.runs import artifact


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
        "version": "onnx-vision-detection-v1",
        "task": "detection",
        "inputs": {"float": {"dtype": "float32", "layout": "NCHW"}},
        "outputs": {"boxes": "float32[N,4]", "scores": "float32[N]", "class_ids": "int64[N]"},
        "names": {str(index): name for index, name in enumerate(names)},
        "nms_required": nms_required,
    }


def classification_contract(names: list[str] | tuple[str, ...]) -> dict[str, Any]:
    return {
        "version": "vision-workflows-classification-v1",
        "task": "classification",
        "inputs": {"images": {"dtype": "float32", "layout": "NCHW"}},
        "outputs": {"probabilities": "float32[N,C]"},
        "names": {str(index): name for index, name in enumerate(names)},
    }


def validate_onnx(path: Path, contract: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    onnx = optional_module("onnx")
    model = onnx.load(str(require_file(path, "ONNX artifact")))
    onnx.checker.check_model(model)
    metadata = {item.key: item.value for item in model.metadata_props}
    checks = [{"name": "onnx_checker", "status": "passed"}]
    if contract.get("version") and metadata.get("contract_version") not in {None, contract["version"]}:
        checks.append({"name": "contract_version", "status": "failed", "expected": contract["version"], "actual": metadata.get("contract_version")})
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
