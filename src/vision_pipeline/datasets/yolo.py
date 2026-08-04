"""Validation helpers for the YOLO data.yaml format."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_yolo_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required and is installed with the selected YOLO backend") from error
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read dataset YAML {path}: {error}") from error
    if not isinstance(document, dict) or "train" not in document or "val" not in document:
        raise ValueError(f"Dataset YAML must define train and val: {path}")
    return document


def class_names(document: dict[str, Any]) -> list[str]:
    names = document.get("names")
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        try:
            ordered = sorted((int(key), str(value)) for key, value in names.items())
        except (TypeError, ValueError) as error:
            raise ValueError("Dataset names mapping must use integer class IDs") from error
        if [key for key, _ in ordered] != list(range(len(ordered))):
            raise ValueError("Dataset class IDs must be contiguous and zero-based")
        return [value for _, value in ordered]
    raise ValueError("Dataset YAML must define names as a list or mapping")


def resolve_workers(requested: int) -> int:
    return requested if requested >= 0 else max(0, min(8, (os.cpu_count() or 1) // 2))


def configure_determinism(enabled: bool) -> None:
    if not enabled:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
