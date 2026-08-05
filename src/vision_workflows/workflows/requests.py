from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..domain.models import ModelRef


@dataclass(frozen=True)
class TrainRequest:
    model: ModelRef
    data: Path
    output: Path
    epochs: int = 100
    batch: int = 16
    image_size: int = 640
    learning_rate: float = 1e-3
    workers: int = -1
    patience: int = 20
    seed: int = 42
    device: str = "auto"
    weights: Path | None = None
    resume: bool = False
    pretrained: bool = True
    deterministic: bool = True
    overwrite: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportRequest:
    model: ModelRef
    checkpoint: Path
    output: Path
    data: Path | None = None
    image_size: int = 640
    opset: int = 18
    simplify: bool = True
    device: str = "auto"
    embedded_preprocessing: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidateRequest:
    model: ModelRef
    target: Path
    data: Path | None = None
    split: str = "val"
    device: str = "cpu"
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TestRequest:
    model: ModelRef
    target: Path
    data: Path
    split: str = "test"
    device: str = "auto"
    options: Mapping[str, Any] = field(default_factory=dict)
