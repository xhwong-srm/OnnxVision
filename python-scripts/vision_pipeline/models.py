"""Small, dependency-free types shared by the unified workflow CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class WorkflowDescriptor:
    """A registered model workflow and the native script it owns."""

    backend: str
    task: str
    model_family: str
    script: str
    operation: str
    description: str


@dataclass(frozen=True)
class TrainRequest:
    backend: str
    task: str
    model: str
    data: Path
    output: Path | None = None
    weights: Path | None = None
    imgsz: int | None = None
    epochs: int | None = None
    batch: int | None = None
    lr: float | None = None
    workers: int | None = None
    patience: int | None = None
    seed: int | None = None
    device: str | None = None
    resume: bool = False
    deterministic: bool | None = None
    pretrained: bool | None = None
    run_test: bool | None = None
    backend_args: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ExportRequest:
    backend: str
    task: str
    model: str
    checkpoint: Path
    output: Path | None = None
    imgsz: int | None = None
    opset: int | None = None
    simplify: bool | None = None
    embedded_preprocessing: bool | None = None
    device: str | None = None
    data: Path | None = None
    validation_split: str | None = None
    validation_limit: int | None = None
    validation_report: Path | None = None
    backend_args: tuple[str, ...] = field(default_factory=tuple)


def as_args(values: Sequence[str]) -> tuple[str, ...]:
    """Normalize argparse remainder values for immutable request objects."""
    return tuple(str(value) for value in values)
