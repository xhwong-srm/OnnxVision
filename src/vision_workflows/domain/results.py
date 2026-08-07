from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class ArtifactRef:
    name: str
    path: Path
    kind: str
    sha256: str | None = None


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    operation: str
    status: RunStatus
    run_dir: Path
    config: dict[str, Any]
    inputs: tuple[dict[str, Any], ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class ConversionResult:
    input_format: str
    output_format: str
    output: Path
    classes: tuple[str, ...]
    split_counts: dict[str, int]
    manifest: Path


@dataclass(frozen=True)
class MergeResult:
    output: Path
    manifest: Path
    task: str
    sample_count: int
    classes: tuple[str, ...]
    split_counts: dict[str, int]


@dataclass(frozen=True)
class SplitResult:
    output: Path
    manifest: Path
    split_counts: dict[str, int]
    seed: int
    grouping: str


@dataclass(frozen=True)
class TrainResult:
    run: RunManifest
    best_checkpoint: ArtifactRef | None
    last_checkpoint: ArtifactRef | None
    metrics: dict[str, Any]


@dataclass(frozen=True)
class TuneResult:
    run: RunManifest
    best_checkpoint: ArtifactRef | None
    last_checkpoint: ArtifactRef | None
    metrics: dict[str, Any]


@dataclass(frozen=True)
class ExportResult:
    run: RunManifest
    contract: dict[str, Any]
    validation: dict[str, Any]


@dataclass(frozen=True)
class ValidationResult:
    run: RunManifest
    valid: bool
    checks: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TestResult:
    run: RunManifest
    split: str
    metrics: dict[str, Any]
