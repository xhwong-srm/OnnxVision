from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..domain.models import ModelInfo, ModelSelection


@dataclass(frozen=True)
class TrainRequest:
    selection: ModelSelection
    data: Path
    output: Path
    weights: Path | None = None
    resume: bool = False
    overwrite: bool = False
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TuneRequest:
    selection: ModelSelection
    data: Path
    output: Path
    overwrite: bool = False
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportRequest:
    selection: ModelSelection
    checkpoint: Path
    output: Path
    data: Path | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidateRequest:
    selection: ModelSelection
    target: Path
    data: Path | None = None
    split: str = "val"
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TestRequest:
    selection: ModelSelection
    target: Path
    data: Path
    split: str = "test"
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedTrainRequest:
    selection: ModelSelection
    model: ModelInfo
    data: Path
    output: Path
    weights: Path | None
    resume: bool
    overwrite: bool
    parameters: Mapping[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self.parameters[name]
        except KeyError as error:
            raise AttributeError(name) from error

    @property
    def options(self) -> Mapping[str, Any]:
        return self.parameters


@dataclass(frozen=True)
class ResolvedTuneRequest:
    selection: ModelSelection
    model: ModelInfo
    data: Path
    output: Path
    overwrite: bool
    parameters: Mapping[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self.parameters[name]
        except KeyError as error:
            raise AttributeError(name) from error

    @property
    def options(self) -> Mapping[str, Any]:
        return self.parameters


@dataclass(frozen=True)
class ResolvedExportRequest:
    selection: ModelSelection
    model: ModelInfo
    checkpoint: Path
    output: Path
    data: Path | None
    parameters: Mapping[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self.parameters[name]
        except KeyError as error:
            raise AttributeError(name) from error

    @property
    def options(self) -> Mapping[str, Any]:
        return self.parameters


@dataclass(frozen=True)
class ResolvedValidateRequest:
    selection: ModelSelection
    model: ModelInfo
    target: Path
    data: Path | None
    split: str
    parameters: Mapping[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self.parameters[name]
        except KeyError as error:
            raise AttributeError(name) from error

    @property
    def options(self) -> Mapping[str, Any]:
        return self.parameters


@dataclass(frozen=True)
class ResolvedTestRequest:
    selection: ModelSelection
    model: ModelInfo
    target: Path
    data: Path
    split: str
    parameters: Mapping[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self.parameters[name]
        except KeyError as error:
            raise AttributeError(name) from error

    @property
    def options(self) -> Mapping[str, Any]:
        return self.parameters


ResolvedRequest = ResolvedTrainRequest | ResolvedTuneRequest | ResolvedExportRequest | ResolvedValidateRequest | ResolvedTestRequest
