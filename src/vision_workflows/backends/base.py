from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..domain.models import BackendDescriptor
from ..domain.results import ArtifactRef
from ..workflows.context import WorkflowContext
from ..workflows.requests import ExportRequest, TestRequest, TrainRequest, ValidateRequest


class BackendExecution(ABC):
    artifacts: tuple[ArtifactRef, ...] = ()
    metrics: dict[str, Any] = {}
    contract: dict[str, Any] = {}
    checks: tuple[dict[str, Any], ...] = ()


class ModelBackend(ABC):
    @property
    @abstractmethod
    def descriptor(self) -> BackendDescriptor:
        raise NotImplementedError

    @abstractmethod
    def train(self, request: TrainRequest, context: WorkflowContext) -> BackendExecution:
        raise NotImplementedError

    @abstractmethod
    def export(self, request: ExportRequest, context: WorkflowContext) -> BackendExecution:
        raise NotImplementedError

    @abstractmethod
    def validate(self, request: ValidateRequest, context: WorkflowContext) -> BackendExecution:
        raise NotImplementedError

    @abstractmethod
    def test(self, request: TestRequest, context: WorkflowContext) -> BackendExecution:
        raise NotImplementedError
