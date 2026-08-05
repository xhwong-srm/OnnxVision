from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from ..domain.models import DatasetRequirement, ModelInfo, Operation, ParameterSchema, ProviderDescriptor
from ..domain.results import ArtifactRef
from ..workflows.context import WorkflowContext
from ..workflows.requests import ResolvedRequest


@dataclass(frozen=True)
class OperationExecution:
    artifacts: tuple[ArtifactRef, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    contract: Mapping[str, Any] = field(default_factory=dict)
    checks: tuple[dict[str, Any], ...] = ()


class ModelCatalog(Protocol):
    def list(self, pattern: str | None = None) -> tuple[ModelInfo, ...]: ...

    def resolve(self, model: str) -> ModelInfo: ...


SchemaFactory = Callable[[ModelInfo], ParameterSchema]
OperationCallable = Callable[[ResolvedRequest, WorkflowContext], OperationExecution]


@dataclass(frozen=True)
class OperationHandler:
    schema: SchemaFactory
    execute: OperationCallable
    dataset: DatasetRequirement | None = None


@dataclass(frozen=True)
class FrameworkTaskPlugin:
    descriptor: ProviderDescriptor
    catalog: ModelCatalog
    handlers: Mapping[Operation, OperationHandler]

    def __post_init__(self) -> None:
        if frozenset(self.handlers) != self.descriptor.operations:
            raise ValueError("descriptor operations must match registered handlers")


BackendExecution = OperationExecution
