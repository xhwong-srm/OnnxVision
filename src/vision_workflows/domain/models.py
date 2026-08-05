from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable, Mapping

from .datasets import DatasetFormat, TaskKind
from .errors import ConfigurationError


class Operation(StrEnum):
    TRAIN = "train"
    EXPORT = "export"
    VALIDATE = "validate"
    TEST = "test"


class ParameterOrigin(StrEnum):
    GENERAL = "general"
    FRAMEWORK = "framework"
    TASK = "task"
    MODEL = "model"


_UNSET = object()


@dataclass(frozen=True)
class ModelSelection:
    task: TaskKind
    framework: str
    model: str

    def __post_init__(self) -> None:
        for name, value in (("framework", self.framework), ("model", self.model)):
            if not value or value != value.strip():
                raise ValueError(f"{name} must be a non-empty normalized identifier")

    def __str__(self) -> str:
        return f"{self.task.value}/{self.framework}/{self.model}"


@dataclass(frozen=True)
class ModelInfo:
    id: str
    native_id: str
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def variant(self) -> str:
        return str(self.metadata.get("variant", self.id))


@dataclass(frozen=True)
class ParameterContext:
    selection: ModelSelection
    model: ModelInfo
    request: Any | None = None


DefaultFactory = Callable[[ParameterContext], Any]


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    value_type: type
    help: str
    default: Any = _UNSET
    default_factory: DefaultFactory | None = None
    choices: tuple[Any, ...] = ()
    origin: ParameterOrigin = ParameterOrigin.GENERAL
    allow_none: bool = False
    minimum: int | float | None = None
    maximum: int | float | None = None

    def __post_init__(self) -> None:
        if not self.name or "-" in self.name:
            raise ValueError("parameter names must be non-empty snake_case identifiers")
        if self.default is not _UNSET and self.default_factory is not None:
            raise ValueError(f"parameter {self.name} cannot define both default and default_factory")

    @property
    def cli_flag(self) -> str:
        return f"--{self.name.replace('_', '-')}"

    @property
    def required(self) -> bool:
        return self.default is _UNSET and self.default_factory is None

    def resolve_default(self, context: ParameterContext) -> Any:
        if self.default_factory is not None:
            return self.default_factory(context)
        if self.default is _UNSET:
            raise ConfigurationError(f"missing required parameter: {self.name}")
        return self.default

    def validate(self, value: Any) -> Any:
        if value is None and self.allow_none:
            return None
        raw: Any = value
        if self.value_type is bool:
            if not isinstance(value, bool):
                raise ConfigurationError(f"{self.name} must be a boolean")
            result = value
        elif self.value_type is int:
            if isinstance(value, bool):
                raise ConfigurationError(f"{self.name} must be an integer")
            try:
                result = int(raw)
            except (TypeError, ValueError) as error:
                raise ConfigurationError(f"{self.name} must be an integer") from error
            if isinstance(value, str) and value.strip() != str(result):
                raise ConfigurationError(f"{self.name} must be an integer")
        elif self.value_type is float:
            if isinstance(value, bool):
                raise ConfigurationError(f"{self.name} must be a number")
            try:
                result = float(raw)
            except (TypeError, ValueError) as error:
                raise ConfigurationError(f"{self.name} must be a number") from error
        elif self.value_type is Path:
            result = Path(raw).expanduser()
        else:
            result = self.value_type(value)
        if self.choices and result not in self.choices:
            choices = ", ".join(map(str, self.choices))
            raise ConfigurationError(f"{self.name} must be one of: {choices}")
        if self.minimum is not None and isinstance(result, (int, float)) and result < self.minimum:
            raise ConfigurationError(f"{self.name} must be at least {self.minimum}")
        if self.maximum is not None and isinstance(result, (int, float)) and result > self.maximum:
            raise ConfigurationError(f"{self.name} must be at most {self.maximum}")
        return result

    def describe_default(self) -> Any:
        if self.default_factory is not None:
            return "resolved by provider"
        return None if self.default is _UNSET else self.default


@dataclass(frozen=True)
class ParameterSchema:
    parameters: tuple[ParameterSpec, ...] = ()

    def __post_init__(self) -> None:
        names = [item.name for item in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("parameter schema contains duplicate names")

    def by_name(self) -> dict[str, ParameterSpec]:
        return {item.name: item for item in self.parameters}

    def compose(self, *layers: ParameterSchema) -> ParameterSchema:
        merged = self.by_name()
        order = [item.name for item in self.parameters]
        for layer in layers:
            for item in layer.parameters:
                previous = merged.get(item.name)
                if previous is not None and previous.value_type is not item.value_type:
                    raise ValueError(f"parameter {item.name} cannot change type between schema layers")
                if previous is None:
                    order.append(item.name)
                merged[item.name] = item
        return ParameterSchema(tuple(merged[name] for name in order))

    def with_origin(self, origin: ParameterOrigin) -> ParameterSchema:
        return ParameterSchema(tuple(replace(item, origin=origin) for item in self.parameters))

    def resolve(self, overrides: Mapping[str, Any], context: ParameterContext) -> dict[str, Any]:
        definitions = self.by_name()
        unknown = sorted(set(overrides) - set(definitions))
        if unknown:
            raise ConfigurationError(f"unsupported parameters: {', '.join(unknown)}")
        resolved: dict[str, Any] = {}
        for name, definition in definitions.items():
            value = overrides[name] if name in overrides else definition.resolve_default(context)
            resolved[name] = definition.validate(value)
        return resolved

    def describe(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "name": item.name,
                "flag": item.cli_flag,
                "type": item.value_type.__name__,
                "default": item.describe_default(),
                "required": item.required,
                "choices": item.choices,
                "minimum": item.minimum,
                "maximum": item.maximum,
                "origin": item.origin.value,
                "help": item.help,
            }
            for item in self.parameters
        )


@dataclass(frozen=True)
class DatasetRequirement:
    formats: tuple[DatasetFormat, ...]
    required_splits: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProviderDescriptor:
    framework: str
    task: TaskKind
    operations: frozenset[Operation]
    description: str
    optional_dependency: str | None = None


@dataclass(frozen=True)
class StaticModelCatalog:
    models: tuple[ModelInfo, ...]

    def list(self, pattern: str | None = None) -> tuple[ModelInfo, ...]:
        return tuple(item for item in self.models if pattern is None or fnmatch(item.id, pattern))

    def resolve(self, model: str) -> ModelInfo:
        match = next((item for item in self.models if item.id == model), None)
        if match is None:
            raise ConfigurationError(f"unsupported model: {model}")
        return match
