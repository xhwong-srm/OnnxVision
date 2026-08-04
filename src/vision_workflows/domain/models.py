from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BackendCapability(StrEnum):
    TRAIN = "train"
    EXPORT = "export"
    VALIDATE = "validate"
    TEST = "test"


@dataclass(frozen=True)
class ModelRef:
    backend: str
    family: str
    variant: str

    @classmethod
    def parse(cls, value: str) -> "ModelRef":
        parts = tuple(part for part in value.split("/") if part)
        if len(parts) != 3:
            raise ValueError("model must use BACKEND/FAMILY/VARIANT")
        return cls(*parts)

    def __str__(self) -> str:
        return f"{self.backend}/{self.family}/{self.variant}"


@dataclass(frozen=True)
class BackendDescriptor:
    backend: str
    family: str
    task: str
    variants: tuple[str, ...]
    capabilities: frozenset[BackendCapability]
    description: str
    optional_dependency: str | None = None

    @property
    def model_prefix(self) -> str:
        return f"{self.backend}/{self.family}"
