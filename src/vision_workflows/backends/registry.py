from __future__ import annotations

from ..domain.errors import ConfigurationError
from ..domain.models import BackendCapability, BackendDescriptor, ModelRef
from .base import ModelBackend
from .libreyolo import LibreYoloBackend
from .timm_classification import TimmClassificationBackend
from .timm_detection import TimmDetectionBackend
from .ultralytics import UltralyticsBackend


_BACKENDS: tuple[ModelBackend, ...] = (
    TimmClassificationBackend(),
    TimmDetectionBackend(),
    UltralyticsBackend(),
    LibreYoloBackend("yolov9", ("t", "s", "m", "c")),
    LibreYoloBackend("picodet", ("s", "m", "l")),
)


def descriptors() -> tuple[BackendDescriptor, ...]:
    return tuple(backend.descriptor for backend in _BACKENDS)


def backends() -> tuple[ModelBackend, ...]:
    return _BACKENDS


def backend_for(model: ModelRef) -> ModelBackend:
    for backend in _BACKENDS:
        descriptor = backend.descriptor
        if descriptor.backend == model.backend and descriptor.family == model.family:
            if "*" in descriptor.variants or model.variant in descriptor.variants:
                return backend
    raise ConfigurationError(f"unsupported model: {model}")


def list_models() -> tuple[BackendDescriptor, ...]:
    return descriptors()
