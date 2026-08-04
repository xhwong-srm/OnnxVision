from __future__ import annotations

from pathlib import Path

import pytest

from vision_workflows.backends.base import BackendExecution, ModelBackend
from vision_workflows.backends.registry import descriptors
from vision_workflows.domain.models import BackendCapability, BackendDescriptor, ModelRef
from vision_workflows.domain.results import ArtifactRef, RunStatus
from vision_workflows.workflows.requests import TrainRequest
from vision_workflows.workflows.runs import RunStore
from vision_workflows.workflows.services import TrainService


def test_registry_exposes_all_active_backend_families() -> None:
    values = {(item.backend, item.family) for item in descriptors()}
    assert values == {
        ("timm", "classification"),
        ("timm", "detection"),
        ("ultralytics", "yolo26"),
        ("libreyolo", "yolov9"),
        ("libreyolo", "picodet"),
    }
    assert all(BackendCapability.TRAIN in item.capabilities for item in descriptors())


def test_model_ref_requires_three_components() -> None:
    assert str(ModelRef.parse("ultralytics/yolo26/n")) == "ultralytics/yolo26/n"
    with pytest.raises(ValueError):
        ModelRef.parse("yolo26n")


def test_train_service_writes_typed_run_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from vision_workflows.backends import registry

    class FakeBackend(ModelBackend):
        descriptor = BackendDescriptor("fake", "unit", "classification", ("v1",), frozenset(BackendCapability), "fake")

        def train(self, request, context):
            path = context.run_dir / "best.pt"
            path.write_bytes(b"checkpoint")
            return BackendExecutionResult((ArtifactRef("best.pt", path, "checkpoint"),), {"accuracy": 1.0})

        def export(self, request, context):
            raise NotImplementedError

        def validate(self, request, context):
            raise NotImplementedError

        def test(self, request, context):
            raise NotImplementedError

    class BackendExecutionResult(BackendExecution):
        def __init__(self, artifacts, metrics):
            self.artifacts = artifacts
            self.metrics = metrics

    monkeypatch.setattr(registry, "_BACKENDS", (FakeBackend(),))
    result = TrainService().run(TrainRequest(ModelRef("fake", "unit", "v1"), tmp_path / "data", tmp_path / "run", epochs=1))
    assert result.best_checkpoint is not None
    assert result.run.status.value == "succeeded"
    assert (tmp_path / "run" / "manifest.json").is_file()


def test_run_store_creates_unique_immutable_run_directories(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    first_context, first_id = store.start("test", {"value": 1})
    first = store.finish(first_id, "test", {"value": 1}, status=RunStatus.SUCCEEDED)
    second_context, second_id = store.start("test", {"value": 2})
    second = store.finish(second_id, "test", {"value": 2}, status=RunStatus.SUCCEEDED)
    assert first.run_dir != second.run_dir
    assert first_context.run_dir == first.run_dir
    assert second_context.run_dir == second.run_dir
