from __future__ import annotations

from pathlib import Path

import pytest

from vision_workflows.backends.base import BackendExecution, ModelBackend
from vision_workflows.backends.libreyolo import LibreYoloBackend
from vision_workflows.backends.registry import descriptors
from vision_workflows.backends.ultralytics import UltralyticsBackend
from vision_workflows.domain.models import BackendCapability, BackendDescriptor, ModelRef
from vision_workflows.domain.results import ArtifactRef, RunStatus
from vision_workflows.workflows.requests import TrainRequest
from vision_workflows.workflows.runs import RunStore
from vision_workflows.workflows.services import TrainService
from vision_workflows.workflows.context import optional_import


def test_registry_exposes_all_active_backend_families() -> None:
    values = {(item.backend, item.family) for item in descriptors()}
    assert values == {
        ("timm", "classification"),
        ("timm", "detection"),
        ("ultralytics", "yolo26"),
        ("ultralytics", "yolo26-cls"),
        ("libreyolo", "yolov9"),
        ("libreyolo", "picodet"),
    }
    assert all(BackendCapability.TRAIN in item.capabilities for item in descriptors())


def test_model_ref_requires_three_components() -> None:
    assert str(ModelRef.parse("ultralytics/yolo26/n")) == "ultralytics/yolo26/n"
    with pytest.raises(ValueError):
        ModelRef.parse("yolo26n")


def test_optional_import_does_not_eagerly_load_optional_exports() -> None:
    libreyolo = optional_import("libreyolo")
    assert libreyolo.LibrePICODET is not None


def test_libreyolo_training_clamps_auto_worker_sentinel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def train(self, **options):
            captured.update(options)
            return {}

    class FakeModule:
        LibrePICODET = lambda *args, **kwargs: FakeModel()

    import vision_workflows.backends.libreyolo as backend_module
    monkeypatch.setattr(backend_module, "optional_import", lambda _: FakeModule())
    data = tmp_path / "data.yaml"
    data.write_text("names: [seal]\n", encoding="utf-8")
    request = TrainRequest(ModelRef("libreyolo", "picodet", "s"), data, tmp_path / "run", workers=-1)
    context = type("Context", (), {"run_dir": tmp_path / "run"})()

    LibreYoloBackend("picodet", ("s",)).train(request, context)

    assert captured["workers"] == 0


def test_ultralytics_classification_training_uses_image_folder_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    events: list[tuple[str, dict[str, object]]] = []

    class FakeModel:
        def __init__(self, checkpoint: str):
            captured["checkpoint"] = checkpoint

        def train(self, **options):
            captured.update(options)
            weights = Path(options["project"]) / options["name"] / "weights"
            weights.mkdir(parents=True)
            (weights / "best.pt").write_bytes(b"best")
            (weights / "last.pt").write_bytes(b"last")

    class FakeModule:
        YOLO = FakeModel

    import vision_workflows.backends.ultralytics as backend_module
    monkeypatch.setattr(backend_module, "optional_import", lambda _: FakeModule())
    data = tmp_path / "classification"
    (data / "train" / "seal").mkdir(parents=True)
    (data / "val" / "seal").mkdir(parents=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    context = type("Context", (), {"run_dir": run_dir, "emit": lambda _, name, value: events.append((name, value))})()
    request = TrainRequest(
        ModelRef("ultralytics", "yolo26-cls", "n"), data, run_dir, workers=2,
        options={"validate_every": 2, "mosaic": 0.0},
    )

    result = UltralyticsBackend("classification").train(request, context)

    assert captured["checkpoint"] == "yolo26n-cls.pt"
    assert captured["data"] == str(data.resolve())
    assert captured["imgsz"] == 224
    assert captured["workers"] == 2
    assert captured["mosaic"] == 0.0
    assert "validate_every" not in captured
    assert "device" not in captured
    assert {item.name for item in result.artifacts} == {"best.pt", "last.pt"}
    assert events == [("backend_train_started", {"backend": "ultralytics/yolo26-cls", "task": "classification", "data": str(data.resolve())})]


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


def test_run_store_overwrite_removes_existing_requested_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "old-artifact.pt").write_bytes(b"old")
    store = RunStore(tmp_path / "runs")

    context, run_id = store.start("train", {}, run_dir=run_dir, overwrite=True)

    assert context.run_dir == run_dir.resolve()
    assert not (run_dir / "old-artifact.pt").exists()
    assert (run_dir / "manifest.json").is_file()


def test_run_store_refuses_overwriting_current_directory(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")

    with pytest.raises(ValueError, match="current directory"):
        store.start("train", {}, run_dir=Path.cwd(), overwrite=True)


def test_run_store_allows_existing_directory_for_resume(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    store = RunStore(tmp_path / "runs")

    context, _ = store.start("train", {}, run_dir=run_dir, allow_existing=True)

    assert context.run_dir == run_dir.resolve()
    assert checkpoint.read_bytes() == b"checkpoint"
