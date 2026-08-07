from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from vision_workflows.backends.base import FrameworkTaskPlugin, OperationExecution, OperationHandler
from vision_workflows.backends.registry import frameworks, models_for, plugin_for
from vision_workflows.domain.datasets import TaskKind
from vision_workflows.domain.errors import ConfigurationError
from vision_workflows.domain.models import ModelInfo, ModelSelection, Operation, ParameterSchema, ParameterSpec, ProviderDescriptor, StaticModelCatalog
from vision_workflows.domain.results import ArtifactRef, RunStatus
from vision_workflows.workflows.context import optional_import
from vision_workflows.workflows.requests import ExportRequest, TrainRequest, TuneRequest
from vision_workflows.workflows.runs import RunStore
from vision_workflows.workflows.services import ExportService, TrainService, TuneService


def test_registry_exposes_framework_task_plugins() -> None:
    values = {(item.framework, item.task) for item in frameworks()}
    assert values == {
        ("timm", TaskKind.CLASSIFICATION),
        ("ultralytics", TaskKind.CLASSIFICATION),
        ("ultralytics", TaskKind.OBJECT_DETECTION),
        ("libreyolo", TaskKind.OBJECT_DETECTION),
    }
    assert all(Operation.TRAIN in item.operations for item in frameworks())
    timm_descriptor = next(item for item in frameworks() if item.framework == "timm" and item.task is TaskKind.CLASSIFICATION)
    assert Operation.TUNE in timm_descriptor.operations
    assert all(Operation.TUNE in item.operations for item in frameworks() if item.framework == "ultralytics")
    assert all(Operation.TUNE not in item.operations for item in frameworks() if item.framework == "libreyolo")
    with pytest.raises(ConfigurationError, match="unsupported framework/task: pytorch/object-detection"):
        plugin_for(ModelSelection(TaskKind.OBJECT_DETECTION, "pytorch", "timm-obd-v1"))


def test_model_selection_is_explicit_and_catalog_resolves_native_model() -> None:
    selection = ModelSelection(TaskKind.CLASSIFICATION, "ultralytics", "yolo26n")
    assert str(selection) == "classification/ultralytics/yolo26n"
    assert plugin_for(selection).catalog.resolve("yolo26n").native_id == "yolo26n-cls.pt"
    assert models_for(TaskKind.OBJECT_DETECTION, "libreyolo", "picodet*")


def test_timm_classification_catalog_exposes_only_validated_model() -> None:
    model = "mobilenetv4_conv_small_050.e3000_r224_in1k"
    assert [item.id for item in models_for(TaskKind.CLASSIFICATION, "timm")] == [model]
    with pytest.raises(ConfigurationError, match="unsupported model: resnet18"):
        plugin_for(ModelSelection(TaskKind.CLASSIFICATION, "timm", "resnet18")).catalog.resolve("resnet18")


def test_timm_training_and_tuning_schemas_expose_first_class_integrations() -> None:
    selection = ModelSelection(TaskKind.CLASSIFICATION, "timm", "mobilenetv4_conv_small_050.e3000_r224_in1k")
    plugin = plugin_for(selection)
    model = plugin.catalog.resolve(selection.model)
    context = type("Context", (), {"selection": selection, "model": model, "request": None})()

    train = plugin.handlers[Operation.TRAIN].schema(model).resolve({}, context)
    tune = plugin.handlers[Operation.TUNE].schema(model).resolve({}, context)
    assert train["augmentation"] is True
    assert train["augmentation_backend"] == "torchvision"
    assert train["augmentation_policy"] == "standard"
    assert train["cache"] == "none"
    assert train["val_workers"] == 0
    assert train["weight_decay"] == 0.01
    assert train["label_smoothing"] == 0.0
    assert train["warmup_epochs"] == 2
    assert tune["trials"] == 20
    assert tune["epochs"] == 10
    assert tune["patience"] == 5
    assert tune["label_smoothing_min"] < tune["label_smoothing_max"]
    assert tune["learning_rate_min"] < tune["learning_rate_max"]
    assert tune["weight_decay_min"] < tune["weight_decay_max"]


def test_provider_schema_rejects_parameters_owned_by_another_provider() -> None:
    selection = ModelSelection(TaskKind.CLASSIFICATION, "ultralytics", "yolo26n")
    plugin = plugin_for(selection)
    model = plugin.catalog.resolve(selection.model)
    schema = plugin.handlers[Operation.TRAIN].schema(model)
    with pytest.raises(ConfigurationError, match="validate_every"):
        schema.resolve({"validate_every": 2}, type("Context", (), {"selection": selection, "model": model, "request": None})())


def test_ultralytics_defaults_workers_to_zero() -> None:
    selection = ModelSelection(TaskKind.CLASSIFICATION, "ultralytics", "yolo26n")
    plugin = plugin_for(selection)
    model = plugin.catalog.resolve(selection.model)
    schema = plugin.handlers[Operation.TRAIN].schema(model)
    context = type("Context", (), {"selection": selection, "model": model, "request": None})()
    assert schema.resolve({}, context)["workers"] == 0
    with pytest.raises(ConfigurationError, match="workers must be at least 0"):
        schema.resolve({"workers": -1}, context)


def test_export_schema_defaults_to_dynamic_and_rejects_nonpositive_fixed_batch() -> None:
    selection = ModelSelection(TaskKind.CLASSIFICATION, "timm", "mobilenetv4_conv_small_050.e3000_r224_in1k")
    plugin = plugin_for(selection)
    model = plugin.catalog.resolve(selection.model)
    schema = plugin.handlers[Operation.EXPORT].schema(model)
    context = type("Context", (), {"selection": selection, "model": model, "request": None})()
    assert schema.resolve({}, context)["batch_size"] is None
    assert schema.resolve({"batch_size": 4}, context)["batch_size"] == 4
    with pytest.raises(ConfigurationError, match="batch_size must be at least 1"):
        schema.resolve({"batch_size": 0}, context)


def test_libreyolo_picodet_export_defaults_to_native_image_size() -> None:
    selection = ModelSelection(TaskKind.OBJECT_DETECTION, "libreyolo", "picodets")
    plugin = plugin_for(selection)
    model = plugin.catalog.resolve(selection.model)
    schema = plugin.handlers[Operation.EXPORT].schema(model)
    context = type("Context", (), {"selection": selection, "model": model, "request": None})()

    assert schema.resolve({}, context)["image_size"] == 320


def test_ultralytics_auto_validation_omits_device(monkeypatch: pytest.MonkeyPatch) -> None:
    import vision_workflows.backends.ultralytics as integration

    assert integration._validation_options(Path("data.yaml"), "val", "auto") == {
        "data": str(Path("data.yaml")),
        "split": "val",
    }
    assert integration._validation_options(Path("data.yaml"), "val", "0") == {
        "data": str(Path("data.yaml")),
        "split": "val",
        "device": "0",
    }


def test_ultralytics_classification_translates_only_its_supported_parameters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self, checkpoint: str):
            captured["checkpoint"] = checkpoint

        def train(self, **options):
            captured.update(options)
            weights = Path(options["project"]) / str(options["name"]) / "weights"
            weights.mkdir(parents=True)
            (weights / "best.pt").write_bytes(b"best")
            (weights / "last.pt").write_bytes(b"last")

    import vision_workflows.backends.ultralytics as integration
    monkeypatch.setattr(integration, "optional_import", lambda _: type("Module", (), {"YOLO": FakeModel})())
    data = tmp_path / "images"
    data.mkdir()
    selection = ModelSelection(TaskKind.CLASSIFICATION, "ultralytics", "yolo26n")
    TrainService().run(TrainRequest(selection, data, tmp_path / "run"))
    assert captured["checkpoint"] == "yolo26n-cls.pt"
    assert captured["workers"] == 0
    assert captured["imgsz"] == 224
    assert "validate_every" not in captured


def test_libreyolo_translates_amp_and_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self, *args, **kwargs):
            captured["model_kwargs"] = kwargs

        def train(self, **options):
            captured.update(options)
            weights = Path(options["project"]) / str(options["name"]) / "weights"
            weights.mkdir(parents=True)
            (weights / "best.pt").write_bytes(b"best")
            (weights / "last.pt").write_bytes(b"last")
            return {}

    import vision_workflows.backends.libreyolo as integration
    monkeypatch.setattr(
        integration,
        "optional_import",
        lambda _: type("Module", (), {"LibreYOLO9": FakeModel, "LibrePICODET": FakeModel})(),
    )
    data = tmp_path / "data.yaml"
    data.write_text("names: [seal]\n", encoding="utf-8")
    selection = ModelSelection(TaskKind.OBJECT_DETECTION, "libreyolo", "yolov9t")

    TrainService().run(
        TrainRequest(
            selection,
            data,
            tmp_path / "run",
            parameters={"amp": False, "cache": "disk"},
        )
    )

    assert captured["amp"] is False
    assert captured["cache"] == "disk"


def test_ultralytics_tune_delegates_to_native_tune_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self, checkpoint: str):
            captured["checkpoint"] = checkpoint

        def tune(self, **options):
            captured.update(options)
            return {"best_fitness": 0.9}

    import vision_workflows.backends.ultralytics as integration
    monkeypatch.setattr(integration, "optional_import", lambda _: type("Module", (), {"YOLO": FakeModel})())
    data = tmp_path / "images"
    data.mkdir()
    selection = ModelSelection(TaskKind.CLASSIFICATION, "ultralytics", "yolo26n")

    result = TuneService().run(
        TuneRequest(
            selection,
            data,
            tmp_path / "run",
            parameters={"iterations": 2, "epochs": 3, "optimizer": "AdamW"},
        )
    )

    assert captured["checkpoint"] == "yolo26n-cls.pt"
    assert captured["iterations"] == 2
    assert captured["epochs"] == 3
    assert captured["optimizer"] == "AdamW"
    assert captured["plots"] is False
    assert result.metrics["best_fitness"] == 0.9


def test_optional_import_does_not_eagerly_load_optional_exports() -> None:
    libreyolo = optional_import("libreyolo")
    assert libreyolo.LibrePICODET is not None


def test_train_service_records_requested_and_effective_parameters(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    caplog.set_level(logging.INFO, logger="vision_workflows.workflows.services")
    from vision_workflows.backends import registry

    selection = ModelSelection(TaskKind.CLASSIFICATION, "fake", "unit")

    def train(request, context):
        path = context.run_dir / "best.pt"
        path.write_bytes(b"checkpoint")
        return OperationExecution((ArtifactRef("best.pt", path, "checkpoint"),), {"accuracy": 1.0})

    schema = ParameterSchema((ParameterSpec("epochs", int, "epochs", 3), ParameterSpec("device", str, "device", "cpu")))
    plugin = FrameworkTaskPlugin(
        ProviderDescriptor("fake", TaskKind.CLASSIFICATION, frozenset({Operation.TRAIN}), "fake"),
        StaticModelCatalog((ModelInfo("unit", "native-unit"),)),
        {Operation.TRAIN: OperationHandler(lambda _: schema, train)},
    )
    monkeypatch.setattr(registry, "_PLUGINS", (plugin,))
    result = TrainService().run(TrainRequest(selection, tmp_path / "data", tmp_path / "run", parameters={"epochs": 1}))
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert result.best_checkpoint is not None
    assert result.run.status is RunStatus.SUCCEEDED
    assert manifest["config"]["resolved_model"]["native_id"] == "native-unit"
    assert manifest["config"]["parameters"]["requested"] == {"epochs": 1}
    assert manifest["config"]["parameters"]["effective"] == {"device": "cpu", "epochs": 1}
    assert "Training outputs saved under" in caplog.text
    assert "config.json" in caplog.text
    assert "manifest.json" in caplog.text
    assert "best.pt" in caplog.text


def test_export_service_saves_result_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from vision_workflows.backends import registry

    selection = ModelSelection(TaskKind.CLASSIFICATION, "fake", "unit")

    def export(request, context):
        output = context.run_dir / "model-bw8.onnx"
        output.write_bytes(b"onnx")
        return OperationExecution((ArtifactRef(output.name, output, "onnx"),), {"images": 2}, {"name": "contract"})

    schema = ParameterSchema((ParameterSpec("device", str, "device", "auto"),))
    plugin = FrameworkTaskPlugin(
        ProviderDescriptor("fake", TaskKind.CLASSIFICATION, frozenset({Operation.EXPORT}), "fake"),
        StaticModelCatalog((ModelInfo("unit", "native-unit"),)),
        {Operation.EXPORT: OperationHandler(lambda _: schema, export)},
    )
    monkeypatch.setattr(registry, "_PLUGINS", (plugin,))

    result = ExportService().run(ExportRequest(selection, tmp_path / "checkpoint.pt", tmp_path / "model.onnx"))
    saved = tmp_path / "model.json"
    payload = json.loads(saved.read_text(encoding="utf-8"))

    assert payload["run"]["status"] == "succeeded"
    assert payload["contract"] == {"name": "contract"}
    assert payload["run"]["artifacts"][0]["name"] == "model-bw8.onnx"
    assert payload["run"]["metrics"] == {"images": 2}
    assert payload["validation"] == {"checks": []}
    assert "artifacts" not in payload


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
    context, _ = RunStore(tmp_path / "runs").start("train", {}, run_dir=run_dir, overwrite=True)
    assert context.run_dir == run_dir.resolve()
    assert not (run_dir / "old-artifact.pt").exists()


def test_run_store_refuses_overwriting_current_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="current directory"):
        RunStore(tmp_path / "runs").start("train", {}, run_dir=Path.cwd(), overwrite=True)
