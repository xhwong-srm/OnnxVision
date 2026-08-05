from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain.errors import ConfigurationError
from ..domain.models import BackendCapability, BackendDescriptor
from ..domain.results import ArtifactRef
from ..workflows.context import WorkflowContext, optional_import
from ..workflows.requests import ExportRequest, TestRequest, TrainRequest, ValidateRequest
from ..workflows.runs import artifact
from .base import BackendExecution, ModelBackend
from .common import (
    artifacts_for,
    class_names_from_model,
    classification_contract,
    detection_contract,
    metadata_for_contract,
    require_file,
    set_onnx_metadata,
    validate_onnx,
)


@dataclass(frozen=True)
class Execution(BackendExecution):
    artifacts: tuple[ArtifactRef, ...] = ()
    metrics: dict[str, Any] = None  # type: ignore[assignment]
    contract: dict[str, Any] = None  # type: ignore[assignment]
    checks: tuple[dict[str, Any], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "metrics", self.metrics or {})
        object.__setattr__(self, "contract", self.contract or {})


class UltralyticsBackend(ModelBackend):
    def __init__(self, task: str = "detection"):
        if task not in {"classification", "detection"}:
            raise ValueError(f"unsupported Ultralytics task: {task}")
        family = "yolo26-cls" if task == "classification" else "yolo26"
        description = "Ultralytics YOLO26 classifier" if task == "classification" else "Ultralytics YOLO26 detector"
        self._task = task
        self._descriptor = BackendDescriptor(
            "ultralytics", family, task, ("n", "s", "m", "l", "x"),
            frozenset(BackendCapability), description, "ultralytics",
        )

    @property
    def descriptor(self) -> BackendDescriptor:
        return self._descriptor

    def _data(self, request: TrainRequest | ValidateRequest | TestRequest) -> Path:
        data = request.data.expanduser().resolve()
        if self._task == "classification":
            if not data.is_dir():
                raise ConfigurationError(f"classification dataset directory does not exist: {data}")
            return data
        return require_file(data, "YOLO dataset YAML")

    def _model(self, request: TrainRequest | ExportRequest | ValidateRequest | TestRequest):
        module = optional_import("ultralytics")
        checkpoint = getattr(request, "weights", None) or getattr(request, "checkpoint", None) or getattr(request, "target", None)
        if checkpoint is None:
            suffix = "-cls" if self._task == "classification" else ""
            checkpoint = Path(f"yolo26{request.model.variant}{suffix}.pt")
        return module.YOLO(str(checkpoint))

    def train(self, request: TrainRequest, context: WorkflowContext) -> BackendExecution:
        data = self._data(request)
        model = self._model(request)
        options: dict[str, Any] = {
            "data": str(data), "epochs": request.epochs,
            "imgsz": request.image_size or (224 if self._task == "classification" else 640),
            "batch": request.batch, "lr0": request.learning_rate, "workers": request.workers,
            "patience": request.patience, "seed": request.seed, "project": str(context.run_dir.parent),
            "name": context.run_dir.name, "exist_ok": True, "resume": request.resume,
        }
        if request.device != "auto":
            options["device"] = request.device
        # This is a vision-workflows scheduler option consumed by the timm
        # trainers; Ultralytics rejects unknown model.train overrides.
        options.update({key: value for key, value in request.options.items() if key != "validate_every"})
        context.emit("backend_train_started", {"backend": str(self.descriptor.model_prefix), "task": self._task, "data": str(data)})
        model.train(**options)
        weights = context.run_dir / "weights"
        best, last = weights / "best.pt", weights / "last.pt"
        return Execution(artifacts_for([(best, "checkpoint"), (last, "checkpoint")]))

    def export(self, request: ExportRequest, context: WorkflowContext) -> BackendExecution:
        checkpoint = require_file(request.checkpoint, "checkpoint")
        model = self._model(request)
        options: dict[str, Any] = {"format": "onnx", "imgsz": request.image_size, "opset": request.opset, "simplify": request.simplify}
        if request.device != "auto":
            options["device"] = request.device
        options.update(request.options)
        exported = Path(model.export(**options))
        output = request.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if exported.resolve() != output:
            exported.replace(output)
        names = class_names_from_model(model)
        if not names:
            raise ValueError(f"the exported {self._task} model does not expose class names")
        contract = (
            classification_contract(names)
            if self._task == "classification"
            else detection_contract(names, nms_required=bool(request.options.get("nms_required", False)))
        )
        set_onnx_metadata(output, metadata_for_contract(contract))
        checks = validate_onnx(output, contract)
        return Execution((artifact(output, "onnx"),), {}, contract, checks)

    def validate(self, request: ValidateRequest, context: WorkflowContext) -> BackendExecution:
        target = require_file(request.target, "model artifact")
        if target.suffix.casefold() == ".onnx":
            contract = classification_contract([]) if self._task == "classification" else detection_contract([])
            checks = validate_onnx(target, contract)
            return Execution((artifact(target, "onnx"),), {}, contract, checks)
        model = self._model(request)
        if request.data:
            values = model.val(data=str(self._data(request)), split=request.split, device=request.device)
            metrics = getattr(values, "results_dict", {})
        else:
            metrics = {}
        return Execution((artifact(target, "checkpoint"),), metrics, {}, (({"name": "native_validation", "status": "passed"}),))

    def test(self, request: TestRequest, context: WorkflowContext) -> BackendExecution:
        model = self._model(request)
        values = model.val(data=str(self._data(request)), split=request.split, device=request.device)
        return Execution((artifact(require_file(request.target, "model artifact"), "model"),), getattr(values, "results_dict", {}), {}, ())
