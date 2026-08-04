from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain.models import BackendCapability, BackendDescriptor
from ..domain.results import ArtifactRef
from ..workflows.context import WorkflowContext, optional_import
from ..workflows.requests import ExportRequest, TestRequest, TrainRequest, ValidateRequest
from ..workflows.runs import artifact
from .base import BackendExecution, ModelBackend
from .common import artifacts_for, detection_contract, require_file, set_onnx_metadata, validate_onnx


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
    descriptor = BackendDescriptor(
        "ultralytics", "yolo26", "detection", ("n", "s", "m", "l", "x"),
        frozenset(BackendCapability), "Ultralytics YOLO26 detector", "ultralytics",
    )

    def _model(self, request: TrainRequest | ExportRequest | ValidateRequest | TestRequest):
        module = optional_import("ultralytics")
        checkpoint = getattr(request, "weights", None) or getattr(request, "checkpoint", None) or getattr(request, "target", None)
        if checkpoint is None:
            checkpoint = Path(f"yolo26{request.model.variant}.pt")
        return module.YOLO(str(checkpoint))

    def train(self, request: TrainRequest, context: WorkflowContext) -> BackendExecution:
        data = require_file(request.data, "YOLO dataset YAML")
        model = self._model(request)
        options: dict[str, Any] = {
            "data": str(data), "epochs": request.epochs, "imgsz": request.image_size,
            "batch": request.batch, "lr0": request.learning_rate, "workers": request.workers,
            "patience": request.patience, "seed": request.seed, "project": str(context.run_dir.parent),
            "name": context.run_dir.name, "exist_ok": True, "resume": request.resume,
        }
        if request.device != "auto":
            options["device"] = request.device
        options.update(request.options)
        context.emit("backend_train_started", {"backend": "ultralytics/yolo26", "data": str(data)})
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
        contract = detection_contract([], nms_required=bool(request.options.get("nms_required", False)))
        set_onnx_metadata(output, {"contract_version": contract["version"], "nms_required": contract["nms_required"], "names": contract["names"]})
        checks = validate_onnx(output, contract)
        return Execution((artifact(output, "onnx"),), {}, contract, checks)

    def validate(self, request: ValidateRequest, context: WorkflowContext) -> BackendExecution:
        target = require_file(request.target, "model artifact")
        if target.suffix.casefold() == ".onnx":
            contract = detection_contract([])
            checks = validate_onnx(target, contract)
            return Execution((artifact(target, "onnx"),), {}, contract, checks)
        model = self._model(request)
        if request.data:
            values = model.val(data=str(require_file(request.data, "dataset YAML")), split=request.split, device=request.device)
            metrics = getattr(values, "results_dict", {})
        else:
            metrics = {}
        return Execution((artifact(target, "checkpoint"),), metrics, {}, (({"name": "native_validation", "status": "passed"}),))

    def test(self, request: TestRequest, context: WorkflowContext) -> BackendExecution:
        model = self._model(request)
        values = model.val(data=str(require_file(request.data, "dataset YAML")), split=request.split, device=request.device)
        return Execution((artifact(require_file(request.target, "model artifact"), "model"),), getattr(values, "results_dict", {}), {}, ())
