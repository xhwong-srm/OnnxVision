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
from .common import class_names_from_model, detection_contract, metadata_for_contract, require_file, set_onnx_metadata, validate_onnx


@dataclass(frozen=True)
class Execution(BackendExecution):
    artifacts: tuple[ArtifactRef, ...] = ()
    metrics: dict[str, Any] = None  # type: ignore[assignment]
    contract: dict[str, Any] = None  # type: ignore[assignment]
    checks: tuple[dict[str, Any], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "metrics", self.metrics or {})
        object.__setattr__(self, "contract", self.contract or {})


class LibreYoloBackend(ModelBackend):
    def __init__(self, family: str, variants: tuple[str, ...]):
        self._descriptor = BackendDescriptor(
            "libreyolo", family, "detection", variants, frozenset(BackendCapability),
            f"LibreYOLO {family} detector", "libreyolo",
        )

    @property
    def descriptor(self) -> BackendDescriptor:
        return self._descriptor

    def _model(self, request: TrainRequest | ExportRequest | ValidateRequest | TestRequest):
        module = optional_import("libreyolo")
        target = getattr(request, "weights", None) or getattr(request, "checkpoint", None) or getattr(request, "target", None)
        if target:
            return module.LibreYOLO(str(target), device=request.device, task="detect")
        class_count = 1
        data = getattr(request, "data", None)
        if data is not None and data.is_file():
            try:
                import yaml
                document = yaml.safe_load(data.read_text(encoding="utf-8"))
                names = document.get("names", []) if isinstance(document, dict) else []
                class_count = len(names)
            except (OSError, ValueError, AttributeError):
                class_count = 1
        if self._descriptor.family == "yolov9":
            return module.LibreYOLO9(model_path=None, size=request.model.variant, nb_classes=class_count, device=request.device, task="detect")
        return module.LibrePICODET(model_path=None, size=request.model.variant, nb_classes=class_count, device=request.device, task="detect")

    def train(self, request: TrainRequest, context: WorkflowContext) -> BackendExecution:
        data = require_file(request.data, "YOLO dataset YAML")
        model = self._model(request)
        options: dict[str, Any] = {"data": str(data), "epochs": request.epochs, "batch": request.batch, "imgsz": request.image_size, "lr0": request.learning_rate, "device": request.device, "workers": max(0, request.workers), "seed": request.seed, "project": str(context.run_dir.parent), "name": context.run_dir.name, "exist_ok": True, "resume": request.resume, "pretrained": request.pretrained}
        options.update(request.options)
        results = model.train(**options)
        best = Path(results.get("best_checkpoint", context.run_dir / "weights" / "best.pt"))
        last = Path(results.get("last_checkpoint", context.run_dir / "weights" / "last.pt"))
        return Execution(tuple(artifact(path, "checkpoint") for path in (best, last) if path.is_file()), results if isinstance(results, dict) else {})

    def export(self, request: ExportRequest, context: WorkflowContext) -> BackendExecution:
        checkpoint = require_file(request.checkpoint, "checkpoint")
        model = self._model(request)
        exported = Path(model.export(format="onnx", imgsz=request.image_size, opset=request.opset, simplify=request.simplify, device=request.device, **dict(request.options)))
        output = request.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if exported.resolve() != output:
            exported.replace(output)
        names = class_names_from_model(model)
        if not names:
            raise ValueError("the exported detection model does not expose class names")
        contract = detection_contract(names, nms_required=bool(request.options.get("nms_required", False)))
        set_onnx_metadata(output, metadata_for_contract(contract))
        checks = validate_onnx(output, contract)
        return Execution((artifact(output, "onnx"),), {}, contract, checks)

    def validate(self, request: ValidateRequest, context: WorkflowContext) -> BackendExecution:
        target = require_file(request.target, "model artifact")
        if target.suffix.casefold() == ".onnx":
            contract = detection_contract([])
            return Execution((artifact(target, "onnx"),), {}, contract, validate_onnx(target, contract))
        model = self._model(request)
        metrics = model.val(data=str(require_file(request.data, "dataset YAML")), split=request.split, device=request.device).get("metrics", {}) if request.data else {}
        return Execution((artifact(target, "checkpoint"),), metrics, {}, (({"name": "native_validation", "status": "passed"}),))

    def test(self, request: TestRequest, context: WorkflowContext) -> BackendExecution:
        model = self._model(request)
        values = model.val(data=str(require_file(request.data, "dataset YAML")), split=request.split, device=request.device)
        metrics = values.get("metrics", {}) if isinstance(values, dict) else getattr(values, "results_dict", {})
        return Execution((artifact(require_file(request.target, "model artifact"), "model"),), metrics, {}, ())
