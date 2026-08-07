from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain.errors import ConfigurationError
from ..domain.results import ArtifactRef
from ..workflows.context import WorkflowContext, optional_import
from ..workflows.requests import ResolvedExportRequest, ResolvedTestRequest, ResolvedTrainRequest, ResolvedValidateRequest
from ..workflows.runs import artifact
from .base import BackendExecution
from .common import (
    artifacts_for,
    class_names_from_model,
    classification_contract,
    detection_contract,
    embedded_output_paths,
    metadata_for_contract,
    require_file,
    set_onnx_metadata,
    standardize_detection_core,
    validate_onnx,
    wrap_embedded_variants,
)
from .export_validation import native_validation_metrics, validate_classification_wrappers, validate_detection_wrappers


@dataclass(frozen=True)
class Execution(BackendExecution):
    artifacts: tuple[ArtifactRef, ...] = ()
    metrics: dict[str, Any] = None  # type: ignore[assignment]
    contract: dict[str, Any] = None  # type: ignore[assignment]
    checks: tuple[dict[str, Any], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "metrics", self.metrics or {})
        object.__setattr__(self, "contract", self.contract or {})


class UltralyticsBackend:
    def __init__(self, task: str = "detection"):
        if task not in {"classification", "detection"}:
            raise ValueError(f"unsupported Ultralytics task: {task}")
        self._task = task

    def _data(self, request: ResolvedTrainRequest | ResolvedValidateRequest | ResolvedTestRequest) -> Path:
        if request.data is None:
            raise ConfigurationError("this operation requires a dataset")
        data = request.data.expanduser().resolve()
        if self._task == "classification":
            if not data.is_dir():
                raise ConfigurationError(f"classification dataset directory does not exist: {data}")
            return data
        return require_file(data, "YOLO dataset YAML")

    def _model(self, request: ResolvedTrainRequest | ResolvedExportRequest | ResolvedValidateRequest | ResolvedTestRequest):
        module = optional_import("ultralytics")
        checkpoint = getattr(request, "weights", None) or getattr(request, "checkpoint", None) or getattr(request, "target", None)
        if checkpoint is None:
            checkpoint = Path(request.model.native_id)
        return module.YOLO(str(checkpoint))

    def train(self, request: ResolvedTrainRequest, context: WorkflowContext) -> BackendExecution:
        data = self._data(request)
        model = self._model(request)
        options: dict[str, Any] = {
            "data": str(data), "epochs": request.epochs,
            "imgsz": request.image_size or (224 if self._task == "classification" else 640),
            "batch": request.batch, "lr0": request.learning_rate, "workers": max(0, request.workers),
            "patience": request.patience, "seed": request.seed, "project": str(context.run_dir.parent),
            "name": context.run_dir.name, "exist_ok": True, "resume": request.resume,
            "pretrained": request.pretrained, "deterministic": request.deterministic,
            "amp": request.amp, "compile": request.compile,
        }
        if request.device != "auto":
            options["device"] = request.device
        if self._task == "classification":
            options["dropout"] = request.dropout
        else:
            options["mosaic"] = request.mosaic
        context.emit("backend_train_started", {"framework": "ultralytics", "task": self._task, "data": str(data)})
        model.train(**options)
        weights = context.run_dir / "weights"
        best, last = weights / "best.pt", weights / "last.pt"
        return Execution(artifacts_for([(best, "checkpoint"), (last, "checkpoint")]))

    def export(self, request: ResolvedExportRequest, context: WorkflowContext) -> BackendExecution:
        checkpoint = require_file(request.checkpoint, "checkpoint")
        model = self._model(request)
        nms_required = bool(request.options.get("nms_required", False))
        if self._task == "detection" and nms_required:
            raise ConfigurationError(
                "Ultralytics raw detection output is not the contract's [B,Q,6] "
                "candidate layout; export with embedded NMS (nms_required=false)"
            )
        options: dict[str, Any] = {
            "format": "onnx",
            "imgsz": request.image_size,
            "opset": request.opset,
            "simplify": request.simplify,
            "dynamic": request.batch_size is None,
            "batch": request.batch_size or 1,
        }
        if self._task == "detection":
            options["nms"] = not nms_required
            if not nms_required:
                options["conf"] = 0.0
                options["iou"] = 0.7
        if request.device != "auto":
            options["device"] = request.device
        exported = Path(model.export(**options))
        names = class_names_from_model(model)
        if not names:
            raise ValueError(f"the exported {self._task} model does not expose class names")
        if self._task == "detection":
            exported = standardize_detection_core(
                exported,
                context.run_dir / "core-detection-contract.onnx",
                image_size=request.image_size,
                source_box_format="xyxy",
                source_box_space="pixels",
            )
        outputs = embedded_output_paths(request.output)
        paths = wrap_embedded_variants(
            exported,
            outputs,
            image_size=request.image_size,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            batch_size=request.batch_size,
            apply_softmax=self._task == "classification",
            output_names=("probabilities",) if self._task == "classification" else None,
        )
        contract = (
            classification_contract(names, batch_size=request.batch_size)
            if self._task == "classification"
            else detection_contract(
                names, nms_required=nms_required, batch_size=request.batch_size
            )
        )
        checks: list[dict[str, Any]] = []
        for variant, path in outputs.items():
            variant_contract = (
                classification_contract(
                    names, input_variant=variant, batch_size=request.batch_size
                )
                if self._task == "classification"
                else detection_contract(
                    names,
                    nms_required=nms_required,
                    input_variant=variant,
                    batch_size=request.batch_size,
                )
            )
            set_onnx_metadata(path, metadata_for_contract(variant_contract))
            checks.extend({**check, "variant": variant} for check in validate_onnx(path, variant_contract))

        metrics: dict[str, Any] = {}
        artifacts = [artifact(path, "onnx") for path in paths]
        if request.data is not None:
            data = self._data(request)
            native = model.val(data=str(data), split="val", device=request.device)
            metrics.update(native_validation_metrics(native))
            if self._task == "classification":
                metrics.update(validate_classification_wrappers(
                    outputs,
                    data,
                    classes=names,
                    image_size=request.image_size,
                    batch_size=request.batch_size,
                ))
            else:
                metrics.update(validate_detection_wrappers(
                    outputs,
                    data,
                    class_count=len(names),
                    image_size=request.image_size,
                    batch_size=request.batch_size,
                ))
            report = context.write_json("dataset-validation.json", metrics)
            artifacts.append(artifact(report, "report"))
            checks.append({"name": "checkpoint_native_validation", "status": "passed", "split": "val"})
            for variant in ("bw8", "c24"):
                checks.append({
                    "name": "wrapped_dataset_validation",
                    "status": "passed",
                    "variant": variant,
                    "split": "val",
                    "images": metrics[f"{variant}_images"] if self._task == "classification" else metrics["validation_images"],
                })
        return Execution(tuple(artifacts), metrics, contract, tuple(checks))

    def validate(self, request: ResolvedValidateRequest, context: WorkflowContext) -> BackendExecution:
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

    def test(self, request: ResolvedTestRequest, context: WorkflowContext) -> BackendExecution:
        model = self._model(request)
        values = model.val(data=str(self._data(request)), split=request.split, device=request.device)
        return Execution((artifact(require_file(request.target, "model artifact"), "model"),), getattr(values, "results_dict", {}), {}, ())
