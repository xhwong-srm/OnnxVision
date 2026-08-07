from __future__ import annotations

import logging
import json
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from ..backends.registry import plugin_for
from ..domain.errors import ConfigurationError, ValidationFailedError
from ..domain.models import ModelInfo, Operation, ParameterContext
from ..domain.results import ExportResult, RunStatus, TestResult, TrainResult, ValidationResult
from .requests import (
    ExportRequest,
    ResolvedExportRequest,
    ResolvedTestRequest,
    ResolvedTrainRequest,
    ResolvedValidateRequest,
    TestRequest,
    TrainRequest,
    ValidateRequest,
)
from .runs import RunStore


logger = logging.getLogger(__name__)


def _dependency_versions(value: str | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for package in (value or "").split(","):
        package = package.strip()
        if not package:
            continue
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _resolve(request, operation: Operation):
    plugin = plugin_for(request.selection)
    try:
        handler = plugin.handlers[operation]
    except KeyError as error:
        raise ConfigurationError(f"{request.selection} does not support {operation.value}") from error
    model = plugin.catalog.resolve(request.selection.model)
    schema = handler.schema(model)
    effective = schema.resolve(request.parameters, ParameterContext(request.selection, model, request))
    if operation is Operation.TRAIN:
        resolved = ResolvedTrainRequest(request.selection, model, request.data, request.output, request.weights, request.resume, request.overwrite, effective)
    elif operation is Operation.EXPORT:
        resolved = ResolvedExportRequest(request.selection, model, request.checkpoint, request.output, request.data, effective)
    elif operation is Operation.VALIDATE:
        resolved = ResolvedValidateRequest(request.selection, model, request.target, request.data, request.split, effective)
    else:
        resolved = ResolvedTestRequest(request.selection, model, request.target, request.data, request.split, effective)
    config = {
        "selection": {
            "task": request.selection.task.value,
            "framework": request.selection.framework,
            "model": request.selection.model,
        },
        "resolved_model": {
            "id": model.id,
            "native_id": model.native_id,
            "metadata": dict(model.metadata),
        },
        "operation": operation.value,
        "inputs": {
            name: getattr(request, name)
            for name in ("data", "output", "checkpoint", "target", "weights", "split", "resume", "overwrite")
            if hasattr(request, name)
        },
        "parameters": {
            "requested": dict(request.parameters),
            "effective": effective,
            "schema": schema.describe(),
        },
        "provider": {
            "description": plugin.descriptor.description,
            "dependencies": _dependency_versions(plugin.descriptor.optional_dependency),
        },
    }
    return handler, resolved, config


class TrainService:
    def run(self, request: TrainRequest) -> TrainResult:
        if request.resume and request.overwrite and request.weights is None:
            raise ConfigurationError("--resume cannot use --overwrite without --weights; overwriting would delete last.pt")
        handler, resolved, config = _resolve(request, Operation.TRAIN)
        store = RunStore(request.output.parent)
        context, run_id = store.start("train", config, device=resolved.device, run_dir=request.output, overwrite=request.overwrite, allow_existing=request.resume)
        try:
            execution = handler.execute(resolved, context)
        except Exception as error:
            run = store.finish(run_id, "train", config, status=RunStatus.FAILED, error=str(error))
            context.emit("run_failed", {"error": str(error)})
            raise
        run = store.finish(run_id, "train", config, status=RunStatus.SUCCEEDED, artifacts=execution.artifacts, metrics=dict(execution.metrics))
        context.emit("run_finished", {"status": "succeeded"})
        saved_paths = [
            run.run_dir / name
            for name in ("config.json", "metrics.json", "manifest.json", "events.jsonl")
        ]
        saved_paths.extend(artifact.path for artifact in run.artifacts)
        saved_paths = list(dict.fromkeys(path for path in saved_paths if path.is_file()))
        saved_names = []
        for path in saved_paths:
            try:
                saved_names.append(str(path.relative_to(run.run_dir)))
            except ValueError:
                saved_names.append(str(path))
        logger.info(
            "Training outputs saved under %s: %s",
            run.run_dir,
            ", ".join(saved_names),
        )
        best = next((item for item in execution.artifacts if item.name == "best.pt"), None)
        last = next((item for item in execution.artifacts if item.name == "last.pt"), None)
        return TrainResult(run, best, last, dict(execution.metrics))


class ExportService:
    def run(self, request: ExportRequest) -> ExportResult:
        handler, resolved, config = _resolve(request, Operation.EXPORT)
        store = RunStore(request.output.parent / ".seal-runs")
        context, run_id = store.start("export", config, device=resolved.device)
        try:
            execution = handler.execute(resolved, context)
        except Exception as error:
            store.finish(run_id, "export", config, status=RunStatus.FAILED, error=str(error))
            context.emit("run_failed", {"error": str(error)})
            raise
        if any(check.get("status") == "failed" for check in execution.checks):
            error = "exported model failed its ONNX contract checks"
            store.finish(run_id, "export", config, status=RunStatus.FAILED, error=error)
            context.emit("run_failed", {"error": error})
            raise ValidationFailedError(f"{error}; report: {context.run_dir / 'manifest.json'}")
        run = store.finish(run_id, "export", config, status=RunStatus.SUCCEEDED, artifacts=execution.artifacts, metrics=dict(execution.metrics))
        result = ExportResult(run, dict(execution.contract), {"checks": execution.checks})
        result_path = request.output.expanduser().absolute().with_suffix(".json")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(asdict(result), indent=2, default=str) + "\n", encoding="utf-8")
        logger.info("Export result saved to %s", result_path)
        return result


class ValidationService:
    def run(self, request: ValidateRequest) -> ValidationResult:
        handler, resolved, config = _resolve(request, Operation.VALIDATE)
        store = RunStore(request.target.parent / ".seal-runs")
        context, run_id = store.start("validate", config, device=resolved.device)
        try:
            execution = handler.execute(resolved, context)
        except Exception as error:
            store.finish(run_id, "validate", config, status=RunStatus.FAILED, error=str(error))
            raise
        valid = not any(check.get("status") == "failed" for check in execution.checks)
        run = store.finish(run_id, "validate", config, status=RunStatus.SUCCEEDED if valid else RunStatus.FAILED, artifacts=execution.artifacts, metrics=dict(execution.metrics))
        if not valid:
            raise ValidationFailedError(f"model validation failed; report: {run.run_dir / 'manifest.json'}")
        return ValidationResult(run, valid, execution.checks)


class TestService:
    def run(self, request: TestRequest) -> TestResult:
        handler, resolved, config = _resolve(request, Operation.TEST)
        store = RunStore(request.target.parent / ".seal-runs")
        context, run_id = store.start("test", config, device=resolved.device)
        try:
            execution = handler.execute(resolved, context)
        except Exception as error:
            store.finish(run_id, "test", config, status=RunStatus.FAILED, error=str(error))
            raise
        run = store.finish(run_id, "test", config, status=RunStatus.SUCCEEDED, artifacts=execution.artifacts, metrics=dict(execution.metrics))
        return TestResult(run, request.split, dict(execution.metrics))
