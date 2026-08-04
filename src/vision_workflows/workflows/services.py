from __future__ import annotations

from dataclasses import asdict

from ..backends.registry import backend_for
from ..domain.errors import ValidationFailedError
from ..domain.results import ExportResult, RunStatus, TestResult, TrainResult, ValidationResult
from .requests import ExportRequest, TestRequest, TrainRequest, ValidateRequest
from .runs import RunStore


def _request_config(request) -> dict:
    value = asdict(request)
    value["model"] = str(request.model)
    return value


class TrainService:
    def run(self, request: TrainRequest) -> TrainResult:
        backend = backend_for(request.model)
        config = _request_config(request)
        store = RunStore(request.output.parent)
        context, run_id = store.start("train", config, device=request.device, run_dir=request.output)
        try:
            execution = backend.train(request, context)
        except Exception as error:
            run = store.finish(run_id, "train", config, status=RunStatus.FAILED, error=str(error))
            context.emit("run_failed", {"error": str(error)})
            raise
        run = store.finish(run_id, "train", config, status=RunStatus.SUCCEEDED, artifacts=execution.artifacts, metrics=execution.metrics)
        context.emit("run_finished", {"status": "succeeded"})
        best = next((item for item in execution.artifacts if item.name == "best.pt"), None)
        last = next((item for item in execution.artifacts if item.name == "last.pt"), None)
        return TrainResult(run, best, last, execution.metrics)


class ExportService:
    def run(self, request: ExportRequest) -> ExportResult:
        backend = backend_for(request.model)
        config = _request_config(request)
        store = RunStore(request.output.parent / ".seal-runs")
        context, run_id = store.start("export", config, device=request.device)
        try:
            execution = backend.export(request, context)
        except Exception as error:
            store.finish(run_id, "export", config, status=RunStatus.FAILED, error=str(error))
            context.emit("run_failed", {"error": str(error)})
            raise
        run = store.finish(run_id, "export", config, status=RunStatus.SUCCEEDED, artifacts=execution.artifacts, metrics=execution.metrics)
        return ExportResult(run, execution.artifacts, execution.contract, {"checks": execution.checks, "metrics": execution.metrics})


class ValidationService:
    def run(self, request: ValidateRequest) -> ValidationResult:
        backend = backend_for(request.model)
        config = _request_config(request)
        store = RunStore(request.target.parent / ".seal-runs")
        context, run_id = store.start("validate", config, device=request.device)
        try:
            execution = backend.validate(request, context)
        except Exception as error:
            store.finish(run_id, "validate", config, status=RunStatus.FAILED, error=str(error))
            raise
        valid = not any(check.get("status") == "failed" for check in execution.checks)
        run = store.finish(run_id, "validate", config, status=RunStatus.SUCCEEDED if valid else RunStatus.FAILED, artifacts=execution.artifacts, metrics=execution.metrics)
        if not valid:
            raise ValidationFailedError(f"model validation failed; report: {run.run_dir / 'manifest.json'}")
        return ValidationResult(run, valid, execution.checks)


class TestService:
    def run(self, request: TestRequest) -> TestResult:
        backend = backend_for(request.model)
        config = _request_config(request)
        store = RunStore(request.target.parent / ".seal-runs")
        context, run_id = store.start("test", config, device=request.device)
        try:
            execution = backend.test(request, context)
        except Exception as error:
            store.finish(run_id, "test", config, status=RunStatus.FAILED, error=str(error))
            raise
        run = store.finish(run_id, "test", config, status=RunStatus.SUCCEEDED, artifacts=execution.artifacts, metrics=execution.metrics)
        return TestResult(run, request.split, execution.metrics)
