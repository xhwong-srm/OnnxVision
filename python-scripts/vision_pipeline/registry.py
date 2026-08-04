"""Registry and argument translation for the current training/export scripts."""

from __future__ import annotations

from .models import ExportRequest, TrainRequest, WorkflowDescriptor
from .runtime import CommandResult, run_script


_DESCRIPTORS = (
    WorkflowDescriptor(
        "timm", "classification", "classification", "python-scripts/timm/train_timm_classification.py", "train",
        "timm image-folder classifier",
    ),
    WorkflowDescriptor(
        "timm", "detection", "query", "python-scripts/timm/train_timm_detection.py", "train",
        "custom timm NMS-free query detector",
    ),
    WorkflowDescriptor(
        "ultralytics", "detection", "yolo26", "python-scripts/yolo/train_yolo26_detection.py", "train",
        "Ultralytics YOLO26 detector",
    ),
    WorkflowDescriptor(
        "libreyolo", "detection", "yolov9", "python-scripts/libreyolo/train_yolov9_detection.py", "train",
        "LibreYOLO YOLOv9 detector",
    ),
    WorkflowDescriptor(
        "libreyolo", "detection", "picodet", "python-scripts/libreyolo/train_picodet_detection.py", "train",
        "LibreYOLO PicoDet detector",
    ),
)

_EXPORT_SCRIPTS = {
    ("timm", "classification", "classification"): "python-scripts/timm/export_timm_classification.py",
    ("timm", "detection", "query"): "python-scripts/timm/export_timm_detection.py",
    ("ultralytics", "detection", "yolo26"): "python-scripts/yolo/export_yolo26_detection.py",
    ("libreyolo", "detection", "yolov9"): "python-scripts/libreyolo/export_yolov9_detection.py",
    ("libreyolo", "detection", "picodet"): "python-scripts/libreyolo/export_picodet_detection.py",
}


def descriptors() -> tuple[WorkflowDescriptor, ...]:
    return _DESCRIPTORS


def _family(request_backend: str, request_task: str, model: str) -> str:
    value = model.casefold()
    if request_backend == "timm":
        return "classification" if request_task == "classification" else "query"
    if request_backend == "ultralytics":
        return "yolo26"
    if request_backend == "libreyolo":
        if value.startswith("picodet"):
            return "picodet"
        if value.startswith("yolov9"):
            return "yolov9"
    raise ValueError(
        f"Unsupported workflow: backend={request_backend!r}, task={request_task!r}, model={model!r}"
    )


def _descriptor(backend: str, task: str, model: str) -> WorkflowDescriptor:
    family = _family(backend, task, model)
    matches = [item for item in _DESCRIPTORS if (item.backend, item.task, item.model_family) == (backend, task, family)]
    if not matches:
        raise ValueError(f"No registered training workflow for {backend}/{task}/{model}")
    return matches[0]


def _add(arguments: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    arguments.extend((flag, str(value)))


def _bool(arguments: list[str], flag: str, value: bool | None) -> None:
    if value is True:
        arguments.append(flag)
    elif value is False:
        arguments.append(f"--no-{flag.removeprefix('--')}")


def _model_variant(backend: str, family: str, model: str) -> str:
    value = model.strip()
    if backend == "ultralytics" and value.casefold().startswith("yolo26"):
        return value[6:] or "n"
    if backend == "libreyolo":
        prefix = "picodet" if family == "picodet" else "yolov9"
        if value.casefold().startswith(prefix):
            return value[len(prefix):].lstrip("-_") or ("s" if family == "picodet" else "t")
    return value


def _train_arguments(request: TrainRequest) -> tuple[str, ...]:
    family = _family(request.backend, request.task, request.model)
    arguments: list[str] = ["--data", str(request.data)]
    if request.backend == "timm" and request.task == "classification":
        _add(arguments, "--model", request.model)
    elif request.backend == "timm":
        _add(arguments, "--model", request.model)
    else:
        _add(arguments, "--model", _model_variant(request.backend, family, request.model))

    if request.weights is not None:
        _add(arguments, "--weights", request.weights)
    if request.output is not None:
        _add(arguments, "--output", request.output)
    if request.imgsz is not None and not (request.backend == "timm" and request.task == "classification"):
        _add(arguments, "--imgsz" if request.backend == "timm" else "--resolution", request.imgsz)
    for flag, value in (("--epochs", request.epochs), ("--batch", request.batch), ("--lr", request.lr),
                        ("--workers", request.workers), ("--patience", request.patience), ("--seed", request.seed)):
        _add(arguments, flag, value)
    _add(arguments, "--device", request.device)
    if request.backend == "timm" and request.task == "classification":
        _bool(arguments, "--deterministic", request.deterministic)
    elif request.backend == "timm":
        _bool(arguments, "--deterministic", request.deterministic)
        _bool(arguments, "--pretrained", request.pretrained)
        _bool(arguments, "--run-test", request.run_test)
    else:
        _bool(arguments, "--deterministic", request.deterministic)
        _bool(arguments, "--pretrained", request.pretrained)
        _bool(arguments, "--run-test", request.run_test)
    if request.resume:
        arguments.append("--resume")
    arguments.extend(request.backend_args)
    return tuple(arguments)


def train(request: TrainRequest) -> CommandResult:
    descriptor = _descriptor(request.backend, request.task, request.model)
    if descriptor.operation != "train":
        raise ValueError(f"Registered workflow is not trainable: {descriptor}")
    return run_script(descriptor.script, _train_arguments(request))


def _export_arguments(request: ExportRequest) -> tuple[str, ...]:
    arguments: list[str] = ["--checkpoint", str(request.checkpoint)]
    if request.backend == "timm":
        arguments[0:1] = [str(request.checkpoint)]
    if request.output is not None:
        _add(arguments, "--output", request.output)
    if request.imgsz is not None:
        _add(arguments, "--imgsz" if request.backend == "timm" else "--resolution", request.imgsz)
    _add(arguments, "--opset", request.opset)
    _bool(arguments, "--simplify", request.simplify)
    _add(arguments, "--device", request.device)
    if request.backend == "timm":
        if request.embedded_preprocessing is True:
            arguments.append("--embedded-preprocessing")
        if request.data is not None:
            _add(arguments, "--data" if request.task == "detection" else "--dataset", request.data)
    else:
        if request.embedded_preprocessing is False:
            arguments.append("--skip-embedded-preprocessing")
        if request.data is not None:
            if request.backend == "ultralytics":
                arguments.append("--validate-image")
            _add(arguments, "--data", request.data)
        _add(arguments, "--validation-split", request.validation_split)
        _add(arguments, "--validation-limit", request.validation_limit)
        _add(arguments, "--validation-report", request.validation_report)
    arguments.extend(request.backend_args)
    return tuple(arguments)


def export(request: ExportRequest) -> CommandResult:
    family = _family(request.backend, request.task, request.model)
    script = _EXPORT_SCRIPTS.get((request.backend, request.task, family))
    if script is None:
        raise ValueError(f"No registered export workflow for {request.backend}/{request.task}/{request.model}")
    return run_script(script, _export_arguments(request))
