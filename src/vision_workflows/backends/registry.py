from __future__ import annotations

from ..domain.datasets import DatasetFormat, TaskKind
from ..domain.errors import ConfigurationError
from ..domain.models import (
    DatasetRequirement,
    ModelInfo,
    ModelSelection,
    Operation,
    ParameterOrigin,
    ParameterSchema,
    ParameterSpec,
    ProviderDescriptor,
    StaticModelCatalog,
)
from .base import FrameworkTaskPlugin, OperationHandler
from .libreyolo import LibreYoloBackend
from .timm_classification import TimmClassificationBackend
from .ultralytics import UltralyticsBackend


def _schema(*parameters: ParameterSpec) -> ParameterSchema:
    return ParameterSchema(parameters)


def _training(image_size: int | None) -> ParameterSchema:
    return _schema(
        ParameterSpec("epochs", int, "number of training epochs", 100, minimum=1),
        ParameterSpec("batch", int, "training batch size", 16, minimum=1),
        ParameterSpec("image_size", int, "square input image size", image_size, allow_none=image_size is None, minimum=1),
        ParameterSpec("learning_rate", float, "initial learning rate", 1e-3, minimum=0.0),
        ParameterSpec("workers", int, "data-loader worker processes", 0, minimum=0),
        ParameterSpec("patience", int, "early-stopping patience", 20, minimum=0),
        ParameterSpec("seed", int, "random seed", 42, minimum=0),
        ParameterSpec("device", str, "training device", "auto"),
        ParameterSpec("pretrained", bool, "start from pretrained weights", True),
        ParameterSpec("deterministic", bool, "request deterministic training", True),
    )


def _export(image_size: int, *, nms_configurable: bool = False) -> ParameterSchema:
    base = _schema(
        ParameterSpec("image_size", int, "square export image size", image_size, minimum=1),
        ParameterSpec(
            "batch_size",
            int,
            "fixed ONNX batch size; omit for a dynamic batch axis",
            None,
            allow_none=True,
            minimum=1,
        ),
        ParameterSpec("opset", int, "ONNX operator-set version", 18, minimum=7),
        ParameterSpec("simplify", bool, "simplify the exported ONNX graph", True),
        ParameterSpec("device", str, "export device", "auto"),
    )
    if not nms_configurable:
        return base
    return base.compose(_schema(ParameterSpec("nms_required", bool, "mark post-export NMS as required", False)).with_origin(ParameterOrigin.TASK))


def _evaluation(device: str) -> ParameterSchema:
    return _schema(ParameterSpec("device", str, "evaluation device", device))


def _timm_train(_: ModelInfo) -> ParameterSchema:
    framework = _schema(
        ParameterSpec("validate_every", int, "run validation every N epochs", 1, minimum=1),
        ParameterSpec("prefetch_factor", int, "batches prefetched per worker", None, allow_none=True, minimum=1),
        ParameterSpec("persistent_workers", bool, "keep workers alive between epochs", False),
        ParameterSpec("pin_memory", bool, "pin data-loader memory", False),
        ParameterSpec("amp", bool, "use automatic mixed precision", False),
        ParameterSpec("amp_dtype", str, "AMP data type", None, choices=("float16", "bfloat16"), allow_none=True),
        ParameterSpec("compile", bool, "compile the model with torch.compile", False),
    ).with_origin(ParameterOrigin.FRAMEWORK)
    return _training(None).compose(framework)


def _ultralytics_train(task: TaskKind):
    def factory(_: ModelInfo) -> ParameterSchema:
        task_schema = (
            _schema(ParameterSpec("dropout", float, "classifier dropout", 0.0, minimum=0.0, maximum=1.0))
            if task is TaskKind.CLASSIFICATION
            else _schema(ParameterSpec("mosaic", float, "mosaic augmentation probability", 1.0, minimum=0.0, maximum=1.0))
        ).with_origin(ParameterOrigin.TASK)
        framework = _schema(
            ParameterSpec("amp", bool, "use automatic mixed precision", True),
            ParameterSpec("compile", bool, "compile the model", False),
        ).with_origin(ParameterOrigin.FRAMEWORK)
        return _training(224 if task is TaskKind.CLASSIFICATION else 640).compose(framework, task_schema)
    return factory


def _libre_train(_: ModelInfo) -> ParameterSchema:
    framework = _schema(
        ParameterSpec("amp", bool, "use automatic mixed precision", True),
        ParameterSpec(
            "cache",
            str,
            "decoded image cache mode",
            "none",
            choices=("none", "ram", "disk"),
        ),
    ).with_origin(ParameterOrigin.FRAMEWORK)
    return _training(640).compose(framework)


def _libre_export(model: ModelInfo) -> ParameterSchema:
    native_sizes = {"s": 320, "m": 416, "l": 640}
    image_size = native_sizes.get(str(model.metadata.get("variant")), 640)
    return _export(image_size, nms_configurable=True)


def _all_handlers(backend, train_schema, export_size: int, dataset: DatasetRequirement, *, nms_configurable: bool = False):
    return {
        Operation.TRAIN: OperationHandler(train_schema, backend.train, dataset),
        Operation.EXPORT: OperationHandler(lambda _: _export(export_size, nms_configurable=nms_configurable), backend.export),
        Operation.VALIDATE: OperationHandler(lambda _: _evaluation("cpu"), backend.validate),
        Operation.TEST: OperationHandler(lambda _: _evaluation("auto"), backend.test, dataset),
    }


_TIMM_CLASSIFICATION_MODELS = (
    ModelInfo("mobilenetv4_conv_small_050.e3000_r224_in1k", "mobilenetv4_conv_small_050.e3000_r224_in1k"),
)


def _plugins() -> tuple[FrameworkTaskPlugin, ...]:
    classification_data = DatasetRequirement((DatasetFormat.IMAGE_FOLDER,), ("train", "val"))
    yolo_data = DatasetRequirement((DatasetFormat.YOLO,), ("train", "val"))

    timm_backend = TimmClassificationBackend()
    ultralytics_cls = UltralyticsBackend("classification")
    ultralytics_det = UltralyticsBackend("detection")
    yolo26 = tuple(ModelInfo(f"yolo26{size}", f"yolo26{size}.pt", metadata={"variant": size}) for size in "nsmlx")
    yolo26_cls = tuple(ModelInfo(item.id, f"{item.id}-cls.pt", metadata=item.metadata) for item in yolo26)
    libre_models = tuple(
        ModelInfo(f"{family}{size}", f"{family}{size}", metadata={"family": family, "variant": size})
        for family, sizes in (("yolov9", "tsmc"), ("picodet", "sml")) for size in sizes
    )

    def libre_backend(operation: Operation):
        def execute(request, context):
            backend = LibreYoloBackend(str(request.model.metadata["family"]), (request.model.variant,))
            return getattr(backend, operation.value)(request, context)

        return execute

    libre_handlers = {
        operation: OperationHandler(schema, libre_backend(operation), yolo_data if operation in {Operation.TRAIN, Operation.TEST} else None)
        for operation, schema in (
            (Operation.TRAIN, _libre_train),
            (Operation.EXPORT, _libre_export),
            (Operation.VALIDATE, lambda _: _evaluation("cpu")),
            (Operation.TEST, lambda _: _evaluation("auto")),
        )
    }
    return (
        FrameworkTaskPlugin(ProviderDescriptor("timm", TaskKind.CLASSIFICATION, frozenset(Operation), "timm image classifier", "timm"), StaticModelCatalog(_TIMM_CLASSIFICATION_MODELS), _all_handlers(timm_backend, _timm_train, 224, classification_data)),
        FrameworkTaskPlugin(ProviderDescriptor("ultralytics", TaskKind.CLASSIFICATION, frozenset(Operation), "Ultralytics classifier", "ultralytics"), StaticModelCatalog(yolo26_cls), _all_handlers(ultralytics_cls, _ultralytics_train(TaskKind.CLASSIFICATION), 224, classification_data)),
        FrameworkTaskPlugin(ProviderDescriptor("ultralytics", TaskKind.OBJECT_DETECTION, frozenset(Operation), "Ultralytics detector", "ultralytics"), StaticModelCatalog(yolo26), _all_handlers(ultralytics_det, _ultralytics_train(TaskKind.OBJECT_DETECTION), 640, yolo_data, nms_configurable=True)),
        FrameworkTaskPlugin(ProviderDescriptor("libreyolo", TaskKind.OBJECT_DETECTION, frozenset(Operation), "LibreYOLO detector", "libreyolo"), StaticModelCatalog(libre_models), libre_handlers),
    )


_PLUGINS = _plugins()


def plugins() -> tuple[FrameworkTaskPlugin, ...]:
    return _PLUGINS


def plugin_for(selection: ModelSelection) -> FrameworkTaskPlugin:
    for plugin in _PLUGINS:
        if plugin.descriptor.framework == selection.framework and plugin.descriptor.task is selection.task:
            return plugin
    raise ConfigurationError(f"unsupported framework/task: {selection.framework}/{selection.task.value}")


def frameworks() -> tuple[ProviderDescriptor, ...]:
    return tuple(plugin.descriptor for plugin in _PLUGINS)


def models_for(task: TaskKind, framework: str, pattern: str | None = None) -> tuple[ModelInfo, ...]:
    plugin = next((item for item in _PLUGINS if item.descriptor.framework == framework and item.descriptor.task is task), None)
    if plugin is None:
        raise ConfigurationError(f"unsupported framework/task: {framework}/{task.value}")
    return plugin.catalog.list(pattern)
