from __future__ import annotations

import io
import json
import logging
import shutil
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from ..domain.results import ArtifactRef
from ..workflows.context import WorkflowContext, optional_import
from ..workflows.requests import ResolvedExportRequest, ResolvedTestRequest, ResolvedTrainRequest, ResolvedTuneRequest, ResolvedValidateRequest
from ..workflows.runs import artifact
from .base import BackendExecution
from .common import (
    classification_contract,
    embedded_output_paths,
    metadata_for_contract,
    require_file,
    set_onnx_metadata,
    validate_onnx,
    wrap_embedded_variants,
)
from .timm_training import (
    TimmTrainingOptions,
    capture_rng_state,
    restore_rng_state,
    seed_everything,
    worker_seed,
)
from .export_validation import validate_classification_native_export, validate_classification_wrappers


logger = logging.getLogger(__name__)

_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_STD = (0.229, 0.224, 0.225)


def _nonnegative_int_option(options: dict[str, Any], name: str, default: int) -> int:
    value = options.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a non-negative integer") from error
    if result < 0 or str(value).strip() != str(result):
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _classification_metrics(confusion, classes: list[str]) -> dict[str, Any]:
    true_positive = confusion.diag().float()
    support = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()
    precision = true_positive / predicted.clamp_min(1.0)
    recall = true_positive / support.clamp_min(1.0)
    f1 = 2.0 * true_positive / (support + predicted).clamp_min(1.0)
    present = support > 0
    if bool(present.any().item()):
        macro_precision = float(precision[present].mean().item())
        macro_f1 = float(f1[present].mean().item())
        balanced_accuracy = float(recall[present].mean().item())
    else:
        macro_precision = macro_f1 = balanced_accuracy = 0.0
    return {
        "macro_precision": macro_precision,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "per_class_precision": {name: float(value) for name, value in zip(classes, precision.tolist())},
        "per_class_recall": {name: float(value) for name, value in zip(classes, recall.tolist())},
        "per_class_f1": {name: float(value) for name, value in zip(classes, f1.tolist())},
    }


def _create_scheduler(torch, optimizer, total_epochs: int, warmup_epochs: int):
    if total_epochs < 1:
        raise ValueError("epochs must be at least 1")
    if not 0 <= warmup_epochs < total_epochs:
        raise ValueError("warmup_epochs must be at least 0 and less than epochs")
    cosine_epochs = max(1, total_epochs - warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=0.0,
    )
    if warmup_epochs == 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


class AlbumentationsAugmentation:
    """Pickleable ImageFolder transform for Windows DataLoader workers."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, image):
        numpy = optional_import("numpy")
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        result = self.pipeline(image=numpy.asarray(image))["image"]
        return Image.fromarray(numpy.ascontiguousarray(result))


class ResizedImageLoader:
    """Pickleable ImageFolder loader backed by resized RAM or disk images."""

    _MANIFEST_VERSION = 1

    def __init__(self, mode: str, cache_dir: Path | None, image_size: int, resize):
        if mode not in {"ram", "disk"}:
            raise ValueError(f"unsupported resized image cache mode: {mode}")
        if mode == "disk" and cache_dir is None:
            raise ValueError("disk resized image cache requires a cache directory")
        self.mode = mode
        self.cache_dir = cache_dir
        self.image_size = image_size
        self.resize = resize
        self._paths: tuple[str, ...] = ()
        self._indices: dict[str, int] = {}
        self._ram_images: tuple[Image.Image, ...] = ()
        self._disk_images: tuple[Path, ...] = ()

    @staticmethod
    def _resolved(path: str | Path) -> str:
        return str(Path(path).expanduser().resolve())

    @staticmethod
    def _source_info(paths: tuple[str, ...]) -> list[dict[str, Any]]:
        return [
            {
                "path": path,
                "size": Path(path).stat().st_size,
                "mtime_ns": Path(path).stat().st_mtime_ns,
            }
            for path in paths
        ]

    def _read_resized(self, path: str) -> Image.Image:
        with Image.open(path) as image:
            return self.resize(image.convert("RGB"))

    def prepare(self, paths: tuple[str, ...]) -> None:
        self._paths = tuple(self._resolved(path) for path in paths)
        self._indices = {path: index for index, path in enumerate(self._paths)}
        if self.mode == "ram":
            started = time.perf_counter()
            self._ram_images = tuple(self._read_resized(path) for path in self._paths)
            logger.info(
                "Prepared resized RAM image cache: images=%d size=%d build_seconds=%.1f",
                len(self._ram_images),
                self._ram_bytes(),
                time.perf_counter() - started,
            )
            return

        assert self.cache_dir is not None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.cache_dir / "manifest.json"
        manifest = {
            "version": self._MANIFEST_VERSION,
            "image_size": self.image_size,
            "sources": self._source_info(self._paths),
        }
        self._disk_images = tuple(self.cache_dir / f"{index:06d}.npy" for index in range(len(self._paths)))
        cached = False
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8")) == manifest and all(path.is_file() for path in self._disk_images)
        except (OSError, json.JSONDecodeError):
            cached = False
        started = time.perf_counter()
        if not cached:
            numpy = optional_import("numpy")
            for target, source in zip(self._disk_images, self._paths):
                numpy.save(target, numpy.asarray(self._read_resized(source), dtype=numpy.uint8))
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info(
            "Prepared resized disk image cache: images=%d bytes=%d cached=%s build_seconds=%.1f path=%s",
            len(self._disk_images),
            sum(path.stat().st_size for path in self._disk_images),
            cached,
            time.perf_counter() - started,
            self.cache_dir,
        )

    def _ram_bytes(self) -> int:
        return sum(image.width * image.height * len(image.getbands()) for image in self._ram_images)

    def __call__(self, path: str) -> Image.Image:
        index = self._indices.get(self._resolved(path))
        if index is None:
            raise KeyError(f"image is not present in resized cache: {path}")
        if self.mode == "ram":
            return self._ram_images[index]
        numpy = optional_import("numpy")
        return Image.fromarray(numpy.load(self._disk_images[index], allow_pickle=False))


def _positive_int_option(options: dict[str, Any], name: str, default: int) -> int:
    value = options.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result <= 0 or str(value).strip() != str(result):
        raise ValueError(f"{name} must be a positive integer")
    return result


@dataclass(frozen=True)
class Execution(BackendExecution):
    artifacts: tuple[ArtifactRef, ...] = ()
    metrics: dict[str, Any] = None  # type: ignore[assignment]
    contract: dict[str, Any] = None  # type: ignore[assignment]
    checks: tuple[dict[str, Any], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "metrics", self.metrics or {})
        object.__setattr__(self, "contract", self.contract or {})


class TimmClassificationBackend:

    @staticmethod
    def _imports():
        timm = optional_import("timm")
        torch = optional_import("torch")
        torchvision = optional_import("torchvision")
        return timm, torch, torchvision

    @staticmethod
    def _albumentations_augmentation(train: bool, enabled: bool, policy: str):
        if not train or not enabled:
            return None
        if policy not in {"standard", "robust"}:
            raise ValueError(f"unsupported augmentation policy: {policy}")
        albumentations = optional_import("albumentations")
        operations = [
            albumentations.HorizontalFlip(p=0.5),
            albumentations.Rotate(limit=5, p=0.5),
        ]
        if policy == "robust":
            operations.extend([
                albumentations.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.55,
                ),
                albumentations.RandomGamma(gamma_limit=(80, 120), p=0.25),
                albumentations.GaussNoise(std_range=(0.01, 0.03), p=0.15),
                albumentations.GaussianBlur(
                    blur_limit=(3, 5),
                    sigma_limit=(0.1, 1.2),
                    p=0.1,
                ),
            ])
        pipeline = albumentations.Compose(operations)
        return AlbumentationsAugmentation(pipeline)

    @staticmethod
    def _datasets(
        data: Path,
        image_size: int,
        train: bool,
        mean: tuple[float, ...] = _DEFAULT_MEAN,
        std: tuple[float, ...] = _DEFAULT_STD,
        augmentation_enabled: bool = True,
        augmentation_backend: str = "torchvision",
        augmentation_policy: str = "standard",
        cache_mode: str = "none",
    ):
        _, _, torchvision = TimmClassificationBackend._imports()
        if augmentation_backend not in {"torchvision", "albumentations"}:
            raise ValueError(f"unsupported augmentation backend: {augmentation_backend}")
        if augmentation_policy not in {"standard", "robust"}:
            raise ValueError(f"unsupported augmentation policy: {augmentation_policy}")
        if cache_mode not in {"none", "ram", "disk"}:
            raise ValueError(f"unsupported resized image cache mode: {cache_mode}")
        transforms = torchvision.transforms
        resize = transforms.Resize((image_size, image_size))
        cache_loader = None
        split = "train" if train else "val"
        if cache_mode != "none":
            cache_dir = data / ".vision_workflows_cache" / "timm_classification" / f"size-{image_size}" / split
            cache_loader = ResizedImageLoader(cache_mode, cache_dir if cache_mode == "disk" else None, image_size, resize)
        operations = [] if cache_loader is not None else [resize]
        if train and augmentation_enabled:
            if augmentation_backend == "albumentations":
                operations.append(TimmClassificationBackend._albumentations_augmentation(True, True, augmentation_policy))
            else:
                operations.extend([transforms.RandomHorizontalFlip(), transforms.RandomRotation(5)])
                if augmentation_policy == "robust":
                    operations.extend([
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.1),
                    ])
        operations.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
        transform = transforms.Compose(operations)
        root = data / ("train" if train else "val")
        if not root.is_dir():
            raise FileNotFoundError(f"classification split does not exist: {root}")
        dataset_options = {"transform": transform}
        if cache_loader is not None:
            dataset_options["loader"] = cache_loader
        dataset = torchvision.datasets.ImageFolder(str(root), **dataset_options)
        if cache_loader is not None:
            cache_loader.prepare(tuple(path for path, _ in dataset.samples))
        return dataset

    @staticmethod
    def _dataset_classes(data: Path, torchvision) -> tuple[list[str], list[str]]:
        roots = (data / "train", data / "val")
        for root in roots:
            if not root.is_dir():
                raise FileNotFoundError(f"classification split does not exist: {root}")
        return (
            torchvision.datasets.ImageFolder(str(roots[0])).classes,
            torchvision.datasets.ImageFolder(str(roots[1])).classes,
        )

    @staticmethod
    def _model_data_config(timm, model) -> dict[str, Any]:
        try:
            return dict(timm.data.resolve_model_data_config(model))
        except (AttributeError, KeyError, TypeError):
            return dict(getattr(model, "pretrained_cfg", {}) or getattr(model, "default_cfg", {}) or {})

    def train(self, request: ResolvedTrainRequest, context: WorkflowContext) -> BackendExecution:
        return self._train(request, context)

    def _train(self, request: ResolvedTrainRequest, context: WorkflowContext, trial=None) -> BackendExecution:
        started = time.perf_counter()
        optuna = optional_import("optuna") if trial is not None else None
        logger.info(
            "Training parameters: model=%s data=%s output=%s device=%s epochs=%d batch=%d image_size=%s learning_rate=%g workers=%d seed=%d pretrained=%s resume=%s patience=%d deterministic=%s weights=%s options=%s",
            request.model,
            request.data,
            context.run_dir,
            request.device,
            request.epochs,
            request.batch,
            request.image_size,
            request.learning_rate,
            request.workers,
            request.seed,
            request.pretrained,
            request.resume,
            request.patience,
            request.deterministic,
            request.weights,
            dict(request.options),
        )
        logger.info("Loading timm, torch, and torchvision")
        timm, torch, torchvision = self._imports()
        tqdm = optional_import("tqdm.auto").tqdm
        data = request.data.expanduser().resolve()
        logger.info("Loading image-folder datasets from %s", data)
        train_classes, val_classes = self._dataset_classes(data, torchvision)
        if train_classes != val_classes:
            raise ValueError("train and val class folders differ")
        seed_everything(torch, request.seed, request.deterministic)
        device = torch.device("cuda" if request.device in {"auto", "cuda"} and torch.cuda.is_available() else request.device if request.device != "auto" else "cpu")
        logger.info("Using device: %s", device)
        model_pretrained = request.pretrained and not request.resume and not request.weights
        logger.info("Creating timm model %s (pretrained=%s)", request.model.variant, model_pretrained)
        model = timm.create_model(request.model.variant, pretrained=model_pretrained, num_classes=len(train_classes))
        logger.info("Model ready")
        model_data_config = self._model_data_config(timm, model)
        input_size = model_data_config.get("input_size", (3, 224, 224))
        image_size = int(request.image_size or input_size[-1])
        mean = tuple(float(value) for value in model_data_config.get("mean", _DEFAULT_MEAN))
        std = tuple(float(value) for value in model_data_config.get("std", _DEFAULT_STD))
        logger.info("Effective image size: %d (requested=%s, model_suggested=%s)", image_size, request.image_size, input_size[-1])
        augmentation_enabled = bool(request.options.get("augmentation", True))
        augmentation_backend = str(request.options.get("augmentation_backend", "torchvision"))
        augmentation_policy = str(request.options.get("augmentation_policy", "standard"))
        cache_mode = str(request.options.get("cache", "none"))
        val_workers = int(request.options.get("val_workers", 0))
        if val_workers < 0:
            raise ValueError("val_workers must be non-negative")
        train_set = self._datasets(
            data,
            image_size,
            True,
            mean,
            std,
            augmentation_enabled,
            augmentation_backend,
            augmentation_policy,
            cache_mode,
        )
        val_set = self._datasets(
            data,
            image_size,
            False,
            mean,
            std,
            False,
            augmentation_backend,
            augmentation_policy,
            cache_mode,
        )
        logger.info("Dataset ready: train=%d val=%d classes=%s cache=%s", len(train_set), len(val_set), ", ".join(train_set.classes), cache_mode)
        if request.weights and not request.resume:
            logger.info("Loading custom weights from %s", request.weights)
            checkpoint = torch.load(require_file(request.weights, "weights"), map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device)
        requested_workers = max(0, request.workers)
        workers = requested_workers
        training_options = TimmTrainingOptions.from_mapping(request.options)
        loader_generator = torch.Generator().manual_seed(request.seed)
        worker_init_fn = worker_seed if workers > 0 else None
        loader_options = training_options.data_loader_kwargs(workers)
        val_worker_init_fn = worker_seed if val_workers > 0 else None
        val_loader_options = (
            training_options.data_loader_kwargs(val_workers)
            if val_workers > 0
            else {"num_workers": 0, "pin_memory": training_options.pin_memory}
        )
        validation_interval = _positive_int_option(dict(request.options), "validate_every", 1)
        loader = torch.utils.data.DataLoader(train_set, batch_size=request.batch, shuffle=True, generator=loader_generator, worker_init_fn=worker_init_fn, **loader_options)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=request.batch, shuffle=False, worker_init_fn=val_worker_init_fn, **val_loader_options)
        logger.info("DataLoaders ready: batch=%d train_workers=%d val_workers=%d validate_every=%d", request.batch, workers, val_workers, validation_interval)
        optimizer = torch.optim.AdamW(model.parameters(), lr=request.learning_rate, weight_decay=float(request.options.get("weight_decay", 0.01)))
        label_smoothing = float(request.options.get("label_smoothing", 0.0))
        if not 0.0 <= label_smoothing <= 1.0:
            raise ValueError("label_smoothing must be between 0 and 1")
        warmup_epochs = _nonnegative_int_option(dict(request.options), "warmup_epochs", 2)
        scheduler = _create_scheduler(torch, optimizer, request.epochs, warmup_epochs)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        scaler = training_options.grad_scaler(torch, device)
        start_epoch = 1
        best_accuracy = -1.0
        best_epoch = 0
        stale_epochs = 0
        history = []
        if request.resume:
            resume_path = require_file(request.weights or context.run_dir / "last.pt", "resume checkpoint")
            logger.info("Resuming checkpoint from %s", resume_path)
            resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
            model.load_state_dict(resume_state["model_state_dict"], strict=True)
            if resume_state.get("optimizer_state_dict") is not None:
                optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            else:
                logger.warning("Resume checkpoint has no optimizer state; optimizer starts fresh")
            completed_epoch = int(resume_state.get("epoch", 0))
            start_epoch = completed_epoch + 1
            metrics = resume_state.get("metrics", {})
            saved_best_accuracy = resume_state.get("best_accuracy")
            if saved_best_accuracy is None:
                saved_best_accuracy = metrics.get("val_accuracy", -1.0)
            best_accuracy = float(saved_best_accuracy if saved_best_accuracy is not None else -1.0)
            best_epoch = int(resume_state.get("best_epoch", completed_epoch if best_accuracy >= 0 else 0))
            stale_epochs = int(resume_state.get("stale_epochs", 0))
            history = list(resume_state.get("history", []))
            if resume_state.get("rng_state") is not None:
                restore_rng_state(torch, loader_generator, resume_state["rng_state"])
            else:
                logger.warning("Resume checkpoint has no RNG state; continuation is not bitwise identical")
            if scaler is not None and resume_state.get("scaler_state_dict") is not None:
                scaler.load_state_dict(resume_state["scaler_state_dict"])
            if resume_state.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            logger.info("Resume state: start_epoch=%d best_epoch=%d best_val_accuracy=%.4f stale_epochs=%d", start_epoch, best_epoch, best_accuracy, stale_epochs)
        training_model = model
        if training_options.compile:
            if device.type != "cuda":
                raise ValueError("compile requires a CUDA device")
            logger.info("Compiling timm classification model")
            training_model = torch.compile(model)
        for epoch in range(start_epoch, request.epochs + 1):
            epoch_started = time.perf_counter()
            logger.info("Epoch %d/%d started", epoch, request.epochs)
            training_model.train()
            if request.batch == 1:
                for layer in model.modules():
                    if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                        layer.eval()
            train_loss_total = torch.zeros((), device=device, dtype=torch.float32)
            train_correct = torch.zeros((), device=device, dtype=torch.int64)
            train_total = torch.zeros((), device=device, dtype=torch.int64)
            with tqdm(loader, desc=f"Epoch {epoch}/{request.epochs} train", unit="batch", leave=False) as progress:
                for images, labels in progress:
                    images = images.to(device, non_blocking=training_options.pin_memory)
                    labels = labels.to(device, non_blocking=training_options.pin_memory)
                    optimizer.zero_grad(set_to_none=True)
                    with training_options.autocast(torch, device):
                        logits = training_model(images)
                        loss = criterion(logits, labels)
                    training_options.backward_step(loss, optimizer, scaler)
                    batch_size = labels.numel()
                    train_loss_total.add_(loss.detach().float() * batch_size)
                    train_correct.add_((logits.detach().argmax(1) == labels).sum())
                    train_total.add_(batch_size)
            train_samples = max(1, int(train_total.item()))
            train_loss = float(train_loss_total.item()) / train_samples
            train_accuracy = float(train_correct.item()) / train_samples
            training_model.eval()
            validate = epoch == start_epoch or epoch == request.epochs or (epoch - start_epoch) % validation_interval == 0
            val_classification = None
            if validate:
                correct = torch.zeros((), device=device, dtype=torch.int64)
                total = torch.zeros((), device=device, dtype=torch.int64)
                val_loss_total = torch.zeros((), device=device, dtype=torch.float32)
                val_confusion = torch.zeros(
                    (len(train_set.classes), len(train_set.classes)),
                    device=device,
                    dtype=torch.int64,
                )
                with torch.inference_mode():
                    with tqdm(val_loader, desc=f"Epoch {epoch}/{request.epochs} val", unit="batch", leave=False) as progress:
                        for images, labels in progress:
                            images = images.to(device, non_blocking=training_options.pin_memory)
                            labels = labels.to(device, non_blocking=training_options.pin_memory)
                            with training_options.autocast(torch, device):
                                logits = training_model(images)
                                loss = criterion(logits, labels)
                            predictions = logits.argmax(1)
                            val_loss_total.add_(loss.float() * labels.numel())
                            correct.add_((predictions == labels).sum())
                            total.add_(labels.numel())
                            indices = labels * len(train_set.classes) + predictions
                            val_confusion.add_(
                                torch.bincount(
                                    indices,
                                    minlength=len(train_set.classes) ** 2,
                                ).reshape(len(train_set.classes), len(train_set.classes))
                            )
                val_samples = max(1, int(total.item()))
                val_loss = float(val_loss_total.item()) / val_samples
                val_accuracy = float(correct.item()) / val_samples
                val_classification = _classification_metrics(val_confusion, train_set.classes)
            else:
                val_loss = None
                val_accuracy = None
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_macro_precision": val_classification["macro_precision"] if val_classification else None,
                "val_macro_f1": val_classification["macro_f1"] if val_classification else None,
                "val_balanced_accuracy": val_classification["balanced_accuracy"] if val_classification else None,
                "val_per_class_recall": val_classification["per_class_recall"] if val_classification else None,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "validated": validate,
            }
            if trial is not None and validate and val_accuracy is not None:
                trial.report(val_accuracy, step=epoch)
                if trial.should_prune():
                    context.emit("optuna_trial_pruned", {"trial": trial.number, "epoch": epoch, "value": val_accuracy})
                    raise optuna.TrialPruned()
            history.append(row)
            improved = validate and val_accuracy is not None and val_accuracy > best_accuracy
            if improved:
                best_accuracy = val_accuracy
                best_epoch = epoch
                stale_epochs = 0
            elif validate:
                stale_epochs += 1
            scheduler.step()
            payload = {
                "task": "classification", "model_name": request.model.variant, "classes": train_set.classes,
                "data_config": {
                    "input_size": [3, image_size, image_size],
                    "mean": list(mean),
                    "std": list(std),
                    "augmentation_enabled": augmentation_enabled,
                    "augmentation_backend": augmentation_backend,
                    "augmentation_policy": augmentation_policy,
                    "cache_mode": cache_mode,
                },
                "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "training_config": {
                    "optimizer": "AdamW",
                    "label_smoothing": label_smoothing,
                    "scheduler": "cosine",
                    "warmup_epochs": warmup_epochs,
                },
                "metrics": row, "best_accuracy": best_accuracy, "best_epoch": best_epoch,
                "stale_epochs": stale_epochs, "history": history, "rng_state": capture_rng_state(torch, loader_generator),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            }
            torch.save(payload, context.run_dir / "last.pt")
            if improved:
                torch.save(payload, context.run_dir / "best.pt")
            logger.info(
                "Epoch %d/%d complete: train_loss=%.6f train_accuracy=%.4f val_loss=%s val_accuracy=%s learning_rate=%.6g duration=%.1fs%s%s",
                epoch, request.epochs, row["train_loss"], row["train_accuracy"],
                f"{val_loss:.6f}" if val_loss is not None else "skipped",
                f"{val_accuracy:.4f}" if val_accuracy is not None else "skipped",
                row["learning_rate"], time.perf_counter() - epoch_started,
                " best" if improved else "", f" patience={stale_epochs}/{request.patience}" if request.patience > 0 else "",
            )
            if request.patience > 0 and stale_epochs >= request.patience:
                logger.info("Early stopping after epoch %d: no validation improvement for %d epochs", epoch, request.patience)
                break
        completed_epochs = max(start_epoch - 1, max((int(row.get("epoch", 0)) for row in history), default=0))
        stopped_early = request.patience > 0 and stale_epochs >= request.patience
        last_metrics = history[-1] if history else {}
        best_metrics = next((row for row in reversed(history) if int(row.get("epoch", 0)) == best_epoch), {})
        logger.info(
            "Training complete: epochs=%d/%d best_epoch=%d best_val_loss=%s best_val_accuracy=%.4f last_train_loss=%s last_train_accuracy=%s last_val_loss=%s last_val_accuracy=%s elapsed=%.1fs",
            completed_epochs, request.epochs, best_epoch,
            f"{best_metrics['val_loss']:.6f}" if best_metrics.get("val_loss") is not None else "n/a",
            best_accuracy,
            f"{last_metrics['train_loss']:.6f}" if last_metrics.get("train_loss") is not None else "n/a",
            f"{last_metrics['train_accuracy']:.4f}" if last_metrics.get("train_accuracy") is not None else "n/a",
            f"{last_metrics['val_loss']:.6f}" if last_metrics.get("val_loss") is not None else "n/a",
            f"{last_metrics['val_accuracy']:.4f}" if last_metrics.get("val_accuracy") is not None else "n/a",
            time.perf_counter() - started,
        )
        summary = {
            "history": history,
            "best_val_accuracy": best_accuracy,
            "best_val_loss": best_metrics.get("val_loss"),
            "best_val_macro_precision": best_metrics.get("val_macro_precision"),
            "best_val_macro_f1": best_metrics.get("val_macro_f1"),
            "best_val_balanced_accuracy": best_metrics.get("val_balanced_accuracy"),
            "best_val_per_class_recall": best_metrics.get("val_per_class_recall"),
            "best_epoch": best_epoch,
            "last_train_loss": last_metrics.get("train_loss"),
            "last_train_accuracy": last_metrics.get("train_accuracy"),
            "last_val_loss": last_metrics.get("val_loss"),
            "last_val_accuracy": last_metrics.get("val_accuracy"),
            "last_val_macro_precision": last_metrics.get("val_macro_precision"),
            "last_val_macro_f1": last_metrics.get("val_macro_f1"),
            "last_val_balanced_accuracy": last_metrics.get("val_balanced_accuracy"),
            "last_val_per_class_recall": last_metrics.get("val_per_class_recall"),
            "epochs": completed_epochs,
            "stopped_early": stopped_early,
            "image_size": image_size,
            "classes": train_set.classes,
        }
        context.write_json("metrics.json", summary)
        return Execution(
            tuple(artifact(context.run_dir / name, "checkpoint") for name in ("best.pt", "last.pt")),
            {key: value for key, value in summary.items() if key != "history"},
        )

    def tune(self, request: ResolvedTuneRequest, context: WorkflowContext) -> BackendExecution:
        optuna = optional_import("optuna")
        learning_rate_min = float(request.learning_rate_min)
        learning_rate_max = float(request.learning_rate_max)
        weight_decay_min = float(request.weight_decay_min)
        weight_decay_max = float(request.weight_decay_max)
        label_smoothing_min = float(request.label_smoothing_min)
        label_smoothing_max = float(request.label_smoothing_max)
        if not 0.0 < learning_rate_min < learning_rate_max:
            raise ValueError("learning_rate_min must be less than learning_rate_max and greater than zero")
        if not 0.0 <= weight_decay_min < weight_decay_max:
            raise ValueError("weight_decay_min must be less than weight_decay_max")
        if not 0.0 <= label_smoothing_min < label_smoothing_max <= 1.0:
            raise ValueError("label_smoothing_min must be less than label_smoothing_max and both must be between 0 and 1")

        startup_trials = min(5, max(1, int(request.trials) // 4))
        sampler = optuna.samplers.TPESampler(
            seed=int(request.seed),
            n_startup_trials=startup_trials,
        )
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=startup_trials,
                n_warmup_steps=1,
            ),
            storage=request.storage,
            study_name=request.study_name,
            load_if_exists=request.storage is not None,
        )
        current_trials = []

        def objective(trial):
            trial_options = dict(request.options)
            trial_options["learning_rate"] = trial.suggest_float(
                "learning_rate", learning_rate_min, learning_rate_max, log=True
            )
            trial_options["weight_decay"] = trial.suggest_float(
                "weight_decay", weight_decay_min, weight_decay_max
            )
            trial_options["label_smoothing"] = trial.suggest_float(
                "label_smoothing", label_smoothing_min, label_smoothing_max
            )
            trial_dir = context.run_dir / f"trial-{trial.number:04d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            trial_request = ResolvedTrainRequest(
                request.selection,
                request.model,
                request.data,
                trial_dir,
                None,
                False,
                False,
                trial_options,
            )
            trial_context = WorkflowContext(trial_dir, context.emit, context.device)
            execution = self._train(trial_request, trial_context, trial)
            score = float(execution.metrics["best_val_accuracy"])
            trial.set_user_attr("run_dir", str(trial_dir))
            current_trials.append(trial)
            return score

        study.optimize(objective, n_trials=int(request.trials))
        if not current_trials:
            raise ValueError("Optuna completed no trials")
        best_trial = max(current_trials, key=lambda item: item.value)
        best_dir = Path(best_trial.user_attrs["run_dir"])
        for name in ("best.pt", "last.pt"):
            source = require_file(best_dir / name, f"Optuna trial {name}")
            shutil.copy2(source, context.run_dir / name)

        trials = [
            {
                "number": item.number,
                "state": getattr(item.state, "name", str(item.state)),
                "value": item.value,
                "params": dict(item.params),
                "user_attrs": dict(item.user_attrs),
            }
            for item in study.trials
        ]
        summary = {
            "study_name": study.study_name,
            "direction": "maximize",
            "best_value": best_trial.value,
            "best_params": dict(best_trial.params),
            "study_best_value": study.best_value,
            "trials": trials,
        }
        context.write_json("optuna.json", summary)
        metrics = {
            "study_name": study.study_name,
            "best_val_accuracy": float(best_trial.value),
            "best_params": dict(best_trial.params),
            "trials": len(study.trials),
            "completed_trials": sum(getattr(item.state, "name", str(item.state)) == "COMPLETE" for item in study.trials),
        }
        return Execution(
            tuple(artifact(context.run_dir / name, "checkpoint") for name in ("best.pt", "last.pt"))
            + (artifact(context.run_dir / "optuna.json", "report"),),
            metrics,
        )

    def _load(self, checkpoint: Path, device: str):
        timm, torch, _ = self._imports()
        target = require_file(checkpoint, "classification checkpoint")
        value = torch.load(target, map_location="cpu", weights_only=False)
        classes = list(value.get("classes", []))
        model_name = str(value.get("model_name", ""))
        if not classes or not model_name:
            raise ValueError("checkpoint must contain model_name and classes")
        model = timm.create_model(model_name, pretrained=False, num_classes=len(classes))
        model.load_state_dict(value["model_state_dict"], strict=True)
        model.eval()
        return model, classes, value

    def export(self, request: ResolvedExportRequest, context: WorkflowContext) -> BackendExecution:
        _, torch, _ = self._imports()
        model, classes, value = self._load(request.checkpoint, request.device)
        size = int(request.image_size or value.get("data_config", {}).get("input_size", [3, 224, 224])[-1])
        data_config = value.get("data_config", {})
        mean = tuple(float(item) for item in data_config.get("mean", _DEFAULT_MEAN))
        std = tuple(float(item) for item in data_config.get("std", _DEFAULT_STD))
        core = context.run_dir / "core-float32.onnx"
        export_batch = request.batch_size or 1
        tensor = torch.zeros(export_batch, 3, size, size)
        batch = torch.export.Dim("batch", min=1) if request.batch_size is None else None

        class ProbabilityModel(torch.nn.Module):
            def __init__(self, classifier):
                super().__init__()
                self.classifier = classifier

            def forward(self, images):
                return torch.softmax(self.classifier(images), dim=1)

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            torch.onnx.export(
                ProbabilityModel(model).eval(),
                (tensor,),
                core,
                input_names=["images"],
                output_names=["probabilities"],
                opset_version=request.opset,
                dynamo=True,
                dynamic_shapes=({0: batch},) if batch is not None else None,
                external_data=False,
                optimize=True,
                verify=True,
                verbose=False,
            )
        outputs = embedded_output_paths(request.output)
        paths = wrap_embedded_variants(
            core, outputs, image_size=size, mean=mean, std=std,
            batch_size=request.batch_size,
        )
        contract = classification_contract(classes, batch_size=request.batch_size)
        checks: list[dict[str, Any]] = []
        for variant, path in outputs.items():
            variant_contract = classification_contract(
                classes, input_variant=variant, batch_size=request.batch_size
            )
            set_onnx_metadata(path, metadata_for_contract(variant_contract))
            checks.extend({**check, "variant": variant} for check in validate_onnx(path, variant_contract))

        metrics: dict[str, Any] = {}
        artifacts = [artifact(path, "onnx") for path in paths]
        if request.data is not None:
            metrics = self._validate_export_dataset(
                model,
                classes,
                value,
                outputs,
                core,
                request.data,
                image_size=size,
                batch_size=request.batch_size,
                device=request.device,
            )
            report = context.write_json("dataset-validation.json", metrics)
            artifacts.append(artifact(report, "report"))
            checks.append({
                "name": "checkpoint_native_validation",
                "status": "passed",
                "split": "val",
                "images": metrics["native"]["images"],
            })
            checks.append({
                "name": "native_export_validation",
                "status": "passed",
                "split": "val",
                "images": metrics["native-export"]["images"],
            })
            for variant in ("bw8", "c24"):
                checks.append({
                    "name": "wrapped_dataset_validation",
                    "status": "passed",
                    "variant": variant,
                    "split": "val",
                    "images": metrics[variant]["images"],
                })
        return Execution(tuple(artifacts), metrics, contract, tuple(checks))

    def _validate_export_dataset(
        self,
        model: Any,
        classes: list[str],
        value: dict[str, Any],
        outputs: dict[str, Path],
        core: Path,
        data: Path | None,
        *,
        image_size: int,
        batch_size: int | None,
        device: str,
    ) -> dict[str, Any]:
        native_metrics, native_probabilities, labels = self._evaluate_predictions(
            model,
            classes,
            value,
            data,
            "val",
            device,
            image_size=image_size,
        )
        native_section = {
            key.removeprefix("native_"): value
            for key, value in native_metrics.items()
            if key.startswith("native_")
        }
        metrics: dict[str, Any] = {"validation_split": "val", "native": native_section}
        native_export, native_export_probabilities = validate_classification_native_export(
            core,
            data,
            classes=classes,
            image_size=image_size,
            batch_size=batch_size,
            mean=tuple(float(item) for item in value.get("data_config", {}).get("mean", _DEFAULT_MEAN)),
            std=tuple(float(item) for item in value.get("data_config", {}).get("std", _DEFAULT_STD)),
            reference_probabilities=native_probabilities,
        )
        metrics.update(native_export)
        wrapped = validate_classification_wrappers(
            outputs,
            data,
            classes=classes,
            image_size=image_size,
            batch_size=batch_size,
            reference_probabilities=native_export_probabilities,
            reference_name="native_export",
        )
        metrics.update(wrapped)
        metrics["validation_split"] = "val"
        return metrics

    def _evaluate(self, request: ResolvedValidateRequest | ResolvedTestRequest):
        model, classes, value = self._load(request.target, request.device)
        metrics, _, _ = self._evaluate_predictions(
            model, classes, value, request.data, request.split, request.device
        )
        return metrics

    def _evaluate_predictions(
        self,
        model: Any,
        classes: list[str],
        value: dict[str, Any],
        data: Path | None,
        split: str,
        device_name: str,
        *,
        image_size: int | None = None,
    ) -> tuple[dict[str, Any], Any, Any]:
        _, torch, _ = self._imports()
        if data is None:
            raise ValueError("evaluation requires a classification dataset")
        _, _, torchvision = self._imports()
        data_config = value.get("data_config", {})
        size = int(image_size or data_config.get("input_size", [3, 224, 224])[-1])
        mean = tuple(float(item) for item in data_config.get("mean", _DEFAULT_MEAN))
        std = tuple(float(item) for item in data_config.get("std", _DEFAULT_STD))
        transform_ops = torchvision.transforms.Compose([
            torchvision.transforms.Resize((size,) * 2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.ImageFolder(str(data / split), transform=transform_ops)
        if dataset.classes != classes:
            raise ValueError("dataset classes differ from checkpoint classes")
        if len(dataset) == 0:
            raise ValueError("classification validation split is empty")
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        device = torch.device(
            "cuda" if device_name in {"auto", "cuda"} and torch.cuda.is_available()
            else device_name if device_name != "auto" else "cpu"
        )
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        loss_total = 0.0
        correct = total = 0
        confusion = torch.zeros(
            (len(classes), len(classes)),
            device=device,
            dtype=torch.int64,
        )
        probabilities: list[Any] = []
        collected_labels: list[Any] = []
        with torch.inference_mode():
            for images, labels in loader:
                labels = labels.to(device)
                logits = model(images.to(device))
                batch_probabilities = torch.softmax(logits, dim=1)
                loss_total += float(criterion(logits, labels).item()) * len(labels)
                predictions = logits.argmax(1)
                correct += int((predictions == labels).sum())
                total += len(labels)
                indices = labels * len(classes) + predictions
                confusion.add_(
                    torch.bincount(
                        indices,
                        minlength=len(classes) ** 2,
                    ).reshape(len(classes), len(classes))
                )
                probabilities.append(batch_probabilities.detach().cpu())
                collected_labels.append(labels.detach().cpu())
        native_probabilities = torch.cat(probabilities).numpy()
        collected_labels_array = torch.cat(collected_labels).numpy()
        classification = _classification_metrics(confusion, classes)
        metrics = {
            "split": split,
            "loss": loss_total / max(1, total),
            "accuracy": correct / max(1, total),
            "correct": correct,
            "images": total,
            **classification,
            "native_accuracy": correct / max(1, total),
            "native_loss": loss_total / max(1, total),
            "native_correct": correct,
            "native_images": total,
            "native_macro_precision": classification["macro_precision"],
            "native_macro_f1": classification["macro_f1"],
            "native_balanced_accuracy": classification["balanced_accuracy"],
            "native_per_class_precision": classification["per_class_precision"],
            "native_per_class_recall": classification["per_class_recall"],
            "native_per_class_f1": classification["per_class_f1"],
        }
        logger.info(
            "Evaluation complete: split=%s images=%d loss=%.6f accuracy=%.4f correct=%d",
            split, total, metrics["native_loss"], metrics["native_accuracy"], correct,
        )
        return metrics, native_probabilities, collected_labels_array

    def validate(self, request: ResolvedValidateRequest, context: WorkflowContext) -> BackendExecution:
        if request.target.suffix.casefold() == ".onnx":
            contract = classification_contract([])
            return Execution((artifact(require_file(request.target, "ONNX artifact"), "onnx"),), {}, contract, validate_onnx(request.target, contract))
        metrics = self._evaluate(request) if request.data else {}
        return Execution((artifact(require_file(request.target, "checkpoint"), "checkpoint"),), metrics, {}, (({"name": "native_validation", "status": "passed"}),))

    def test(self, request: ResolvedTestRequest, context: WorkflowContext) -> BackendExecution:
        return Execution((artifact(require_file(request.target, "checkpoint"), "checkpoint"),), self._evaluate(request), {}, ())
