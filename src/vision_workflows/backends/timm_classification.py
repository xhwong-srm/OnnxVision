from __future__ import annotations

import io
import logging
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain.models import BackendCapability, BackendDescriptor
from ..domain.results import ArtifactRef
from ..workflows.context import WorkflowContext, optional_import
from ..workflows.requests import ExportRequest, TestRequest, TrainRequest, ValidateRequest
from ..workflows.runs import artifact
from .base import BackendExecution, ModelBackend
from .common import classification_contract, metadata_for_contract, require_file, set_onnx_metadata, validate_onnx
from .timm_training import (
    TimmTrainingOptions,
    capture_rng_state,
    restore_rng_state,
    seed_everything,
    worker_seed,
)


logger = logging.getLogger(__name__)

_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Execution(BackendExecution):
    artifacts: tuple[ArtifactRef, ...] = ()
    metrics: dict[str, Any] = None  # type: ignore[assignment]
    contract: dict[str, Any] = None  # type: ignore[assignment]
    checks: tuple[dict[str, Any], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "metrics", self.metrics or {})
        object.__setattr__(self, "contract", self.contract or {})


class TimmClassificationBackend(ModelBackend):
    descriptor = BackendDescriptor(
        "timm", "classification", "classification", ("*",),
        frozenset(BackendCapability), "timm image-folder classifier", "timm",
    )

    @staticmethod
    def _imports():
        timm = optional_import("timm")
        torch = optional_import("torch")
        torchvision = optional_import("torchvision")
        return timm, torch, torchvision

    @staticmethod
    def _datasets(data: Path, image_size: int, train: bool, mean: tuple[float, ...] = _DEFAULT_MEAN, std: tuple[float, ...] = _DEFAULT_STD):
        _, _, torchvision = TimmClassificationBackend._imports()
        transforms = torchvision.transforms
        operations = [transforms.Resize((image_size, image_size))]
        if train:
            operations.extend([transforms.RandomHorizontalFlip(), transforms.RandomRotation(5)])
        operations.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
        transform = transforms.Compose(operations)
        root = data / ("train" if train else "val")
        if not root.is_dir():
            raise FileNotFoundError(f"classification split does not exist: {root}")
        return torchvision.datasets.ImageFolder(str(root), transform=transform)

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

    def train(self, request: TrainRequest, context: WorkflowContext) -> BackendExecution:
        started = time.perf_counter()
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
        train_set = self._datasets(data, image_size, True, mean, std)
        val_set = self._datasets(data, image_size, False, mean, std)
        logger.info("Dataset ready: train=%d val=%d classes=%s", len(train_set), len(val_set), ", ".join(train_set.classes))
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
        loader = torch.utils.data.DataLoader(train_set, batch_size=request.batch, shuffle=True, generator=loader_generator, worker_init_fn=worker_init_fn, **loader_options)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=request.batch, shuffle=False, worker_init_fn=worker_init_fn, **loader_options)
        logger.info("DataLoaders ready: batch=%d requested_workers=%d effective_workers=%d", request.batch, requested_workers, workers)
        optimizer = torch.optim.AdamW(model.parameters(), lr=request.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
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
            best_accuracy = float(resume_state.get("best_accuracy", metrics.get("val_accuracy", -1.0)))
            best_epoch = int(resume_state.get("best_epoch", completed_epoch if best_accuracy >= 0 else 0))
            stale_epochs = int(resume_state.get("stale_epochs", 0))
            history = list(resume_state.get("history", []))
            if resume_state.get("rng_state") is not None:
                restore_rng_state(torch, loader_generator, resume_state["rng_state"])
            else:
                logger.warning("Resume checkpoint has no RNG state; continuation is not bitwise identical")
            if scaler is not None and resume_state.get("scaler_state_dict") is not None:
                scaler.load_state_dict(resume_state["scaler_state_dict"])
            logger.info("Resume state: start_epoch=%d best_epoch=%d best_val_accuracy=%.4f stale_epochs=%d", start_epoch, best_epoch, best_accuracy, stale_epochs)
        for epoch in range(start_epoch, request.epochs + 1):
            epoch_started = time.perf_counter()
            logger.info("Epoch %d/%d started", epoch, request.epochs)
            model.train()
            if request.batch == 1:
                for layer in model.modules():
                    if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                        layer.eval()
            train_loss_total = torch.zeros((), device=device, dtype=torch.float32)
            with tqdm(loader, desc=f"Epoch {epoch}/{request.epochs} train", unit="batch", leave=False) as progress:
                for images, labels in progress:
                    images = images.to(device, non_blocking=training_options.pin_memory)
                    labels = labels.to(device, non_blocking=training_options.pin_memory)
                    optimizer.zero_grad(set_to_none=True)
                    with training_options.autocast(torch, device):
                        loss = criterion(model(images), labels)
                    training_options.backward_step(loss, optimizer, scaler)
                    train_loss_total.add_(loss.detach().float())
            train_loss = float(train_loss_total.cpu()) / max(1, len(loader))
            model.eval()
            correct = torch.zeros((), device=device, dtype=torch.int64)
            total = torch.zeros((), device=device, dtype=torch.int64)
            with torch.inference_mode():
                with tqdm(val_loader, desc=f"Epoch {epoch}/{request.epochs} val", unit="batch", leave=False) as progress:
                    for images, labels in progress:
                        with training_options.autocast(torch, device):
                            predictions = model(images.to(device, non_blocking=training_options.pin_memory)).argmax(1)
                        labels = labels.to(device, non_blocking=training_options.pin_memory)
                        correct.add_((predictions == labels).sum())
                        total.add_(labels.numel())
            accuracy = correct.float().div(total.clamp_min(1)).item()
            row = {"epoch": epoch, "train_loss": train_loss, "val_accuracy": accuracy}
            history.append(row)
            improved = accuracy > best_accuracy
            if improved:
                best_accuracy = accuracy
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
            payload = {
                "task": "classification", "model_name": request.model.variant, "classes": train_set.classes,
                "data_config": {"input_size": [3, image_size, image_size], "mean": list(mean), "std": list(std)},
                "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "metrics": row, "best_accuracy": best_accuracy, "best_epoch": best_epoch,
                "stale_epochs": stale_epochs, "history": history, "rng_state": capture_rng_state(torch, loader_generator),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            }
            torch.save(payload, context.run_dir / "last.pt")
            if improved:
                torch.save(payload, context.run_dir / "best.pt")
            logger.info(
                "Epoch %d/%d complete: train_loss=%.6f val_accuracy=%.4f duration=%.1fs%s%s",
                epoch, request.epochs, row["train_loss"], accuracy, time.perf_counter() - epoch_started,
                " best" if improved else "", f" patience={stale_epochs}/{request.patience}" if request.patience > 0 else "",
            )
            if request.patience > 0 and stale_epochs >= request.patience:
                logger.info("Early stopping after epoch %d: no validation improvement for %d epochs", epoch, request.patience)
                break
        completed_epochs = max(start_epoch - 1, max((int(row.get("epoch", 0)) for row in history), default=0))
        stopped_early = request.patience > 0 and stale_epochs >= request.patience
        logger.info("Training complete: epochs=%d/%d best_epoch=%d best_val_accuracy=%.4f elapsed=%.1fs", completed_epochs, request.epochs, best_epoch, best_accuracy, time.perf_counter() - started)
        context.write_json("metrics.json", {"history": history, "best_val_accuracy": best_accuracy, "best_epoch": best_epoch, "epochs": completed_epochs, "stopped_early": stopped_early, "image_size": image_size, "classes": train_set.classes})
        return Execution(tuple(artifact(context.run_dir / name, "checkpoint") for name in ("best.pt", "last.pt")), {"best_val_accuracy": best_accuracy, "best_epoch": best_epoch, "epochs": completed_epochs, "stopped_early": stopped_early, "image_size": image_size})

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

    def export(self, request: ExportRequest, context: WorkflowContext) -> BackendExecution:
        _, torch, _ = self._imports()
        model, classes, value = self._load(request.checkpoint, request.device)
        size = int(request.image_size or value.get("data_config", {}).get("input_size", [3, 224, 224])[-1])
        output = request.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        tensor = torch.zeros(1, 3, size, size)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            torch.onnx.export(model, tensor, output, input_names=["images"], output_names=["probabilities"], opset_version=request.opset, dynamo=True)
        contract = classification_contract(classes)
        set_onnx_metadata(output, metadata_for_contract(contract))
        checks = validate_onnx(output, contract)
        return Execution((artifact(output, "onnx"),), {}, contract, checks)

    def _evaluate(self, request: ValidateRequest | TestRequest):
        _, torch, _ = self._imports()
        model, classes, value = self._load(request.target, request.device)
        data = request.data
        split = request.split
        _, _, torchvision = self._imports()
        data_config = value.get("data_config", {})
        transform_ops = torchvision.transforms.Compose([
            torchvision.transforms.Resize((int(data_config.get("input_size", [3, 224, 224])[-1]),) * 2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(tuple(data_config.get("mean", _DEFAULT_MEAN)), tuple(data_config.get("std", _DEFAULT_STD))),
        ])
        dataset = torchvision.datasets.ImageFolder(str(data / split), transform=transform_ops)
        if dataset.classes != classes:
            raise ValueError("dataset classes differ from checkpoint classes")
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        correct = total = 0
        with torch.inference_mode():
            for images, labels in loader:
                correct += int((model(images).argmax(1) == labels).sum())
                total += len(labels)
        return {"accuracy": correct / max(1, total), "images": total}

    def validate(self, request: ValidateRequest, context: WorkflowContext) -> BackendExecution:
        if request.target.suffix.casefold() == ".onnx":
            contract = classification_contract([])
            return Execution((artifact(require_file(request.target, "ONNX artifact"), "onnx"),), {}, contract, validate_onnx(request.target, contract))
        metrics = self._evaluate(request) if request.data else {}
        return Execution((artifact(require_file(request.target, "checkpoint"), "checkpoint"),), metrics, {}, (({"name": "native_validation", "status": "passed"}),))

    def test(self, request: TestRequest, context: WorkflowContext) -> BackendExecution:
        return Execution((artifact(require_file(request.target, "checkpoint"), "checkpoint"),), self._evaluate(request), {}, ())
