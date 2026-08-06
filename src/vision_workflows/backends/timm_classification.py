from __future__ import annotations

import io
import logging
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain.results import ArtifactRef
from ..workflows.context import WorkflowContext, optional_import
from ..workflows.requests import ResolvedExportRequest, ResolvedTestRequest, ResolvedTrainRequest, ResolvedValidateRequest
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


logger = logging.getLogger(__name__)

_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_STD = (0.229, 0.224, 0.225)


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

    def train(self, request: ResolvedTrainRequest, context: WorkflowContext) -> BackendExecution:
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
        validation_interval = _positive_int_option(dict(request.options), "validate_every", 1)
        loader = torch.utils.data.DataLoader(train_set, batch_size=request.batch, shuffle=True, generator=loader_generator, worker_init_fn=worker_init_fn, **loader_options)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=request.batch, shuffle=False, worker_init_fn=worker_init_fn, **loader_options)
        logger.info("DataLoaders ready: batch=%d requested_workers=%d effective_workers=%d validate_every=%d", request.batch, requested_workers, workers, validation_interval)
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
            if validate:
                correct = torch.zeros((), device=device, dtype=torch.int64)
                total = torch.zeros((), device=device, dtype=torch.int64)
                val_loss_total = torch.zeros((), device=device, dtype=torch.float32)
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
                val_samples = max(1, int(total.item()))
                val_loss = float(val_loss_total.item()) / val_samples
                val_accuracy = float(correct.item()) / val_samples
            else:
                val_loss = None
                val_accuracy = None
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "validated": validate,
            }
            history.append(row)
            improved = validate and val_accuracy is not None and val_accuracy > best_accuracy
            if improved:
                best_accuracy = val_accuracy
                best_epoch = epoch
                stale_epochs = 0
            elif validate:
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
            "best_epoch": best_epoch,
            "last_train_loss": last_metrics.get("train_loss"),
            "last_train_accuracy": last_metrics.get("train_accuracy"),
            "last_val_loss": last_metrics.get("val_loss"),
            "last_val_accuracy": last_metrics.get("val_accuracy"),
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
        return Execution(tuple(artifact(path, "onnx") for path in paths), {}, contract, tuple(checks))

    def _evaluate(self, request: ResolvedValidateRequest | ResolvedTestRequest):
        _, torch, _ = self._imports()
        model, classes, value = self._load(request.target, request.device)
        data = request.data
        split = request.split
        if data is None:
            raise ValueError("evaluation requires a classification dataset")
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
        device = torch.device(
            "cuda" if request.device in {"auto", "cuda"} and torch.cuda.is_available()
            else request.device if request.device != "auto" else "cpu"
        )
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        loss_total = 0.0
        correct = total = 0
        with torch.inference_mode():
            for images, labels in loader:
                labels = labels.to(device)
                logits = model(images.to(device))
                loss_total += float(criterion(logits, labels).item()) * len(labels)
                correct += int((logits.argmax(1) == labels).sum())
                total += len(labels)
        metrics = {
            "split": split,
            "loss": loss_total / max(1, total),
            "accuracy": correct / max(1, total),
            "correct": correct,
            "images": total,
        }
        logger.info(
            "Evaluation complete: split=%s images=%d loss=%.6f accuracy=%.4f correct=%d",
            split, total, metrics["loss"], metrics["accuracy"], correct,
        )
        return metrics

    def validate(self, request: ResolvedValidateRequest, context: WorkflowContext) -> BackendExecution:
        if request.target.suffix.casefold() == ".onnx":
            contract = classification_contract([])
            return Execution((artifact(require_file(request.target, "ONNX artifact"), "onnx"),), {}, contract, validate_onnx(request.target, contract))
        metrics = self._evaluate(request) if request.data else {}
        return Execution((artifact(require_file(request.target, "checkpoint"), "checkpoint"),), metrics, {}, (({"name": "native_validation", "status": "passed"}),))

    def test(self, request: ResolvedTestRequest, context: WorkflowContext) -> BackendExecution:
        return Execution((artifact(require_file(request.target, "checkpoint"), "checkpoint"),), self._evaluate(request), {}, ())
