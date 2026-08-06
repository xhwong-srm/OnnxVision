from __future__ import annotations

import io
import logging
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain.datasets import DatasetFormat, DetectionDataset, Split
from ..domain.results import ArtifactRef
from ..workflows.context import WorkflowContext, optional_import
from ..workflows.requests import ResolvedExportRequest, ResolvedTestRequest, ResolvedTrainRequest, ResolvedValidateRequest
from ..workflows.runs import artifact
from .base import BackendExecution
from .common import (
    detection_contract,
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


@dataclass(frozen=True)
class Execution(BackendExecution):
    artifacts: tuple[ArtifactRef, ...] = ()
    metrics: dict[str, Any] = None  # type: ignore[assignment]
    contract: dict[str, Any] = None  # type: ignore[assignment]
    checks: tuple[dict[str, Any], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "metrics", self.metrics or {})
        object.__setattr__(self, "contract", self.contract or {})


def _torch_components():
    torch = optional_import("torch")
    timm = optional_import("timm")
    torchvision = optional_import("torchvision")
    return torch, timm, torchvision


def _model_class(torch, timm, classes: int, variant: str, queries: int, pretrained: bool):
    nn = torch.nn

    class QueryDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(variant, pretrained=pretrained, features_only=True, out_indices=(-1,))
            channels = self.backbone.feature_info.channels()[-1]
            hidden = int(max(64, min(256, channels)))
            self.projection = nn.Conv2d(channels, hidden, 1)
            self.queries = nn.Parameter(torch.randn(queries, hidden) * 0.02)
            heads = max(1, min(8, hidden // 32))
            while hidden % heads:
                heads -= 1
            self.decoder = nn.MultiheadAttention(hidden, heads, batch_first=True)
            self.class_head = nn.Linear(hidden, classes + 1)
            self.box_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4), nn.Sigmoid())
            self.num_classes = classes
            self.num_queries = queries

        def forward(self, images):
            features = self.backbone(images)[-1]
            pooled = self.projection(features).mean(dim=(2, 3)).unsqueeze(1)
            queries_tensor = self.queries.unsqueeze(0).expand(images.shape[0], -1, -1)
            decoded, _ = self.decoder(queries_tensor, pooled, pooled)
            return self.class_head(decoded), self.box_head(decoded)

        def export_outputs(self, images):
            logits, boxes = self(images)
            probabilities = torch.softmax(logits, dim=-1)[..., : self.num_classes]
            scores, class_ids = probabilities.max(dim=-1)
            return boxes, scores, class_ids.to(torch.int64)

    return QueryDetector


class TimmDetectionDataset:
    def __init__(self, dataset: DetectionDataset, split: str, image_size: int):
        torch, _, torchvision = _torch_components()
        self.torch = torch
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        selected = Split(split)
        self.samples = tuple(sample for sample in dataset.samples if sample.split == selected)
        if not self.samples:
            raise ValueError(f"detection dataset has no {split} samples")
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image

        sample = self.samples[index]
        with Image.open(sample.image.path) as image:
            tensor = self.transform(image.convert("RGB"))
        boxes = [[item.box.x1 / sample.image.width, item.box.y1 / sample.image.height, item.box.x2 / sample.image.width, item.box.y2 / sample.image.height] for item in sample.annotations]
        labels = [item.class_id for item in sample.annotations]
        return tensor, self.torch.tensor(boxes, dtype=self.torch.float32).reshape(-1, 4), self.torch.tensor(labels, dtype=self.torch.long)


def _collate(batch):
    images, boxes, labels = zip(*batch)
    return __import__("torch").stack(images), list(boxes), list(labels)


class TimmDetectionBackend:

    def _dataset(self, request: ResolvedTrainRequest | ResolvedValidateRequest | ResolvedTestRequest, split: str):
        from ..datasets.formats.base import load_dataset

        if request.data is None:
            raise ValueError("evaluation requires a detection dataset")
        dataset = load_dataset(request.data, DatasetFormat.COCO)
        if not isinstance(dataset, DetectionDataset):
            raise ValueError("timm-obd-v1 requires an object-detection dataset")
        image_size = request.image_size or int(request.options.get("image_size", 640))
        return dataset, TimmDetectionDataset(dataset, split, image_size)

    def train(self, request: ResolvedTrainRequest, context: WorkflowContext) -> BackendExecution:
        started = time.perf_counter()
        torch, timm, _ = _torch_components()
        dataset, train_set = self._dataset(request, "train")
        _, val_set = self._dataset(request, "val")
        image_size = train_set.image_size
        requested_queries = int(request.options.get("num_queries", 0))
        queries = requested_queries or max(8, max((len(sample.annotations) for sample in dataset.samples), default=1) + 4)
        variant = str(request.options.get("backbone", "mobilenetv4_conv_small.e3600_r256_in1k"))
        seed_everything(torch, request.seed, request.deterministic)
        device = torch.device("cuda" if request.device == "auto" and torch.cuda.is_available() else request.device if request.device != "auto" else "cpu")
        model_pretrained = request.pretrained and not request.resume and not request.weights
        logger.info(
            "Training timm detection: model=%s data=%s output=%s device=%s epochs=%d batch=%d image_size=%d learning_rate=%g workers=%d seed=%d pretrained=%s resume=%s patience=%d deterministic=%s weights=%s options=%s",
            variant,
            request.data,
            context.run_dir,
            request.device,
            request.epochs,
            request.batch,
            image_size,
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
        model = _model_class(torch, timm, len(dataset.classes), variant, queries, model_pretrained)()
        if request.weights and not request.resume:
            logger.info("Loading custom detection weights from %s", request.weights)
            checkpoint = torch.load(require_file(request.weights, "weights"), map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device)
        workers = max(0, request.workers)
        training_options = TimmTrainingOptions.from_mapping(request.options)
        loader_options = training_options.data_loader_kwargs(workers)
        loader_generator = torch.Generator().manual_seed(request.seed)
        worker_init_fn = worker_seed if workers > 0 else None
        loader = torch.utils.data.DataLoader(train_set, request.batch, shuffle=True, collate_fn=_collate, generator=loader_generator, worker_init_fn=worker_init_fn, **loader_options)
        val_loader = torch.utils.data.DataLoader(val_set, request.batch, shuffle=False, collate_fn=_collate, worker_init_fn=worker_init_fn, **loader_options)
        optimizer = torch.optim.AdamW(model.parameters(), lr=request.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        scaler = training_options.grad_scaler(torch, device)
        start_epoch = 1
        best = -1.0
        best_epoch = 0
        stale_epochs = 0
        history = []
        if request.resume:
            resume_path = require_file(request.weights or context.run_dir / "last.pt", "resume checkpoint")
            logger.info("Resuming detection checkpoint from %s", resume_path)
            resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
            model.load_state_dict(resume_state["model_state_dict"], strict=True)
            if resume_state.get("optimizer_state_dict") is not None:
                optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            else:
                logger.warning("Resume checkpoint has no optimizer state; optimizer starts fresh")
            completed_epoch = int(resume_state.get("epoch", 0))
            start_epoch = completed_epoch + 1
            metrics = resume_state.get("metrics", {})
            best = float(resume_state.get("best_mean_score", metrics.get("mean_score", -1.0)))
            best_epoch = int(resume_state.get("best_epoch", completed_epoch if best >= 0 else 0))
            stale_epochs = int(resume_state.get("stale_epochs", 0))
            history = list(resume_state.get("history", []))
            if resume_state.get("rng_state") is not None:
                restore_rng_state(torch, loader_generator, resume_state["rng_state"])
            else:
                logger.warning("Resume checkpoint has no RNG state; continuation is not bitwise identical")
            if scaler is not None and resume_state.get("scaler_state_dict") is not None:
                scaler.load_state_dict(resume_state["scaler_state_dict"])
            logger.info("Resume state: start_epoch=%d best_epoch=%d best_mean_score=%.4f stale_epochs=%d", start_epoch, best_epoch, best, stale_epochs)
        for epoch in range(start_epoch, request.epochs + 1):
            model.train()
            if request.batch == 1:
                for layer in model.modules():
                    if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                        layer.eval()
            loss_total = 0.0
            for images, boxes, labels in loader:
                images = images.to(device, non_blocking=training_options.pin_memory)
                with training_options.autocast(torch, device):
                    logits, predicted_boxes = model(images)
                    target_classes = torch.full(logits.shape[:2], len(dataset.classes), dtype=torch.long, device=device)
                    target_boxes = torch.zeros_like(predicted_boxes)
                    for index, (sample_boxes, sample_labels) in enumerate(zip(boxes, labels)):
                        count = min(queries, len(sample_labels))
                        if count:
                            target_classes[index, :count] = sample_labels[:count].to(device, non_blocking=training_options.pin_memory)
                            target_boxes[index, :count] = sample_boxes[:count].to(device, non_blocking=training_options.pin_memory)
                    loss = criterion(logits.reshape(-1, logits.shape[-1]), target_classes.reshape(-1))
                    positive = target_classes != len(dataset.classes)
                    if positive.any():
                        loss = loss + torch.nn.functional.l1_loss(predicted_boxes[positive], target_boxes[positive])
                optimizer.zero_grad(set_to_none=True)
                training_options.backward_step(loss, optimizer, scaler)
                loss_total += float(loss.detach().cpu())
            metrics = self._evaluate(model, val_loader, device, len(dataset.classes), training_options)
            row = {"epoch": epoch, "loss": loss_total / max(1, len(loader)), **metrics}
            history.append(row)
            improved = metrics["mean_score"] > best
            if improved:
                best = metrics["mean_score"]
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
            payload = {
                "task": "detection", "architecture": "query", "model_name": variant,
                "classes": list(dataset.classes), "num_queries": queries, "image_size": image_size,
                "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "metrics": row, "best_mean_score": best, "best_epoch": best_epoch,
                "stale_epochs": stale_epochs, "history": history,
                "rng_state": capture_rng_state(torch, loader_generator),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            }
            torch.save(payload, context.run_dir / "last.pt")
            if improved:
                torch.save(payload, context.run_dir / "best.pt")
            if request.patience > 0 and stale_epochs >= request.patience:
                logger.info("Early stopping detection after epoch %d: no validation improvement for %d epochs", epoch, request.patience)
                break
        completed_epochs = max(start_epoch - 1, max((int(row.get("epoch", 0)) for row in history), default=0))
        stopped_early = request.patience > 0 and stale_epochs >= request.patience
        logger.info("Detection training complete: epochs=%d/%d best_epoch=%d best_mean_score=%.4f elapsed=%.1fs", completed_epochs, request.epochs, best_epoch, best, time.perf_counter() - started)
        context.write_json("metrics.json", {"history": history, "best_mean_score": best, "best_epoch": best_epoch, "epochs": completed_epochs, "stopped_early": stopped_early, "image_size": image_size, "classes": list(dataset.classes)})
        return Execution(
            tuple(artifact(context.run_dir / name, "checkpoint") for name in ("best.pt", "last.pt")),
            {"best_mean_score": best, "best_epoch": best_epoch, "epochs": completed_epochs, "stopped_early": stopped_early, "image_size": image_size},
        )

    def _load(self, target: Path, device):
        torch, timm, _ = _torch_components()
        checkpoint = torch.load(require_file(target, "detection checkpoint"), map_location=device, weights_only=False)
        classes = list(checkpoint["classes"])
        variant = checkpoint["model_name"]
        model = _model_class(torch, timm, len(classes), variant, int(checkpoint["num_queries"]), False)()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return model, classes, checkpoint

    @staticmethod
    def _evaluate(model, loader, device, class_count, training_options=None):
        torch = __import__("torch")
        training_options = training_options or TimmTrainingOptions()
        model.eval()
        scores = []
        with torch.inference_mode():
            for images, _, _ in loader:
                with training_options.autocast(torch, device):
                    logits, _ = model(images.to(device, non_blocking=training_options.pin_memory))
                scores.extend(torch.softmax(logits, -1)[..., :class_count].max(-1).values.mean(-1).cpu().tolist())
        return {"images": len(scores), "mean_score": sum(scores) / max(1, len(scores))}

    def export(self, request: ResolvedExportRequest, context: WorkflowContext) -> BackendExecution:
        torch, _, _ = _torch_components()
        model, classes, checkpoint = self._load(request.checkpoint, torch.device("cpu"))
        core = context.run_dir / "core-float32.onnx"
        batch = torch.export.Dim("batch", min=1)
        class ExportWrapper(torch.nn.Module):
            def __init__(self, detector):
                super().__init__()
                self.detector = detector

            def forward(self, images):
                return self.detector.export_outputs(images)

        wrapper = ExportWrapper(model.eval())
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            torch.onnx.export(
                wrapper,
                (torch.zeros(1, 3, request.image_size, request.image_size),),
                core,
                input_names=["images"],
                output_names=["boxes", "scores", "class_ids"],
                opset_version=request.opset,
                dynamo=True,
                dynamic_shapes=({0: batch},),
                external_data=False,
                optimize=True,
                verify=True,
                verbose=False,
            )
        outputs = embedded_output_paths(request.output)
        paths = wrap_embedded_variants(
            core,
            outputs,
            image_size=request.image_size,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        contract = detection_contract(classes, nms_required=False)
        checks: list[dict[str, Any]] = []
        for variant, path in outputs.items():
            variant_contract = detection_contract(classes, nms_required=False, input_variant=variant)
            set_onnx_metadata(path, metadata_for_contract(variant_contract))
            checks.extend({**check, "variant": variant} for check in validate_onnx(path, variant_contract))
        return Execution(tuple(artifact(path, "onnx") for path in paths), {}, contract, tuple(checks))

    def validate(self, request: ResolvedValidateRequest, context: WorkflowContext) -> BackendExecution:
        if request.target.suffix.casefold() == ".onnx":
            checks = validate_onnx(request.target, detection_contract([]))
            return Execution((artifact(request.target, "onnx"),), {}, detection_contract([]), checks)
        torch, _, _ = _torch_components()
        self._load(request.target, torch.device("cpu"))
        return Execution((artifact(request.target, "checkpoint"),), {}, {}, (({"name": "checkpoint_load", "status": "passed"}),))

    def test(self, request: ResolvedTestRequest, context: WorkflowContext) -> BackendExecution:
        torch, _, _ = _torch_components()
        dataset, test_set = self._dataset(request, request.split)
        model, _, _ = self._load(request.target, torch.device("cpu"))
        loader = torch.utils.data.DataLoader(test_set, 8, shuffle=False, collate_fn=_collate)
        metrics = self._evaluate(model, loader, torch.device("cpu"), len(dataset.classes))
        return Execution((artifact(request.target, "checkpoint"),), metrics, detection_contract(dataset.classes, nms_required=False), ())
