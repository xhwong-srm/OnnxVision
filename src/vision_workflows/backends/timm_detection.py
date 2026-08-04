from __future__ import annotations

import json
import io
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain.datasets import DatasetFormat, DetectionDataset, Split
from ..domain.models import BackendCapability, BackendDescriptor
from ..domain.results import ArtifactRef
from ..workflows.context import WorkflowContext, optional_import
from ..workflows.requests import ExportRequest, TestRequest, TrainRequest, ValidateRequest
from ..workflows.runs import artifact
from .base import BackendExecution, ModelBackend
from .common import detection_contract, metadata_for_contract, require_file, set_onnx_metadata, validate_onnx


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


class TimmDetectionBackend(ModelBackend):
    descriptor = BackendDescriptor(
        "timm", "detection", "detection", ("query",),
        frozenset(BackendCapability), "timm NMS-free query detector", "timm,torchvision",
    )

    def _dataset(self, request: TrainRequest | ValidateRequest | TestRequest, split: str):
        from ..datasets.formats.base import load_dataset

        dataset = load_dataset(request.data, DatasetFormat.COCO)
        return dataset, TimmDetectionDataset(dataset, split, request.image_size if hasattr(request, "image_size") else int(request.options.get("image_size", 384)))

    def train(self, request: TrainRequest, context: WorkflowContext) -> BackendExecution:
        torch, timm, _ = _torch_components()
        dataset, train_set = self._dataset(request, "train")
        _, val_set = self._dataset(request, "val")
        queries = int(request.options.get("num_queries", max(8, max((len(sample.annotations) for sample in dataset.samples), default=1) + 4)))
        variant = str(request.options.get("backbone", "mobilenetv4_conv_small.e3600_r256_in1k"))
        device = torch.device("cuda" if request.device == "auto" and torch.cuda.is_available() else request.device if request.device != "auto" else "cpu")
        model = _model_class(torch, timm, len(dataset.classes), variant, queries, request.pretrained)()
        model.to(device)
        loader = torch.utils.data.DataLoader(train_set, request.batch, shuffle=True, collate_fn=_collate, num_workers=max(0, request.workers))
        val_loader = torch.utils.data.DataLoader(val_set, request.batch, shuffle=False, collate_fn=_collate, num_workers=max(0, request.workers))
        optimizer = torch.optim.AdamW(model.parameters(), lr=request.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        best = -1.0
        history = []
        for epoch in range(1, request.epochs + 1):
            model.train()
            loss_total = 0.0
            for images, boxes, labels in loader:
                images = images.to(device)
                logits, predicted_boxes = model(images)
                target_classes = torch.full(logits.shape[:2], len(dataset.classes), dtype=torch.long, device=device)
                target_boxes = torch.zeros_like(predicted_boxes)
                for index, (sample_boxes, sample_labels) in enumerate(zip(boxes, labels)):
                    count = min(queries, len(sample_labels))
                    if count:
                        target_classes[index, :count] = sample_labels[:count].to(device)
                        target_boxes[index, :count] = sample_boxes[:count].to(device)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), target_classes.reshape(-1))
                positive = target_classes != len(dataset.classes)
                if positive.any():
                    loss = loss + torch.nn.functional.l1_loss(predicted_boxes[positive], target_boxes[positive])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_total += float(loss.detach().cpu())
            metrics = self._evaluate(model, val_loader, device, len(dataset.classes))
            row = {"epoch": epoch, "loss": loss_total / max(1, len(loader)), **metrics}
            history.append(row)
            payload = {"task": "detection", "architecture": "query", "model_name": variant, "classes": list(dataset.classes), "num_queries": queries, "image_size": request.image_size, "model_state_dict": model.state_dict(), "metrics": row}
            torch.save(payload, context.run_dir / "last.pt")
            if metrics["mean_score"] > best:
                best = metrics["mean_score"]
                torch.save(payload, context.run_dir / "best.pt")
        context.write_json("metrics.json", {"history": history, "best_mean_score": best})
        return Execution(tuple(artifact(context.run_dir / name, "checkpoint") for name in ("best.pt", "last.pt")), {"best_mean_score": best})

    def _load(self, target: Path, device):
        torch, timm, _ = _torch_components()
        checkpoint = torch.load(require_file(target, "detection checkpoint"), map_location=device, weights_only=False)
        classes = list(checkpoint["classes"])
        variant = checkpoint["model_name"]
        model = _model_class(torch, timm, len(classes), variant, int(checkpoint["num_queries"]), False)()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return model, classes, checkpoint

    @staticmethod
    def _evaluate(model, loader, device, class_count):
        torch = __import__("torch")
        model.eval()
        scores = []
        with torch.inference_mode():
            for images, _, _ in loader:
                logits, _ = model(images.to(device))
                scores.extend(torch.softmax(logits, -1)[..., :class_count].max(-1).values.mean(-1).cpu().tolist())
        return {"images": len(scores), "mean_score": sum(scores) / max(1, len(scores))}

    def export(self, request: ExportRequest, context: WorkflowContext) -> BackendExecution:
        torch, _, _ = _torch_components()
        model, classes, checkpoint = self._load(request.checkpoint, torch.device("cpu"))
        output = request.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        class ExportWrapper(torch.nn.Module):
            def __init__(self, detector):
                super().__init__()
                self.detector = detector

            def forward(self, images):
                return self.detector.export_outputs(images)

        wrapper = ExportWrapper(model.eval())
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            torch.onnx.export(wrapper, torch.zeros(1, 3, request.image_size, request.image_size), output, input_names=["images"], output_names=["boxes", "scores", "class_ids"], opset_version=request.opset, dynamo=True)
        contract = detection_contract(classes, nms_required=False)
        set_onnx_metadata(output, metadata_for_contract(contract))
        return Execution((artifact(output, "onnx"),), {}, contract, validate_onnx(output, contract))

    def validate(self, request: ValidateRequest, context: WorkflowContext) -> BackendExecution:
        if request.target.suffix.casefold() == ".onnx":
            checks = validate_onnx(request.target, detection_contract([]))
            return Execution((artifact(request.target, "onnx"),), {}, detection_contract([]), checks)
        torch, _, _ = _torch_components()
        self._load(request.target, torch.device("cpu"))
        return Execution((artifact(request.target, "checkpoint"),), {}, {}, (({"name": "checkpoint_load", "status": "passed"}),))

    def test(self, request: TestRequest, context: WorkflowContext) -> BackendExecution:
        torch, _, _ = _torch_components()
        dataset, test_set = self._dataset(request, request.split)
        model, _, _ = self._load(request.target, torch.device("cpu"))
        loader = torch.utils.data.DataLoader(test_set, 8, shuffle=False, collate_fn=_collate)
        metrics = self._evaluate(model, loader, torch.device("cpu"), len(dataset.classes))
        return Execution((artifact(request.target, "checkpoint"),), metrics, detection_contract(dataset.classes, nms_required=False), ())
