from __future__ import annotations

import json
import io
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
from .common import classification_contract, require_file, set_onnx_metadata, validate_onnx


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
    def _datasets(data: Path, image_size: int, train: bool):
        _, _, torchvision = TimmClassificationBackend._imports()
        transforms = torchvision.transforms
        operations = [transforms.Resize((image_size, image_size))]
        if train:
            operations.extend([transforms.RandomHorizontalFlip(), transforms.RandomRotation(5)])
        operations.extend([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        transform = transforms.Compose(operations)
        root = data / ("train" if train else "val")
        if not root.is_dir():
            raise FileNotFoundError(f"classification split does not exist: {root}")
        return torchvision.datasets.ImageFolder(str(root), transform=transform)

    def train(self, request: TrainRequest, context: WorkflowContext) -> BackendExecution:
        timm, torch, _ = self._imports()
        data = request.data.expanduser().resolve()
        train_set = self._datasets(data, request.image_size, True)
        val_set = self._datasets(data, request.image_size, False)
        if train_set.classes != val_set.classes:
            raise ValueError("train and val class folders differ")
        device = torch.device("cuda" if request.device in {"auto", "cuda"} and torch.cuda.is_available() else request.device if request.device != "auto" else "cpu")
        model = timm.create_model(request.model.variant, pretrained=request.pretrained, num_classes=len(train_set.classes))
        if request.weights:
            checkpoint = torch.load(require_file(request.weights, "weights"), map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device)
        loader = torch.utils.data.DataLoader(train_set, batch_size=request.batch, shuffle=True, num_workers=max(0, request.workers), generator=torch.Generator().manual_seed(request.seed))
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=request.batch, shuffle=False, num_workers=max(0, request.workers))
        optimizer = torch.optim.AdamW(model.parameters(), lr=request.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        best_accuracy = -1.0
        history = []
        for epoch in range(1, request.epochs + 1):
            model.train()
            if request.batch == 1:
                for layer in model.modules():
                    if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                        layer.eval()
            train_loss = 0.0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.detach().cpu())
            model.eval()
            correct = total = 0
            with torch.inference_mode():
                for images, labels in val_loader:
                    predictions = model(images.to(device)).argmax(1).cpu()
                    correct += int((predictions == labels).sum())
                    total += len(labels)
            accuracy = correct / max(1, total)
            row = {"epoch": epoch, "train_loss": train_loss / max(1, len(loader)), "val_accuracy": accuracy}
            history.append(row)
            payload = {"task": "classification", "model_name": request.model.variant, "classes": train_set.classes, "data_config": {"input_size": [3, request.image_size, request.image_size], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}, "epoch": epoch, "model_state_dict": model.state_dict(), "metrics": row}
            torch.save(payload, context.run_dir / "last.pt")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(payload, context.run_dir / "best.pt")
        context.write_json("metrics.json", {"history": history, "best_val_accuracy": best_accuracy, "classes": train_set.classes})
        return Execution(tuple(artifact(context.run_dir / name, "checkpoint") for name in ("best.pt", "last.pt")), {"best_val_accuracy": best_accuracy, "epochs": request.epochs})

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
        set_onnx_metadata(output, {"contract_version": contract["version"], "names": contract["names"]})
        checks = validate_onnx(output, contract)
        return Execution((artifact(output, "onnx"),), {}, contract, checks)

    def _evaluate(self, request: ValidateRequest | TestRequest):
        _, torch, _ = self._imports()
        model, classes, value = self._load(request.target, request.device)
        data = request.data
        split = request.split
        _, _, torchvision = self._imports()
        transform_ops = torchvision.transforms.Compose([
            torchvision.transforms.Resize((int(value.get("data_config", {}).get("input_size", [3, 224, 224])[-1]),) * 2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
