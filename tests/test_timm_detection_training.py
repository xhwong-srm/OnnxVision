from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vision_workflows.backends import timm_detection as backend_module
from vision_workflows.backends.timm_detection import TimmDetectionBackend
from vision_workflows.domain.datasets import TaskKind
from vision_workflows.domain.models import ModelInfo, ModelSelection
from vision_workflows.workflows.requests import ResolvedTrainRequest


class _EmptyDetectionSet(torch.utils.data.Dataset):
    image_size = 8

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return torch.zeros(3, 8, 8), torch.empty((0, 4)), torch.empty((0,), dtype=torch.long)


class _FakeDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(1, 2))
        self.boxes = torch.nn.Parameter(torch.zeros(1, 4))

    def forward(self, images):
        batch = images.shape[0]
        return (
            self.logits.unsqueeze(0).expand(batch, -1, -1),
            self.boxes.sigmoid().unsqueeze(0).expand(batch, -1, -1),
        )


class _Context:
    def __init__(self, run_dir):
        self.run_dir = run_dir

    def write_json(self, name, value):
        path = self.run_dir / name
        path.write_text(json.dumps(value, default=str), encoding="utf-8")
        return path


def test_timm_detection_supports_resume_patience_and_determinism(monkeypatch, tmp_path) -> None:
    dataset = SimpleNamespace(classes=("seal",), samples=())
    data_set = _EmptyDetectionSet()
    scores = [1.0, 1.0]

    monkeypatch.setattr(backend_module, "_torch_components", lambda: (torch, object(), object()))
    monkeypatch.setattr(backend_module, "_model_class", lambda *args, **kwargs: _FakeDetector)
    monkeypatch.setattr(TimmDetectionBackend, "_dataset", lambda self, request, split: (dataset, data_set))

    def evaluate(*args, **kwargs):
        return {"images": 2, "mean_score": scores.pop(0)}

    monkeypatch.setattr(TimmDetectionBackend, "_evaluate", staticmethod(evaluate))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    context = _Context(run_dir)
    parameters = {
        "epochs": 5, "batch": 1, "image_size": 8, "learning_rate": 1e-3,
        "workers": 0, "patience": 1, "seed": 7, "device": "cpu",
        "pretrained": False, "deterministic": True, "num_queries": 1,
        "backbone": "unused", "validate_every": 1, "prefetch_factor": None,
        "persistent_workers": False, "pin_memory": False, "amp": False,
        "amp_dtype": None, "compile": False,
    }
    request = ResolvedTrainRequest(
        ModelSelection(TaskKind.OBJECT_DETECTION, "pytorch", "timm-obd-v1"),
        ModelInfo("timm-obd-v1", "timm-obd-v1"),
        tmp_path / "data",
        run_dir,
        None,
        False,
        False,
        parameters,
    )

    first = TimmDetectionBackend().train(request, context)

    assert first.metrics["epochs"] == 2
    assert first.metrics["best_epoch"] == 1
    assert first.metrics["stopped_early"] is True
    checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 2
    assert checkpoint["optimizer_state_dict"] is not None
    assert checkpoint["rng_state"] is not None

    scores[:] = [0.5]
    resumed = TimmDetectionBackend().train(
        request.__class__(request.selection, request.model, request.data, request.output, request.weights, True, request.overwrite, {**request.parameters, "epochs": 3}),
        context,
    )

    assert resumed.metrics["epochs"] == 3
    assert resumed.metrics["best_epoch"] == 1
    assert resumed.metrics["stopped_early"] is True
