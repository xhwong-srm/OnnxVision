from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vision_workflows.backends.timm_classification import TimmClassificationBackend


class _Classifier(torch.nn.Module):
    def forward(self, images):
        return torch.tensor([[5.0, 0.0], [0.0, 5.0]], device=images.device)[: images.shape[0]]


def test_timm_classification_evaluation_reports_loss_and_counts(monkeypatch, tmp_path) -> None:
    dataset = torch.utils.data.TensorDataset(
        torch.ones((2, 1)),
        torch.tensor([0, 1]),
    )
    dataset.classes = ["first", "second"]
    torchvision = SimpleNamespace(
        transforms=SimpleNamespace(
            Compose=lambda operations: None,
            Resize=lambda size: None,
            ToTensor=lambda: None,
            Normalize=lambda mean, std: None,
        ),
        datasets=SimpleNamespace(ImageFolder=lambda path, transform: dataset),
    )
    model = _Classifier()
    monkeypatch.setattr(
        TimmClassificationBackend,
        "_imports",
        staticmethod(lambda: (object(), torch, torchvision)),
    )
    monkeypatch.setattr(
        TimmClassificationBackend,
        "_load",
        lambda self, checkpoint, device: (model, ["first", "second"], {"data_config": {}}),
    )
    request = SimpleNamespace(
        target=tmp_path / "model.pt",
        data=tmp_path,
        split="val",
        device="cpu",
    )

    metrics = TimmClassificationBackend()._evaluate(request)

    assert metrics["split"] == "val"
    assert metrics["images"] == 2
    assert metrics["correct"] == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["loss"] > 0.0
