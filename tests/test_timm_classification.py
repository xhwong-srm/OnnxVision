from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vision_workflows.backends.timm_classification import TimmClassificationBackend
from vision_workflows.backends.timm_classification import Execution
from vision_workflows.workflows.context import WorkflowContext


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


def test_albumentations_robust_policy_adds_lighting_and_camera_variation(monkeypatch) -> None:
    numpy = pytest.importorskip("numpy")
    created: list[str] = []

    class FakePipeline:
        def __call__(self, *, image):
            return {"image": image}

    def operation(name):
        def factory(**kwargs):
            created.append(name)
            return (name, kwargs)

        return factory

    fake_albumentations = SimpleNamespace(
        HorizontalFlip=operation("HorizontalFlip"),
        Rotate=operation("Rotate"),
        RandomBrightnessContrast=operation("RandomBrightnessContrast"),
        RandomGamma=operation("RandomGamma"),
        GaussNoise=operation("GaussNoise"),
        GaussianBlur=operation("GaussianBlur"),
        Compose=lambda operations: FakePipeline(),
    )
    import vision_workflows.backends.timm_classification as integration

    monkeypatch.setattr(
        integration,
        "optional_import",
        lambda name: fake_albumentations if name == "albumentations" else numpy,
    )
    transform = TimmClassificationBackend._albumentations_augmentation(True, True, "robust")
    output = transform(numpy.zeros((2, 2, 3), dtype=numpy.uint8))

    assert output.size == (2, 2)
    assert created == [
        "HorizontalFlip",
        "Rotate",
        "RandomBrightnessContrast",
        "RandomGamma",
        "GaussNoise",
        "GaussianBlur",
    ]


def test_timm_tune_uses_optuna_trials_and_copies_best_checkpoints(monkeypatch, tmp_path) -> None:
    class Trial:
        def __init__(self, number: int):
            self.number = number
            self.params = {}
            self.user_attrs = {}
            self.value = None
            self.state = SimpleNamespace(name="COMPLETE")

        def suggest_float(self, name, low, high, log=False):
            value = (low * high) ** 0.5 if log else (low + high) / 2
            self.params[name] = value
            return value

        def report(self, value, step):
            pass

        def should_prune(self):
            return False

        def set_user_attr(self, name, value):
            self.user_attrs[name] = value

    class Study:
        study_name = "unit-study"

        def __init__(self):
            self.trials = []

        def optimize(self, objective, n_trials):
            for number in range(n_trials):
                trial = Trial(number)
                trial.value = objective(trial)
                self.trials.append(trial)

        @property
        def best_trial(self):
            return max(self.trials, key=lambda item: item.value)

        @property
        def best_value(self):
            return self.best_trial.value

        @property
        def best_params(self):
            return self.best_trial.params

    fake_optuna = SimpleNamespace(
        samplers=SimpleNamespace(TPESampler=lambda seed: SimpleNamespace(seed=seed)),
        pruners=SimpleNamespace(MedianPruner=lambda: object()),
        create_study=lambda **kwargs: Study(),
        TrialPruned=RuntimeError,
    )
    import vision_workflows.backends.timm_classification as integration
    monkeypatch.setattr(integration, "optional_import", lambda name: fake_optuna)

    def fake_train(self, request, context, trial=None):
        score = 0.5 + request.options["learning_rate"]
        (context.run_dir / "best.pt").write_bytes(b"best")
        (context.run_dir / "last.pt").write_bytes(b"last")
        return Execution(metrics={"best_val_accuracy": score})

    monkeypatch.setattr(TimmClassificationBackend, "_train", fake_train)
    run_dir = tmp_path / "tune"
    run_dir.mkdir()
    request = SimpleNamespace(
        selection=SimpleNamespace(),
        model=SimpleNamespace(),
        data=tmp_path / "data",
        output=run_dir,
        options={
            "trials": 2,
            "learning_rate_min": 1e-5,
            "learning_rate_max": 1e-3,
            "weight_decay_min": 0.0,
            "weight_decay_max": 0.1,
            "storage": None,
            "study_name": "unit-study",
            "seed": 7,
        },
        trials=2,
        learning_rate_min=1e-5,
        learning_rate_max=1e-3,
        weight_decay_min=0.0,
        weight_decay_max=0.1,
        storage=None,
        study_name="unit-study",
        seed=7,
    )
    context = WorkflowContext(run_dir, lambda name, values: None, "cpu")

    execution = TimmClassificationBackend().tune(request, context)

    assert (run_dir / "best.pt").read_bytes() == b"best"
    assert (run_dir / "last.pt").read_bytes() == b"last"
    assert (run_dir / "optuna.json").is_file()
    assert [item.kind for item in execution.artifacts] == ["checkpoint", "checkpoint", "report"]
    assert execution.metrics["completed_trials"] == 2
