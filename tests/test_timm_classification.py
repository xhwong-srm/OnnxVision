from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from vision_workflows.backends.timm_classification import (
    Execution,
    ResizedImageLoader,
    TimmClassificationBackend,
    _classification_metrics,
    _create_scheduler,
)
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
    assert metrics["macro_f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["loss"] > 0.0


def test_classification_metrics_report_macro_and_per_class_values() -> None:
    metrics = _classification_metrics(
        torch.tensor([[2, 1], [0, 1]], dtype=torch.int64),
        ["first", "second"],
    )

    assert metrics["macro_precision"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx((0.8 + (2 / 3)) / 2)
    assert metrics["balanced_accuracy"] == pytest.approx((2 / 3 + 1.0) / 2)
    assert metrics["per_class_recall"] == {"first": pytest.approx(2 / 3), "second": 1.0}


def test_scheduler_warms_up_then_decays_cosine() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = _create_scheduler(torch, optimizer, total_epochs=6, warmup_epochs=2)
    learning_rates = []

    for _ in range(6):
        learning_rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert learning_rates[0] < learning_rates[1] < learning_rates[2]
    assert learning_rates[-1] < learning_rates[2]


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


@pytest.mark.parametrize("mode", ["ram", "disk"])
def test_resized_image_loader_caches_rgb_images(tmp_path, mode) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 6), (20, 40, 60)).save(source)
    cache_dir = tmp_path / "cache" if mode == "disk" else None
    loader = ResizedImageLoader(mode, cache_dir, 4, lambda image: image.resize((4, 4)))

    loader.prepare((str(source),))
    output = loader(str(source))

    assert output.mode == "RGB"
    assert output.size == (4, 4)
    if mode == "disk":
        assert (cache_dir / "manifest.json").is_file()
        assert (cache_dir / "000000.npy").is_file()


def test_albumentations_runs_after_common_resize(monkeypatch, tmp_path) -> None:
    operations: list[str] = []

    class Transform:
        def __init__(self, name):
            self.name = name
            operations.append(name)

    class Transforms:
        Resize = lambda self, size: Transform("Resize")
        ToTensor = lambda self: Transform("ToTensor")
        Normalize = lambda self, mean, std: Transform("Normalize")
        Compose = lambda self, values: values

    root = tmp_path / "train"
    root.mkdir()
    dataset = SimpleNamespace(classes=["class"], samples=[])
    torchvision = SimpleNamespace(
        transforms=Transforms(),
        datasets=SimpleNamespace(ImageFolder=lambda path, **kwargs: dataset),
    )
    monkeypatch.setattr(
        TimmClassificationBackend,
        "_imports",
        staticmethod(lambda: (object(), torch, torchvision)),
    )
    monkeypatch.setattr(
        TimmClassificationBackend,
        "_albumentations_augmentation",
        staticmethod(lambda train, enabled, policy: Transform("Albumentations")),
    )

    TimmClassificationBackend._datasets(
        tmp_path,
        4,
        True,
        augmentation_backend="albumentations",
    )

    assert operations == ["Resize", "Albumentations", "ToTensor", "Normalize"]


def test_resized_disk_cache_integrates_with_albumentations(monkeypatch, tmp_path) -> None:
    torchvision = pytest.importorskip("torchvision")
    pytest.importorskip("albumentations")
    data = tmp_path / "data"
    (data / "train" / "class").mkdir(parents=True)
    (data / "val" / "class").mkdir(parents=True)
    Image.new("RGB", (8, 6), (20, 40, 60)).save(data / "train" / "class" / "source.png")
    Image.new("RGB", (8, 6), (60, 40, 20)).save(data / "val" / "class" / "source.png")
    monkeypatch.setattr(
        TimmClassificationBackend,
        "_imports",
        staticmethod(lambda: (object(), torch, torchvision)),
    )

    dataset = TimmClassificationBackend._datasets(
        data,
        4,
        True,
        augmentation_backend="albumentations",
        cache_mode="disk",
    )
    image, label = dataset[0]

    assert tuple(image.shape) == (3, 4, 4)
    assert label == 0
    validation_dataset = TimmClassificationBackend._datasets(
        data,
        4,
        False,
        augmentation_enabled=False,
        cache_mode="disk",
    )
    validation_image, validation_label = validation_dataset[0]

    assert tuple(validation_image.shape) == (3, 4, 4)
    assert validation_label == 0
    assert (data / ".vision_workflows_cache" / "timm_classification" / "size-4" / "val" / "manifest.json").is_file()


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
        samplers=SimpleNamespace(TPESampler=lambda **kwargs: SimpleNamespace(**kwargs)),
        pruners=SimpleNamespace(MedianPruner=lambda **kwargs: SimpleNamespace(**kwargs)),
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
            "label_smoothing_min": 0.0,
            "label_smoothing_max": 0.1,
            "storage": None,
            "study_name": "unit-study",
            "seed": 7,
        },
        trials=2,
        learning_rate_min=1e-5,
        learning_rate_max=1e-3,
        weight_decay_min=0.0,
        weight_decay_max=0.1,
        label_smoothing_min=0.0,
        label_smoothing_max=0.1,
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
