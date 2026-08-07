from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

import vision_workflows.backends.export_validation as validation


class _Session:
    def __init__(self, path: str, *, detection: bool = False):
        self.detection = detection

    def get_inputs(self):
        return [SimpleNamespace(name="images")]

    def run(self, _outputs, feed):
        batch = next(iter(feed.values())).shape[0]
        if self.detection:
            return [
                np.tile(np.asarray([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32), (batch, 1, 1)),
                np.full((batch, 1), 0.9, dtype=np.float32),
                np.zeros((batch, 1), dtype=np.int64),
            ]
        return [np.tile(np.asarray([[0.9, 0.1]], dtype=np.float32), (batch, 1))]


def test_classification_wrappers_report_accuracy_and_agreement(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    validation_root = tmp_path / "val"
    for class_name in ("first", "second"):
        (validation_root / class_name).mkdir(parents=True)
    first = validation_root / "first" / "one.png"
    second = validation_root / "second" / "two.png"
    Image.new("RGB", (4, 4), "white").save(first)
    Image.new("RGB", (4, 4), "black").save(second)
    assert validation._classification_raw_image(str(first), "bw8", np).shape == (1, 4, 4)
    assert validation._classification_raw_image(str(first), "c24", np).shape == (4, 4, 3)

    class Dataset:
        classes = ["first", "second"]
        samples = [(str(first), 0), (str(second), 1)]

        def __len__(self):
            return len(self.samples)

    def image_folder(_path):
        return Dataset()

    fake_torchvision = SimpleNamespace(datasets=SimpleNamespace(ImageFolder=image_folder))

    def fake_import(name: str):
        if name == "numpy":
            return np
        if name == "onnxruntime":
            return SimpleNamespace(InferenceSession=lambda path, providers: _Session(str(path)))
        if name == "torchvision":
            return fake_torchvision
        if name == "PIL.Image":
            return Image
        return importlib.import_module(name)

    monkeypatch.setattr(validation, "optional_import", fake_import)
    reference = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    metrics = validation.validate_classification_wrappers(
        {"bw8": tmp_path / "model-bw8.onnx", "c24": tmp_path / "model-c24.onnx"},
        tmp_path,
        classes=["first", "second"],
        image_size=4,
        batch_size=None,
        reference_probabilities=reference,
    )

    assert metrics["bw8_accuracy"] == 0.5
    assert metrics["c24_accuracy"] == 0.5
    assert metrics["bw8_c24_agreement"] == 1.0
    assert metrics["bw8_native_agreement"] == 0.5


def test_detection_wrappers_report_map_and_variant_agreement(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    images = tmp_path / "images" / "val"
    labels = tmp_path / "labels" / "val"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    image = images / "sample.png"
    Image.new("RGB", (4, 4), "white").save(image)
    assert validation._raw_image(image, "bw8", np).shape == (1, 4, 4)
    assert validation._raw_image(image, "c24", np).shape == (4, 4, 3)
    (labels / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("path: .\nval: images/val\nnames: [seal]\n", encoding="utf-8")

    def fake_import(name: str):
        if name == "numpy":
            return np
        if name == "onnxruntime":
            return SimpleNamespace(InferenceSession=lambda path, providers: _Session(str(path), detection=True))
        if name == "PIL.Image":
            return Image
        return importlib.import_module(name)

    monkeypatch.setattr(validation, "optional_import", fake_import)
    metrics = validation.validate_detection_wrappers(
        {"bw8": tmp_path / "model-bw8.onnx", "c24": tmp_path / "model-c24.onnx"},
        data_yaml,
        class_count=1,
        image_size=4,
        batch_size=1,
    )

    assert metrics["bw8_map50"] == pytest.approx(1.0)
    assert metrics["c24_recall50"] == pytest.approx(1.0)
    assert metrics["bw8_c24_agreement50"] == pytest.approx(1.0)
