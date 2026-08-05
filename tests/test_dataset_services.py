from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PIL import Image

from vision_workflows.api import (
    ConvertDatasetRequest,
    DatasetService,
    MergeDatasetRequest,
    SplitDatasetRequest,
    ValidateDatasetRequest,
)
from vision_workflows.domain.datasets import DatasetFormat, MaterializationMode, SplitPolicy, TaskKind
from vision_workflows.datasets import service as dataset_service


def test_task_kind_is_not_domain_specific() -> None:
    assert TaskKind.CLASSIFICATION.value == "classification"
    assert TaskKind.OBJECT_DETECTION.value == "object-detection"


def image(path: Path, size: tuple[int, int] = (100, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (20, 30, 40)).save(path)


def coco_source(root: Path) -> None:
    for split in ("train", "val", "test"):
        image_path = root / "images" / split / f"{split}.png"
        image(image_path)
        (root / "annotations").mkdir(parents=True, exist_ok=True)
        document = {
            "images": [{"id": 1, "file_name": f"images/{split}/{split}.png", "width": 100, "height": 80}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 10, "bbox": [10, 20, 40, 30]}] if split == "train" else [],
            "categories": [{"id": 10, "name": "seal"}, {"id": 20, "name": "defect"}],
        }
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(document), encoding="utf-8")


def test_classification_split_and_manifest_are_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for index in range(6):
        image(source / ("seal" if index % 2 else "defect") / f"image-{index}.png")
    service = DatasetService()
    request = SplitDatasetRequest(
        source,
        tmp_path / "split-a",
        SplitPolicy(0.5, 0.25, 0.25, seed=7, grouping="sample"),
        materialization=MaterializationMode.COPY,
    )
    first = service.split(request)
    second = service.split(request.__class__(source, tmp_path / "split-b", request.policy, materialization=MaterializationMode.COPY))
    assert first.split_counts == second.split_counts
    assert (first.output / "dataset_manifest.json").is_file()
    assert sorted(path.relative_to(first.output).as_posix() for path in first.output.rglob("*.png")) == sorted(path.relative_to(second.output).as_posix() for path in second.output.rglob("*.png"))


def test_merge_reconciles_classes_case_insensitively(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    image(first / "train" / "Seal" / "a.png")
    image(second / "train" / "seal" / "b.png")
    result = DatasetService().merge(MergeDatasetRequest((first, second), tmp_path / "merged", materialization=MaterializationMode.COPY))
    assert result.classes == ("Seal",)
    assert result.sample_count == 2
    assert result.split_counts["train"] == 2


def test_coco_yolo_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "coco"
    coco_source(source)
    service = DatasetService()
    yolo = service.convert(ConvertDatasetRequest(source, tmp_path / "yolo", DatasetFormat.YOLO, materialization=MaterializationMode.COPY))
    assert yolo.split_counts == {"train": 1, "val": 1, "test": 1}
    assert (yolo.output / "data.yaml").is_file()
    report = service.validate(ValidateDatasetRequest(yolo.output, DatasetFormat.YOLO, require_train_val=True))
    assert report.valid, report.issues
    back = service.convert(ConvertDatasetRequest(yolo.output, tmp_path / "back", DatasetFormat.COCO, materialization=MaterializationMode.COPY))
    assert back.split_counts == yolo.split_counts


def test_convert_handles_broken_output_symlink(tmp_path: Path) -> None:
    source = tmp_path / "coco"
    coco_source(source)
    output = tmp_path / "yolo"
    output.symlink_to(tmp_path / "missing-output", target_is_directory=True)
    service = DatasetService()

    with pytest.raises(FileExistsError, match="output already exists"):
        service.convert(ConvertDatasetRequest(source, output, DatasetFormat.YOLO, materialization=MaterializationMode.COPY))

    result = service.convert(ConvertDatasetRequest(source, output, DatasetFormat.YOLO, materialization=MaterializationMode.COPY, overwrite=True))
    assert result.output.is_dir()
    assert not result.output.is_symlink()
    assert (result.output / "data.yaml").is_file()


def test_finalize_detaches_hardlinks_after_permission_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.png"
    image(source)
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    hardlink = staging / "image.png"
    os.link(source, hardlink)
    real_replace = type(staging).replace
    calls = 0

    def replace(path: Path, target: Path) -> Path:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError(5, "Access is denied")
        return real_replace(path, target)

    monkeypatch.setattr(type(staging), "replace", replace)
    dataset_service._finalize(staging, output, None)

    assert calls == 2
    assert output.is_dir()
    assert os.stat(source).st_nlink == 1
    assert (output / "image.png").read_bytes() == source.read_bytes()


def test_detection_validation_rejects_invalid_box(tmp_path: Path) -> None:
    source = tmp_path / "coco"
    coco_source(source)
    annotation = source / "annotations" / "instances_train.json"
    document = json.loads(annotation.read_text(encoding="utf-8"))
    document["annotations"][0]["bbox"] = [-1, 0, 20, 20]
    annotation.write_text(json.dumps(document), encoding="utf-8")
    report = DatasetService().validate(ValidateDatasetRequest(source, DatasetFormat.COCO))
    assert not report.valid
    assert any(issue.code == "bounding_box" for issue in report.errors)


def test_neurocle_and_rfdetr_outputs_are_readable(tmp_path: Path) -> None:
    source = tmp_path / "coco"
    coco_source(source)
    service = DatasetService()
    neurocle = service.convert(ConvertDatasetRequest(source, tmp_path / "neurocle", DatasetFormat.NEUROCLE, materialization=MaterializationMode.COPY))
    assert (neurocle.output / "neurocle_labeling.json").is_file()
    assert service.validate(ValidateDatasetRequest(neurocle.output, DatasetFormat.NEUROCLE)).valid
    rfdetr = service.convert(ConvertDatasetRequest(source, tmp_path / "rfdetr", DatasetFormat.RFDETR, materialization=MaterializationMode.COPY))
    assert (rfdetr.output / "valid" / "_annotations.coco.json").is_file()
    assert service.validate(ValidateDatasetRequest(rfdetr.output, DatasetFormat.RFDETR, require_train_val=True)).valid
