from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..domain.datasets import (
    BalanceMode,
    ClassificationDataset,
    ClassificationSample,
    DatasetFormat,
    DatasetInspection,
    DatasetValidationIssue,
    DatasetValidationReport,
    DetectionAnnotation,
    DetectionDataset,
    DetectionSample,
    MaterializationMode,
    SplitPolicy,
    TaskKind,
)
from ..domain.errors import ConfigurationError, DatasetFormatError, ValidationFailedError
from ..domain.results import ConversionResult, MergeResult, SplitResult
from .formats.base import detect_format, load_dataset, write_dataset
from .splitting import split_samples


@dataclass(frozen=True)
class ConvertDatasetRequest:
    source: Path
    output: Path
    output_format: DatasetFormat
    input_format: DatasetFormat | None = None
    materialization: MaterializationMode = MaterializationMode.HARDLINK
    overwrite: bool = False


@dataclass(frozen=True)
class ValidateDatasetRequest:
    source: Path
    format: DatasetFormat | None = None
    require_train_val: bool = False


@dataclass(frozen=True)
class SplitDatasetRequest:
    source: Path
    output: Path
    policy: SplitPolicy = SplitPolicy()
    format: DatasetFormat | None = None
    output_format: DatasetFormat | None = None
    materialization: MaterializationMode = MaterializationMode.HARDLINK
    overwrite: bool = False


@dataclass(frozen=True)
class MergeDatasetRequest:
    sources: tuple[Path, ...]
    output: Path
    output_format: DatasetFormat | None = None
    split_policy: SplitPolicy | None = None
    materialization: MaterializationMode = MaterializationMode.HARDLINK
    overwrite: bool = False
    balance: BalanceMode = BalanceMode.NONE


def _class_index(classes: tuple[str, ...]) -> dict[str, int]:
    return {name.casefold(): index for index, name in enumerate(classes)}


def _split_counts(samples) -> dict[str, int]:
    return {
        split: sum(1 for sample in samples if sample.split is not None and sample.split.value == split)
        for split in ("train", "val", "test")
    }


def _merge_classes(datasets: list[ClassificationDataset | DetectionDataset]) -> tuple[tuple[str, ...], list[dict[int, int]]]:
    names: list[str] = []
    lookup: dict[str, int] = {}
    mappings = []
    for dataset in datasets:
        mapping: dict[int, int] = {}
        for index, name in enumerate(dataset.classes):
            key = name.casefold()
            if key not in lookup:
                lookup[key] = len(names)
                names.append(name)
            mapping[index] = lookup[key]
        mappings.append(mapping)
    return tuple(names), mappings


def _remap_dataset(dataset, classes: tuple[str, ...], mapping: dict[int, int]):
    if isinstance(dataset, ClassificationDataset):
        return ClassificationDataset(
            classes,
            tuple(replace(sample, class_id=mapping[sample.class_id], class_name=classes[mapping[sample.class_id]]) for sample in dataset.samples),
            dataset.source,
            dataset.source_format,
            dataset.metadata,
        )
    return DetectionDataset(
        classes,
        tuple(replace(sample, annotations=tuple(replace(annotation, class_id=mapping[annotation.class_id]) for annotation in sample.annotations)) for sample in dataset.samples),
        dataset.source,
        dataset.source_format,
        dataset.metadata,
    )


def _balance_classification(dataset: ClassificationDataset, mode: BalanceMode) -> ClassificationDataset:
    if mode == BalanceMode.NONE:
        return dataset
    by_class = {index: [sample for sample in dataset.samples if sample.class_id == index and sample.split == "train"] for index in range(len(dataset.classes))}
    counts = [len(values) for values in by_class.values() if values]
    if not counts:
        return dataset
    target = min(counts) if mode == BalanceMode.UNDERSAMPLE else max(counts)
    selected = [sample for sample in dataset.samples if sample.split != "train"]
    for class_id, values in by_class.items():
        if mode == BalanceMode.UNDERSAMPLE:
            selected.extend(values[:target])
        else:
            selected.extend(values)
            for index in range(target - len(values)):
                selected.append(replace(values[index % len(values)], group_key=(*values[index % len(values)].group_key, f"copy-{index}")))
    return replace(dataset, samples=tuple(selected))


def _manifest(dataset, output: Path, *, operation: str, extra: dict[str, Any] | None = None) -> Path:
    samples = dataset.samples
    document: dict[str, Any] = {
        "schema_version": 1,
        "operation": operation,
        "task": dataset.task.value,
        "format": dataset.source_format.value,
        "source": str(dataset.source) if dataset.source else None,
        "classes": list(dataset.classes),
        "sample_count": len(samples),
        "split_counts": _split_counts(samples),
        "samples": [
            {
                "path": str(sample.image.path),
                "split": sample.split.value if sample.split else None,
                "group_key": list(sample.group_key),
                "sha256": sample.image.sha256,
            }
            for sample in samples
        ],
    }
    if extra:
        document.update(extra)
    path = output / "dataset_manifest.json"
    path.write_text(json.dumps(document, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def _prepare_output(output: Path, overwrite: bool) -> tuple[Path, Path | None]:
    output = output.expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists; use overwrite=True: {output}")
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging, output if output.exists() else None


def _finalize(staging: Path, output: Path, old: Path | None) -> None:
    if old is not None:
        shutil.rmtree(old)
    staging.replace(output)


class DatasetService:
    def inspect(self, source: Path, format: DatasetFormat | None = None) -> DatasetInspection:
        selected = format or detect_format(source)
        dataset = load_dataset(source, selected)
        split_counts = _split_counts(dataset.samples)
        annotations = sum(len(sample.annotations) for sample in dataset.samples) if isinstance(dataset, DetectionDataset) else 0
        return DatasetInspection(source.expanduser().resolve(), selected, dataset.task, dataset.classes, split_counts, len(dataset.samples), annotations)

    def validate(self, request: ValidateDatasetRequest) -> DatasetValidationReport:
        selected = request.format or detect_format(request.source)
        issues: list[DatasetValidationIssue] = []
        try:
            dataset = load_dataset(request.source, selected)
        except (DatasetFormatError, OSError, ValueError) as error:
            return DatasetValidationReport(False, None, selected, (DatasetValidationIssue("error", "load_failed", str(error), request.source),))
        if request.require_train_val and not {sample.split.value for sample in dataset.samples if sample.split} >= {"train", "val"}:
            issues.append(DatasetValidationIssue("error", "missing_required_split", "dataset must contain train and val samples", request.source))
        if len({name.casefold() for name in dataset.classes}) != len(dataset.classes):
            issues.append(DatasetValidationIssue("error", "duplicate_class", "class names must be unique", request.source))
        for sample in dataset.samples:
            if not sample.image.path.is_file():
                issues.append(DatasetValidationIssue("error", "missing_image", f"image does not exist: {sample.image.path}", sample.image.path))
            if isinstance(sample, DetectionSample):
                for annotation in sample.annotations:
                    box = annotation.box
                    if not 0 <= annotation.class_id < len(dataset.classes):
                        issues.append(DatasetValidationIssue("error", "class_id", f"invalid class ID {annotation.class_id}", sample.image.path))
                    if box.x1 < 0 or box.y1 < 0 or box.x2 > sample.image.width or box.y2 > sample.image.height or box.x2 <= box.x1 or box.y2 <= box.y1:
                        issues.append(DatasetValidationIssue("error", "bounding_box", f"invalid box {box}", sample.image.path))
        return DatasetValidationReport(not any(issue.severity == "error" for issue in issues), dataset.task, selected, tuple(issues), len(dataset.samples), len(dataset.classes))

    def convert(self, request: ConvertDatasetRequest) -> ConversionResult:
        selected = request.input_format or detect_format(request.source)
        dataset = load_dataset(request.source, selected)
        if isinstance(dataset, ClassificationDataset) and request.output_format != DatasetFormat.IMAGE_FOLDER:
            raise ConfigurationError("classification conversion only supports image-folder output")
        staging, old = _prepare_output(request.output, request.overwrite)
        try:
            write_dataset(dataset, staging, request.output_format, request.materialization)
            manifest = _manifest(dataset, staging, operation="convert", extra={"input_format": selected.value, "output_format": request.output_format.value})
            _finalize(staging, request.output.expanduser().resolve(), old)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        counts = _split_counts(dataset.samples)
        return ConversionResult(selected.value, request.output_format.value, request.output.expanduser().resolve(), dataset.classes, counts, request.output.expanduser().resolve() / manifest.name)

    def split(self, request: SplitDatasetRequest) -> SplitResult:
        dataset = load_dataset(request.source, request.format)
        if isinstance(dataset, ClassificationDataset):
            split_dataset = replace(dataset, samples=split_samples(dataset.samples, len(dataset.classes), request.policy))
        else:
            split_dataset = replace(dataset, samples=split_samples(dataset.samples, len(dataset.classes), request.policy))
        output_format = request.output_format or dataset.source_format
        staging, old = _prepare_output(request.output, request.overwrite)
        try:
            write_dataset(split_dataset, staging, output_format, request.materialization)
            manifest = _manifest(split_dataset, staging, operation="split", extra={"seed": request.policy.seed, "grouping": request.policy.grouping, "ratios": request.policy.ratios()})
            _finalize(staging, request.output.expanduser().resolve(), old)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        counts = _split_counts(split_dataset.samples)
        return SplitResult(request.output.expanduser().resolve(), request.output.expanduser().resolve() / manifest.name, counts, request.policy.seed, request.policy.grouping)

    def merge(self, request: MergeDatasetRequest) -> MergeResult:
        if not request.sources:
            raise ConfigurationError("merge requires at least one source")
        datasets = [load_dataset(source) for source in request.sources]
        tasks = {dataset.task for dataset in datasets}
        if len(tasks) != 1:
            raise ConfigurationError("all merged datasets must use the same task")
        classes, mappings = _merge_classes(datasets)
        remapped = [_remap_dataset(dataset, classes, mapping) for dataset, mapping in zip(datasets, mappings)]
        first = remapped[0]
        if isinstance(first, ClassificationDataset):
            merged = ClassificationDataset(classes, tuple(sample for dataset in remapped for sample in dataset.samples), None, DatasetFormat.IMAGE_FOLDER)
            if request.split_policy is not None:
                merged = replace(merged, samples=split_samples(merged.samples, len(classes), request.split_policy))
            merged = _balance_classification(merged, request.balance)
        else:
            merged = DetectionDataset(classes, tuple(sample for dataset in remapped for sample in dataset.samples), None, request.output_format or first.source_format)
            if request.split_policy is not None:
                merged = replace(merged, samples=split_samples(merged.samples, len(classes), request.split_policy))
        output_format = request.output_format or merged.source_format
        staging, old = _prepare_output(request.output, request.overwrite)
        try:
            write_dataset(merged, staging, output_format, request.materialization)
            manifest = _manifest(merged, staging, operation="merge", extra={"sources": [str(source.resolve()) for source in request.sources], "output_format": output_format.value})
            _finalize(staging, request.output.expanduser().resolve(), old)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        counts = _split_counts(merged.samples)
        return MergeResult(request.output.expanduser().resolve(), request.output.expanduser().resolve() / manifest.name, merged.task.value, len(merged.samples), classes, counts)
