from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Iterable


class TaskKind(StrEnum):
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object-detection"


class DatasetFormat(StrEnum):
    IMAGE_FOLDER = "image-folder"
    COCO = "coco"
    YOLO = "yolo"
    RFDETR = "rfdetr"
    NEUROCLE = "neurocle"


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MaterializationMode(StrEnum):
    COPY = "copy"
    HARDLINK = "hardlink"


class BalanceMode(StrEnum):
    NONE = "none"
    UNDERSAMPLE = "undersample"
    OVERSAMPLE = "oversample"


@dataclass(frozen=True)
class SplitPolicy:
    train: float = 0.7
    val: float = 0.2
    test: float = 0.1
    seed: int = 42
    grouping: str = "sample"
    stratify: bool = True
    preserve_existing: bool = False
    train_groups: frozenset[int] | None = None

    def ratios(self) -> tuple[float, float, float]:
        values = (self.train, self.val, self.test)
        if any(value < 0 for value in values) or abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("split ratios must be non-negative and add to 1")
        return values


@dataclass(frozen=True)
class ImageRef:
    path: Path
    width: int
    height: int
    sha256: str | None = None
    source_id: str | None = None


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def clipped(self, width: int, height: int) -> "BoundingBox":
        return BoundingBox(
            max(0.0, min(float(width), self.x1)),
            max(0.0, min(float(height), self.y1)),
            max(0.0, min(float(width), self.x2)),
            max(0.0, min(float(height), self.y2)),
        )


@dataclass(frozen=True)
class DetectionAnnotation:
    class_id: int
    box: BoundingBox
    iscrowd: int = 0


@dataclass(frozen=True)
class ClassificationSample:
    image: ImageRef
    class_id: int
    class_name: str
    split: Split | None = None
    group_key: tuple[str, ...] = ()


@dataclass(frozen=True)
class DetectionSample:
    image: ImageRef
    annotations: tuple[DetectionAnnotation, ...]
    split: Split | None = None
    group_key: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClassificationDataset:
    classes: tuple[str, ...]
    samples: tuple[ClassificationSample, ...]
    source: Path | None = None
    source_format: DatasetFormat = DatasetFormat.IMAGE_FOLDER
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def task(self) -> TaskKind:
        return TaskKind.CLASSIFICATION


@dataclass(frozen=True)
class DetectionDataset:
    classes: tuple[str, ...]
    samples: tuple[DetectionSample, ...]
    source: Path | None = None
    source_format: DatasetFormat = DatasetFormat.COCO
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def task(self) -> TaskKind:
        return TaskKind.OBJECT_DETECTION


@dataclass(frozen=True)
class DatasetValidationIssue:
    severity: str
    code: str
    message: str
    path: Path | None = None


@dataclass(frozen=True)
class DatasetValidationReport:
    valid: bool
    task: TaskKind | None
    format: DatasetFormat | None
    issues: tuple[DatasetValidationIssue, ...]
    sample_count: int = 0
    class_count: int = 0

    @property
    def errors(self) -> tuple[DatasetValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")


@dataclass(frozen=True)
class DatasetInspection:
    path: Path
    format: DatasetFormat
    task: TaskKind
    classes: tuple[str, ...]
    split_counts: dict[str, int]
    sample_count: int
    annotation_count: int = 0


def split_value(value: str | Split | None) -> Split | None:
    if value is None:
        return None
    return value if isinstance(value, Split) else Split(value.casefold())


def samples_for_split(samples: Iterable[ClassificationSample | DetectionSample], split: Split) -> tuple:
    return tuple(sample for sample in samples if sample.split == split)
