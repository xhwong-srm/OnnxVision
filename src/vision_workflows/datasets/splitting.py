from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import replace
from typing import Iterable, TypeVar

from ..domain.datasets import (
    ClassificationSample,
    DetectionSample,
    Split,
    SplitPolicy,
)

Sample = TypeVar("Sample", ClassificationSample, DetectionSample)


def _class_counts(sample: ClassificationSample | DetectionSample) -> dict[int, int]:
    if isinstance(sample, ClassificationSample):
        return {sample.class_id: 1}
    counts: dict[int, int] = defaultdict(int)
    for annotation in sample.annotations:
        counts[annotation.class_id] += 1
    return counts


def _group_key(sample: Sample, index: int, grouping: str) -> tuple[str, ...]:
    if grouping == "sample":
        return ("sample", str(index))
    if sample.group_key:
        return sample.group_key
    return ("sample", str(index))


def split_samples(samples: Iterable[Sample], class_count: int, policy: SplitPolicy) -> tuple[Sample, ...]:
    values = list(samples)
    ratios = policy.ratios()
    if not values:
        return ()
    groups: dict[tuple[str, ...], list[tuple[int, Sample]]] = defaultdict(list)
    for index, sample in enumerate(values):
        groups[_group_key(sample, index, policy.grouping)].append((index, sample))
    grouped = list(groups.values())
    rng = random.Random(policy.seed)
    rng.shuffle(grouped)
    if policy.grouping != "sample":
        grouped.sort(key=len, reverse=True)

    split_names = (Split.TRAIN, Split.VAL, Split.TEST)
    targets = [[0] * len(split_names) for _ in range(class_count)]
    total_by_class = [0] * class_count
    for _, sample in values_with_indices(values):
        for class_id, count in _class_counts(sample).items():
            if class_id < class_count:
                total_by_class[class_id] += count
    for class_id, total in enumerate(total_by_class):
        exact = [total * ratio for ratio in ratios]
        counts = [int(value) for value in exact]
        for split_index in sorted(range(3), key=lambda index: exact[index] - counts[index], reverse=True)[: total - sum(counts)]:
            counts[split_index] += 1
        targets[class_id] = counts

    assigned: dict[tuple[str, ...], Split] = {}
    current = [[0] * len(split_names) for _ in range(class_count)]
    for group in grouped:
        if policy.train_groups is not None:
            duplicate_id = group[0][1].group_key[-1] if group[0][1].group_key else ""
            try:
                forced_train = int(duplicate_id) in policy.train_groups
            except ValueError:
                forced_train = False
            chosen = 0 if forced_train else None
        else:
            chosen = None
        if chosen is None:
            group_counts: dict[int, int] = defaultdict(int)
            for _, sample in group:
                for class_id, count in _class_counts(sample).items():
                    group_counts[class_id] += count

            def cost(split_index: int) -> float:
                total = 0.0
                for class_id, count in group_counts.items():
                    if class_id >= class_count:
                        continue
                    before = current[class_id][split_index] - targets[class_id][split_index]
                    after = before + count
                    total += (after * after - before * before) / max(targets[class_id][split_index], 1)
                return total

            chosen = min(range(3), key=lambda index: (cost(index), index))
        assigned[_group_key(group[0][1], group[0][0], policy.grouping)] = split_names[chosen]
        for _, sample in group:
            for class_id, count in _class_counts(sample).items():
                if class_id < class_count:
                    current[class_id][chosen] += count

    result = []
    for index, sample in enumerate(values):
        key = _group_key(sample, index, policy.grouping)
        result.append(replace(sample, split=assigned[key]))
    return tuple(result)


def values_with_indices(values: list[Sample]):
    return enumerate(values)
