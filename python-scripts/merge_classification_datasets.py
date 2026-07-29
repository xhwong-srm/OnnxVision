"""Merge classification datasets with deterministic, duplicate-group-aware train undersampling."""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
BUILDER_NAME = re.compile(
    r"^(?P<image_id>[0-9a-f]{12})_r(?P<roi>\d+)_g(?P<group>\d+)_n(?P<occurrence>\d+)$",
    re.IGNORECASE,
)


def image_files(folder: Path) -> list[Path]:
    return sorted(
        (path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.as_posix().lower(),
    )


def duplicate_group_key(entry: tuple[Path, Path]) -> tuple[str, ...]:
    """Identify a builder duplicate group; legacy filenames remain independent."""
    path, source_root = entry
    match = BUILDER_NAME.fullmatch(path.stem)
    if match:
        return match["image_id"].lower(), match["group"]
    return "legacy", str(source_root).lower(), str(path).lower()


def grouped_sample(entries: list[tuple[Path, Path]], target: int, rng: random.Random) -> list[tuple[Path, Path]]:
    """Choose complete duplicate groups with a total count closest to target."""
    grouped: dict[tuple[str, ...], list[tuple[Path, Path]]] = {}
    for entry in entries:
        grouped.setdefault(duplicate_group_key(entry), []).append(entry)
    groups = list(grouped.values())
    rng.shuffle(groups)
    if not groups or target <= 0:
        return []

    limit = min(sum(map(len, groups)), target + max(map(len, groups)))
    parents: dict[int, tuple[int, int] | None] = {0: None}
    for index, group in enumerate(groups):
        size = len(group)
        for current in sorted(tuple(parents), reverse=True):
            total = current + size
            if total <= limit and total not in parents:
                parents[total] = current, index
    chosen_total = min(parents, key=lambda total: (abs(total - target), total > target, -total))
    chosen_groups = []
    while chosen_total:
        previous, index = parents[chosen_total]
        chosen_groups.append(index)
        chosen_total = previous
    return sorted(
        (entry for index in chosen_groups for entry in groups[index]),
        key=lambda item: item[0].as_posix().lower(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs=2, type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sources = [path.resolve() for path in args.source]
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists; use --overwrite to replace it: {output}")
        shutil.rmtree(output)

    collected: dict[tuple[str, str], list[tuple[Path, Path]]] = {}
    for split in SPLITS:
        class_names = sorted(
            {
                class_dir.name
                for source in sources
                for class_dir in (source / split).iterdir()
                if class_dir.is_dir()
            }
        )
        for class_name in class_names:
            entries: list[tuple[Path, Path]] = []
            for source in sources:
                source_class = source / split / class_name
                if source_class.is_dir():
                    entries.extend((path, source) for path in image_files(source_class))
            collected[(split, class_name)] = entries

    rng = random.Random(args.seed)
    selected: dict[tuple[str, str], list[tuple[Path, Path]]] = {}
    train_classes = sorted({class_name for split, class_name in collected if split == "train"})
    train_target = min(len(collected[("train", class_name)]) for class_name in train_classes)
    for key, entries in collected.items():
        if key[0] == "train":
            selected[key] = grouped_sample(entries, train_target, rng)
        else:
            selected[key] = entries

    manifest_path = output / "merge_manifest.csv"
    output.mkdir(parents=True)
    manifest_rows: list[dict[str, str]] = []
    used_destinations: set[Path] = set()
    for (split, class_name), entries in selected.items():
        destination_dir = output / split / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        for source_path, source_root in entries:
            destination = destination_dir / source_path.name
            if destination in used_destinations or destination.exists():
                raise RuntimeError(f"Destination filename collision: {destination}")
            used_destinations.add(destination)
            shutil.copy2(source_path, destination)
            manifest_rows.append(
                {
                    "split": split,
                    "class": class_name,
                    "output": destination.relative_to(output).as_posix(),
                    "source_dataset": source_root.name,
                    "source": source_path.relative_to(source_root).as_posix(),
                }
            )

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(manifest_rows, key=lambda row: row["output"].lower()))

    print(f"Created: {output}")
    print(f"Train target per class: {train_target} (nearest complete duplicate groups)")
    for key in sorted(selected):
        print(f"{key[0]}/{key[1]}: {len(selected[key])}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
