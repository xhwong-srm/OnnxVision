"""Merge classification datasets with optional grouped re-splitting and train balancing."""

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
LEGACY_BUILDER_NAME = re.compile(r"^.+_(?P<image_id>[0-9a-f]{10})_\d+_\d+$", re.IGNORECASE)
LEGACY_OVERSAMPLE_SUFFIX = re.compile(r"_n\d+$", re.IGNORECASE)


def image_files(folder: Path) -> list[Path]:
    return sorted(
        (path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.as_posix().lower(),
    )


def builder_identity(entry: tuple[Path, Path, str]) -> tuple[str | None, str | None]:
    """Return image and duplicate-group IDs encoded by the builder filename."""
    path, _, _ = entry
    match = BUILDER_NAME.fullmatch(path.stem)
    if match:
        return match["image_id"].lower(), match["group"]
    legacy_stem = LEGACY_OVERSAMPLE_SUFFIX.sub("", path.stem)
    legacy = LEGACY_BUILDER_NAME.fullmatch(legacy_stem)
    return (legacy["image_id"].lower(), None) if legacy else (None, None)


def parse_group_selection(value: str) -> set[int]:
    """Parse zero-based group indices such as ``3`` or ``1-3,7``."""
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise argparse.ArgumentTypeError("group selection contains an empty item")
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            if not start_text.isdigit() or not end_text.isdigit():
                raise argparse.ArgumentTypeError(f"invalid group range: {part!r}")
            start, end = int(start_text), int(end_text)
            if start > end:
                raise argparse.ArgumentTypeError(f"group range must be ascending: {part!r}")
            selected.update(range(start, end + 1))
        elif part.isdigit():
            selected.add(int(part))
        else:
            raise argparse.ArgumentTypeError(f"invalid group index: {part!r}")
    if not selected:
        raise argparse.ArgumentTypeError("select at least one group")
    return selected


def is_oversample_copy(entry: tuple[Path, Path, str]) -> bool:
    """Return whether a file is an encoded builder or merge oversample copy."""
    match = BUILDER_NAME.fullmatch(entry[0].stem)
    if match:
        return int(match["occurrence"]) > 0
    return bool(LEGACY_OVERSAMPLE_SUFFIX.search(entry[0].stem))


def oversample_name(path: Path, occurrence: int) -> str:
    """Use the builder's nNNN occurrence convention for a repeated sample."""
    match = BUILDER_NAME.fullmatch(path.stem)
    if match:
        return (
            f"{match['image_id']}_r{int(match['roi']):04d}"
            f"_g{int(match['group']):04d}_n{occurrence:03d}{path.suffix}"
        )
    base = LEGACY_OVERSAMPLE_SUFFIX.sub("", path.stem)
    return f"{base}_n{occurrence:03d}{path.suffix}"


def grouping_key(entry: tuple[Path, Path, str], mode: str) -> tuple[str, ...]:
    path, source_root, _ = entry
    image_id, duplicate_group = builder_identity(entry)
    if mode == "image" and image_id is not None:
        return "image", image_id
    if mode == "duplicate" and duplicate_group is not None:
        return "duplicate", image_id, duplicate_group
    return "sample", str(source_root).lower(), str(path).lower()


def grouped_entries(entries: list[tuple[Path, Path, str]], mode: str) -> list[list[tuple[Path, Path, str]]]:
    grouped: dict[tuple[str, ...], list[tuple[Path, Path, str]]] = {}
    for entry in entries:
        grouped.setdefault(grouping_key(entry, mode), []).append(entry)
    return list(grouped.values())


def split_counts(count: int, ratios: tuple[float, ...]) -> list[int]:
    exact = [count * ratio for ratio in ratios]
    result = [int(value) for value in exact]
    for index in sorted(range(len(ratios)), key=lambda i: exact[i] - result[i], reverse=True)[:count - sum(result)]:
        result[index] += 1
    return result


def split_entries(entries, ratios, rng, mode, split_names=SPLITS):
    splits = {name: [] for name in split_names}
    if mode != "sample":
        class_names = sorted({entry[2] for entry in entries})
        class_totals = {
            class_name: sum(entry[2] == class_name for entry in entries)
            for class_name in class_names
        }
        targets = {
            class_name: split_counts(class_totals[class_name], ratios)
            for class_name in class_names
        }
        counts = {
            class_name: [0] * len(split_names)
            for class_name in class_names
        }
        groups = grouped_entries(entries, mode)
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        for group in groups:
            group_counts = {
                class_name: sum(entry[2] == class_name for entry in group)
                for class_name in class_names
            }

            def assignment_cost(index: int) -> float:
                cost = 0.0
                for class_name in class_names:
                    target = targets[class_name][index]
                    updated = counts[class_name][index] + group_counts[class_name]
                    cost += ((updated - target) ** 2 - (counts[class_name][index] - target) ** 2) / max(target, 1)
                return cost

            split_index = min(range(len(split_names)), key=lambda index: (assignment_cost(index), index))
            splits[split_names[split_index]].extend(group)
            for class_name in class_names:
                counts[class_name][split_index] += group_counts[class_name]
        return splits

    by_class = {}
    for entry in entries:
        by_class.setdefault(entry[2], []).append(entry)
    for class_entries in by_class.values():
        rng.shuffle(class_entries)
        counts = split_counts(len(class_entries), ratios)
        offset = 0
        for split, count in zip(split_names, counts):
            splits[split].extend(class_entries[offset:offset + count])
            offset += count
    return splits


def split_entries_with_train_groups(entries, ratios, rng, train_groups):
    """Put selected builder groups in train and split all other groups between val/test."""
    train_entries = []
    evaluation_entries = []
    unsupported = []
    for entry in entries:
        image_id, duplicate_group = builder_identity(entry)
        if image_id is None or duplicate_group is None:
            unsupported.append(entry[0])
        elif int(duplicate_group) in train_groups:
            train_entries.append(entry)
        else:
            evaluation_entries.append(entry)
    if unsupported:
        examples = ", ".join(str(path) for path in unsupported[:3])
        suffix = "" if len(unsupported) <= 3 else f" (and {len(unsupported) - 3} more)"
        raise SystemExit(
            "--train-groups requires builder filenames containing image and group IDs; "
            f"unsupported: {examples}{suffix}"
        )

    val_test_total = ratios[1] + ratios[2]
    if evaluation_entries and val_test_total <= 0:
        raise SystemExit("--train-groups requires a non-zero --val or --test percentage.")
    evaluation_ratios = (
        ratios[1] / val_test_total if val_test_total else 0.0,
        ratios[2] / val_test_total if val_test_total else 0.0,
    )
    evaluation = split_entries(
        evaluation_entries, evaluation_ratios, rng, "duplicate", ("val", "test")
    )
    return {"train": train_entries, **evaluation}


def grouped_sample(entries, target: int, rng: random.Random, mode: str):
    """Choose complete groups with a total count closest to target."""
    groups = grouped_entries(entries, mode)
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


def grouped_oversample(entries, target: int, rng: random.Random, mode: str):
    """Repeat complete groups until reaching the count nearest to target."""
    groups = grouped_entries(entries, mode)
    selected = list(entries)
    while groups and len(selected) < target:
        remaining = target - len(selected)
        group = min(groups, key=lambda value: (abs(len(value) - remaining), rng.random()))
        selected.extend(group)
    return sorted(selected, key=lambda item: item[0].as_posix().lower())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="+", type=Path, help="Two or more source dataset directories.")
    parser.add_argument("output", type=Path, help="Destination dataset directory.")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resplit", action="store_true", help="Ignore existing split assignments and create new splits. Automatically enabled for class-only sources.")
    parser.add_argument("--train", type=float, default=70, help="Training percentage used when re-splitting.")
    parser.add_argument("--val", type=float, default=20, help="Validation percentage used when re-splitting.")
    parser.add_argument("--test", type=float, default=10, help="Test percentage used when re-splitting.")
    parser.add_argument("--group-duplicates", action="store_true", help="Keep builder duplicate groups in one split.")
    parser.add_argument("--group-by-image", action="store_true", help="Keep all builder ROIs from each source image in one split.")
    parser.add_argument(
        "--train-groups",
        type=parse_group_selection,
        metavar="GROUPS",
        help=(
            "Allow only these zero-based builder group indices in train; put all other groups "
            "in val/test. Accepts specific indices and inclusive ranges, for example 3 or 1-3,7."
        ),
    )
    parser.add_argument("--balance", choices=("undersample", "oversample", "none"), default="undersample")
    args = parser.parse_args()

    if len(args.source) < 2:
        parser.error("at least two source datasets are required")

    sources = [path.resolve() for path in args.source]
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists; use --overwrite to replace it: {output}")
        shutil.rmtree(output)

    unsplit_sources = [
        source for source in sources
        if not any((source / split).is_dir() for split in SPLITS)
    ]
    effective_resplit = args.resplit or bool(unsplit_sources) or args.train_groups is not None
    ratios = (args.train / 100, args.val / 100, args.test / 100)
    if effective_resplit and (any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1) > 1e-9):
        raise SystemExit("--train, --val, and --test must be non-negative and add to 100.")
    mode = "image" if args.group_by_image else "duplicate" if args.group_duplicates else "sample"
    effective_balance = (
        "none"
        if args.train_groups is not None and args.balance == "undersample"
        else args.balance
    )

    collected: dict[tuple[str, str], list[tuple[Path, Path, str]]] = {}
    for split in SPLITS:
        class_names = sorted(
            {
                class_dir.name
                for source in sources
                if source not in unsplit_sources and (source / split).is_dir()
                for class_dir in (source / split).iterdir()
                if class_dir.is_dir()
            }
        )
        for class_name in class_names:
            entries: list[tuple[Path, Path]] = []
            for source in sources:
                source_class = source / split / class_name
                if source_class.is_dir():
                    entries.extend((path, source, class_name) for path in image_files(source_class))
            collected[(split, class_name)] = entries

    for source in unsplit_sources:
        for class_dir in sorted((path for path in source.iterdir() if path.is_dir()), key=lambda path: path.name.lower()):
            key = "train", class_dir.name
            collected.setdefault(key, []).extend(
                (path, source, class_dir.name) for path in image_files(class_dir)
            )

    rng = random.Random(args.seed)
    if effective_resplit:
        all_entries = [entry for entries in collected.values() for entry in entries]
        before_filter = len(all_entries)
        all_entries = [entry for entry in all_entries if not is_oversample_copy(entry)]
        removed_oversamples = before_filter - len(all_entries)
        split_result = (
            split_entries_with_train_groups(all_entries, ratios, rng, args.train_groups)
            if args.train_groups is not None
            else split_entries(all_entries, ratios, rng, mode)
        )
        selected = {
            (split, class_name): [entry for entry in split_result[split] if entry[2] == class_name]
            for split in SPLITS
            for class_name in sorted({entry[2] for entry in all_entries})
        }
    else:
        selected = dict(collected)
        removed_oversamples = 0

    if not any(selected.values()):
        raise SystemExit("No images found in the source datasets.")

    train_classes = sorted({class_name for split, class_name in selected if split == "train"})
    train_counts = [len(selected[("train", class_name)]) for class_name in train_classes]
    train_target = None
    if effective_balance != "none" and train_counts:
        train_target = min(train_counts) if effective_balance == "undersample" else max(train_counts)
        for class_name in train_classes:
            key = "train", class_name
            if effective_balance == "undersample":
                selected[key] = grouped_sample(selected[key], train_target, rng, mode)
            else:
                selected[key] = grouped_oversample(selected[key], train_target, rng, mode)

    manifest_path = output / "merge_manifest.csv"
    output.mkdir(parents=True)
    manifest_rows: list[dict[str, str]] = []
    used_destinations: dict[Path, Path] = {}
    for (split, class_name), entries in selected.items():
        destination_dir = output / split / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        for source_path, source_root, _ in entries:
            destination = destination_dir / source_path.name
            if destination in used_destinations and used_destinations[destination] != source_path.resolve():
                raise RuntimeError(f"Destination filename collision: {destination}")
            match = BUILDER_NAME.fullmatch(source_path.stem)
            occurrence = int(match["occurrence"]) + 1 if match else 1
            while destination in used_destinations or destination.exists():
                destination = destination_dir / oversample_name(source_path, occurrence)
                occurrence += 1
            if destination.exists():
                raise RuntimeError(f"Destination filename collision: {destination}")
            used_destinations[destination] = source_path.resolve()
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
    split_mode = "resplit (automatic for class-only source)" if unsplit_sources and not args.resplit else "resplit" if effective_resplit else "preserve"
    print(f"Split mode: {split_mode}; grouping: {mode}; balance: {effective_balance}")
    if args.train_groups is not None:
        groups = ",".join(str(index) for index in sorted(args.train_groups))
        print(f"Train groups (zero-based): {groups}; all other groups restricted to val/test")
        if args.balance == "undersample":
            print("Train undersampling disabled so every selected group remains represented.")
    if effective_resplit:
        print(f"Prior oversample copies removed before split: {removed_oversamples}")
    if train_target is not None:
        print(f"Train target per class: {train_target} (nearest complete groups)")
    for key in sorted(selected):
        print(f"{key[0]}/{key[1]}: {len(selected[key])}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
