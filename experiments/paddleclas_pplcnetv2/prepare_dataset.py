"""Create PaddleClas classification manifests from class-folder splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
SPLITS = ("train", "val", "test")


def build_manifests(source: Path, manifest_dir: Path) -> dict[str, object]:
    source = source.resolve()
    manifest_dir = manifest_dir.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {source}")

    class_names = sorted(
        child.name
        for child in (source / "train").iterdir()
        if child.is_dir()
    )
    if not class_names:
        raise ValueError(f"no class directories found under {source / 'train'}")

    class_ids = {name: index for index, name in enumerate(class_names)}
    split_counts: dict[str, dict[str, int]] = {}
    split_entries: dict[str, list[str]] = {}
    for split in SPLITS:
        split_root = source / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"missing split directory: {split_root}")
        entries: list[str] = []
        counts: dict[str, int] = {}
        actual_classes = sorted(child.name for child in split_root.iterdir() if child.is_dir())
        if actual_classes != class_names:
            raise ValueError(
                f"class mismatch in {split}: expected {class_names}, found {actual_classes}"
            )
        for class_name in class_names:
            class_root = split_root / class_name
            paths = sorted(
                path for path in class_root.rglob("*")
                if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
            )
            if not paths:
                raise ValueError(f"no images found under {class_root}")
            counts[class_name] = len(paths)
            for path in paths:
                relative_path = path.relative_to(source).as_posix()
                entries.append(f"{relative_path} {class_ids[class_name]}")
        if len(entries) != len(set(entries)):
            raise ValueError(f"duplicate manifest entries found in {split}")
        split_counts[split] = counts
        split_entries[split] = entries

    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split, entries in split_entries.items():
        (manifest_dir / f"{split}_list.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")
    (manifest_dir / "class_id_map.txt").write_text(
        "\n".join(f"{index} {name}" for name, index in class_ids.items()) + "\n",
        encoding="utf-8",
    )
    summary = {
        "source": str(source),
        "classes": class_ids,
        "counts": split_counts,
        "total": {split: sum(counts.values()) for split, counts in split_counts.items()},
    }
    (manifest_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="class-folder dataset root")
    parser.add_argument("--manifest-dir", type=Path, required=True, help="output directory")
    args = parser.parse_args()
    summary = build_manifests(args.source, args.manifest_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
