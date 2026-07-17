"""Create a YOLO classification dataset from class-named folders.

Expected input:

    source/
        class_a/image1.jpg
        class_a/image2.png
        class_b/image3.jpg

Generated output:

    output/
        train/class_a/...
        val/class_a/...
        test/class_a/...

Images below each class directory are discovered recursively. Their relative
paths are retained, which avoids filename collisions in nested source folders.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".bmp",
    ".dng",
    ".gif",
    ".heic",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert class-named image folders to a YOLO classification dataset."
    )
    parser.add_argument("source", type=Path, help="Folder containing one subfolder per class")
    parser.add_argument("output", type=Path, help="Destination dataset folder")
    parser.add_argument("--train", type=float, default=0.70, help="Training fraction (default: 0.70)")
    parser.add_argument("--val", type=float, default=0.20, help="Validation fraction (default: 0.20)")
    parser.add_argument("--test", type=float, default=0.10, help="Test fraction (default: 0.10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move images instead of copying them (destructive to the source)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before creating the dataset",
    )
    return parser.parse_args()


def validate_paths(source: Path, output: Path, overwrite: bool) -> tuple[Path, Path]:
    source = source.resolve()
    output = output.resolve()

    if not source.is_dir():
        raise ValueError(f"Source directory does not exist: {source}")
    if source == output:
        raise ValueError("Source and output directories must be different")
    if source in output.parents:
        raise ValueError("Output must not be inside the source directory")

    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output}. Use --overwrite to replace it."
            )
        shutil.rmtree(output)

    return source, output


def split_counts(image_count: int, ratios: tuple[float, float, float]) -> list[int]:
    """Allocate every image using the largest-remainder method."""
    exact = [image_count * ratio for ratio in ratios]
    counts = [int(value) for value in exact]
    remaining = image_count - sum(counts)
    remainder_order = sorted(
        range(len(ratios)), key=lambda index: exact[index] - counts[index], reverse=True
    )
    for index in remainder_order[:remaining]:
        counts[index] += 1
    return counts


def find_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_dataset(args: argparse.Namespace) -> None:
    ratios = (args.train, args.val, args.test)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split fractions cannot be negative")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("--train, --val, and --test must add up to 1.0")

    source, output = validate_paths(args.source, args.output, args.overwrite)
    class_dirs = sorted(path for path in source.iterdir() if path.is_dir() and not path.name.startswith("."))
    if not class_dirs:
        raise ValueError(f"No class subfolders found in: {source}")

    rng = random.Random(args.seed)
    operation = shutil.move if args.move else shutil.copy2
    totals = dict.fromkeys(SPLIT_NAMES, 0)
    used_classes = 0

    for class_dir in class_dirs:
        images = find_images(class_dir)
        if not images:
            print(f"Skipping empty class: {class_dir.name}")
            continue

        used_classes += 1
        rng.shuffle(images)
        counts = split_counts(len(images), ratios)
        offset = 0

        for split_name, count in zip(SPLIT_NAMES, counts):
            for image in images[offset : offset + count]:
                destination = output / split_name / class_dir.name / image.relative_to(class_dir)
                destination.parent.mkdir(parents=True, exist_ok=True)
                operation(image, destination)
            totals[split_name] += count
            offset += count

        print(
            f"{class_dir.name}: {len(images)} images "
            f"(train={counts[0]}, val={counts[1]}, test={counts[2]})"
        )

    if used_classes == 0:
        raise ValueError(f"No supported image files found below: {source}")

    print(f"\nCreated dataset at: {output}")
    print(", ".join(f"{name}={totals[name]}" for name in SPLIT_NAMES))


def main() -> None:
    try:
        build_dataset(parse_args())
    except (FileExistsError, OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
