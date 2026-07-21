"""Randomly sample images from one folder and copy them with numeric names."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def largest_existing_number(destination: Path) -> int:
    """Return the largest numeric image stem in destination, or zero."""
    return max(
        (
            int(path.stem)
            for path in destination.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and path.stem.isdigit()
        ),
        default=0,
    )


def sample_and_copy(
    source: Path,
    destination: Path,
    amount: int,
    seed: int | None = None,
) -> list[Path]:
    if amount < 1:
        raise ValueError("amount must be at least 1")
    if not source.is_dir():
        raise ValueError(f"source folder does not exist: {source}")

    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    images = [
        path
        for path in source.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.parent.resolve() != destination_resolved
    ]

    if amount > len(images):
        raise ValueError(
            f"requested {amount} images, but only {len(images)} were found"
        )

    selected = random.Random(seed).sample(images, amount)
    next_number = largest_existing_number(destination) + 1
    copied: list[Path] = []

    for number, source_path in enumerate(selected, start=next_number):
        target = destination / f"{number}{source_path.suffix.lower()}"
        shutil.copy2(source_path, target)
        copied.append(target)

    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly sample images and copy them using numeric filenames."
    )
    parser.add_argument("source", type=Path, help="folder containing source images")
    parser.add_argument("destination", type=Path, help="folder to copy images into")
    parser.add_argument("amount", type=int, help="number of images to sample")
    parser.add_argument(
        "--seed",
        type=int,
        help="optional random seed for a repeatable selection",
    )
    args = parser.parse_args()

    try:
        copied = sample_and_copy(
            args.source, args.destination, args.amount, args.seed
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Copied {len(copied)} images to {args.destination.resolve()}")
    print(f"Created files {copied[0].name} through {copied[-1].name}")


if __name__ == "__main__":
    main()
