"""Command-line frontend for merging image-classification datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from .merge import merge_datasets, parse_group_selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="+", type=Path, help="Two or more source dataset directories.")
    parser.add_argument("output", type=Path, help="Destination dataset directory.")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resplit",
        action="store_true",
        help="Ignore existing split assignments and create new splits. Automatically enabled for class-only sources.",
    )
    parser.add_argument("--train", type=float, default=70, help="Training percentage used when re-splitting.")
    parser.add_argument("--val", type=float, default=20, help="Validation percentage used when re-splitting.")
    parser.add_argument("--test", type=float, default=10, help="Test percentage used when re-splitting.")
    parser.add_argument(
        "--group-duplicates",
        action="store_true",
        help="Keep builder duplicate groups in one split.",
    )
    parser.add_argument(
        "--group-by-image",
        action="store_true",
        help="Keep all builder ROIs from each source image in one split.",
    )
    parser.add_argument(
        "--train-groups",
        type=parse_group_selection,
        metavar="GROUPS",
        help=(
            "Allow only these zero-based builder group indices in train; put all other groups "
            "in val/test. Accepts specific indices and inclusive ranges, for example 3 or 1-3,7."
        ),
    )
    parser.add_argument(
        "--balance",
        choices=("undersample", "oversample", "none"),
        default="undersample",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.source) < 2:
        parser.error("at least two source datasets are required")

    result = merge_datasets(
        args.source,
        args.output,
        seed=args.seed,
        overwrite=args.overwrite,
        resplit=args.resplit,
        train=args.train,
        val=args.val,
        test=args.test,
        group_duplicates=args.group_duplicates,
        group_by_image=args.group_by_image,
        train_groups=args.train_groups,
        balance=args.balance,
    )

    print(f"Created: {result.output}")
    print(f"Split mode: {result.split_mode}; grouping: {result.grouping}; balance: {result.balance}")
    if result.train_groups is not None:
        groups = ",".join(str(index) for index in result.train_groups)
        print(f"Train groups (zero-based): {groups}; all other groups restricted to val/test")
        if result.requested_balance == "undersample":
            print("Train undersampling disabled so every selected group remains represented.")
    if result.split_mode != "preserve":
        print(f"Prior oversample copies removed before split: {result.removed_oversamples}")
    if result.train_target is not None:
        print(f"Train target per class: {result.train_target} (nearest complete groups)")
    for key in sorted(result.split_counts):
        print(f"{key[0]}/{key[1]}: {result.split_counts[key]}")
    print(f"Manifest: {result.manifest}")


if __name__ == "__main__":
    main()
