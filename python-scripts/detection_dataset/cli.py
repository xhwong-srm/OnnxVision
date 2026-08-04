"""Command-line frontend for the detection dataset converter.

Examples::

    uv run python python-scripts/convert_detection_dataset.py \
        --input-format coco --output-format yolo \
        --data images/seal_dataset --output artifacts/seal-yolo

    uv run python python-scripts/convert_detection_dataset.py \
        --output-format yolo \
        --data images/seal_dataset --output artifacts/seal-yolo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .convert import INPUT_FORMATS, OUTPUT_FORMATS, convert_dataset


def build_parser(
    default_input_format: str | None = None, default_output_format: str | None = None
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-format",
        choices=INPUT_FORMATS,
        default=default_input_format,
        help="Input dataset layout; omitted to detect it automatically",
    )
    parser.add_argument(
        "--output-format",
        choices=OUTPUT_FORMATS,
        default=default_output_format,
        required=default_output_format is None,
        help="Output dataset layout",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Input dataset root, or YOLO data.yaml when --input-format yolo",
    )
    parser.add_argument("--output", required=True, type=Path, help="New or empty output dataset directory")
    parser.add_argument(
        "--image-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Hard links avoid duplicating images on the same volume; use copy across volumes.",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    default_input_format: str | None = None,
    default_output_format: str | None = None,
) -> None:
    args = build_parser(default_input_format, default_output_format).parse_args(argv)
    result = convert_dataset(
        args.data,
        args.output,
        args.output_format,
        input_format=args.input_format,
        image_mode=args.image_mode,
    )
    print(
        f"Converted {result.input_format} dataset to {result.output_format}; "
        f"classes={list(result.classes)}; split images={result.split_image_counts}; "
        f"output={result.output}"
    )


if __name__ == "__main__":
    main()
