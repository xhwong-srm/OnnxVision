"""Export a trained Ultralytics YOLO classification model for deployment."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="Path to the trained .pt checkpoint, typically weights/best.pt",
    )
    parser.add_argument(
        "--format",
        choices=("onnx", "openvino", "both"),
        default="both",
        help="Export format (default: both)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Square input image size used during export (default: 224)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="Optional ONNX opset version; uses the Ultralytics default when omitted",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Allow dynamic input batch/image dimensions",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export FP16 where supported",
    )
    parser.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Simplify the ONNX graph (default: enabled)",
    )
    return parser.parse_args()


def export_model(args: argparse.Namespace) -> list[Path]:
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if model_path.suffix.lower() != ".pt":
        raise ValueError(f"Expected an Ultralytics .pt checkpoint: {model_path}")

    model = YOLO(str(model_path), task="classify")
    formats = ("onnx", "openvino") if args.format == "both" else (args.format,)
    exported: list[Path] = []

    for export_format in formats:
        options: dict[str, object] = {
            "format": export_format,
            "imgsz": args.imgsz,
            "dynamic": args.dynamic,
            "half": args.half,
        }
        if export_format == "onnx":
            options["simplify"] = args.simplify
            if args.opset is not None:
                options["opset"] = args.opset

        print(f"Exporting {model_path.name} to {export_format}...")
        output = Path(model.export(**options)).resolve()
        exported.append(output)
        print(f"Created: {output}")

    return exported


def main() -> None:
    export_model(parse_args())


if __name__ == "__main__":
    main()
