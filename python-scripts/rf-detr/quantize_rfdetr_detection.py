"""Post-training quantize an exported RF-DETR C24 ONNX detector.

Static quantization calibrates with raw BGR uint8 images and emits the QDQ
format preferred by ONNX Runtime CPU. Dynamic quantization is also available
for weight-heavy MatMul/Gemm experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


class RawC24CalibrationReader(CalibrationDataReader):
    def __init__(self, input_name: str, image_paths: list[Path]) -> None:
        self.input_name = input_name
        self.image_paths = iter(image_paths)

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            path = next(self.image_paths)
        except StopIteration:
            return None
        with Image.open(path) as image:
            bgr = np.asarray(image.convert("RGB"), dtype=np.uint8)[..., ::-1].copy()
        return {self.input_name: bgr[None]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("static", "dynamic"), default="static")
    parser.add_argument("--calibration-images", type=Path)
    parser.add_argument("--calibration-count", type=int, default=64)
    parser.add_argument(
        "--operators",
        nargs="+",
        default=["Conv", "Gemm"],
        help="ONNX operator types to quantize (default excludes runtime attention MatMul nodes)",
    )
    parser.add_argument("--per-channel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reduce-range", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def image_paths(directory: Path, count: int) -> list[Path]:
    paths = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No calibration images found under {directory}")
    return paths[:count]


def validate_model(path: Path) -> None:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    if metadata.get("detection_contract") != "onnx-vision-detection-v1":
        raise ValueError("Quantized model lost the OnnxVision detection contract metadata")
    print(f"validated_onnx={path}")
    print(f"model_bytes={path.stat().st_size}")


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "dynamic":
        quantize_dynamic(
            model_input=source,
            model_output=output,
            op_types_to_quantize=args.operators,
            per_channel=args.per_channel,
            reduce_range=args.reduce_range,
            weight_type=QuantType.QInt8,
        )
    else:
        if args.calibration_images is None:
            raise ValueError("--calibration-images is required for static quantization")
        session = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
        input_metadata = session.get_inputs()[0]
        if input_metadata.type != "tensor(uint8)" or input_metadata.shape[-1] != 3:
            raise ValueError("Static calibration currently requires the raw C24 NHWC BGR model")
        calibration = image_paths(
            args.calibration_images.expanduser().resolve(), args.calibration_count
        )
        print(f"calibration_images={len(calibration)}")
        quantize_static(
            model_input=source,
            model_output=output,
            calibration_data_reader=RawC24CalibrationReader(input_metadata.name, calibration),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=args.per_channel,
            reduce_range=args.reduce_range,
            calibrate_method=CalibrationMethod.MinMax,
            op_types_to_quantize=args.operators,
        )

    validate_model(output)


if __name__ == "__main__":
    main()
