"""Wrap the exported PP-LCNetV2 core in the shared classification contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx

from vision_workflows.backends.common import (
    classification_contract,
    embedded_output_paths,
    metadata_for_contract,
    set_onnx_metadata,
    validate_onnx,
    wrap_embedded_variants,
)


def _opset18_core(core: Path, output: Path) -> Path:
    model = onnx.load(str(core))
    default_opset = next(
        (item.version for item in model.opset_import if item.domain == ""),
        0,
    )
    if default_opset >= 18:
        return core

    converted = output.with_name(f"{output.stem}-core-opset18.onnx")
    converted.parent.mkdir(parents=True, exist_ok=True)
    converted_model = onnx.version_converter.convert_version(model, 18)
    onnx.checker.check_model(converted_model)
    onnx.save(converted_model, str(converted))
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help=".onnx base path; -bw8 and -c24 are emitted")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--core-outputs-probabilities",
        action="store_true",
        help="Do not add a softmax when the exported core already emits probabilities",
    )
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    core = args.core.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not core.is_file():
        raise FileNotFoundError(core)
    core = _opset18_core(core, output)
    outputs = embedded_output_paths(output)
    paths = wrap_embedded_variants(
        core,
        outputs,
        image_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        pixel_scale=255.0,
        resize_antialias=True,
        batch_size=args.batch_size,
        apply_softmax=not args.core_outputs_probabilities,
        output_names=("probabilities",),
        resize_mode="stretch",
    )

    checks: dict[str, object] = {"core": str(core), "variants": {}}
    for variant, path in zip(("bw8", "c24"), paths):
        contract = classification_contract(
            ("flipped", "normal"),
            input_variant=variant,
            batch_size=args.batch_size,
        )
        set_onnx_metadata(path, metadata_for_contract(contract))
        checks["variants"][variant] = {
            "path": str(path),
            "checks": validate_onnx(path, contract),
        }
    print(json.dumps(checks, indent=2, default=str))


if __name__ == "__main__":
    main()
