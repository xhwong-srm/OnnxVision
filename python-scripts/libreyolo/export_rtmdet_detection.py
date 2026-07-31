"""Export LibreYOLO RTMDet weights to the repository ONNX detection contract.

The exported graph includes class-aware NMS and emits normalized ``boxes``
[1,N,4], ``scores`` [1,N], and int64 ``class_ids`` [1,N]. Optional BW8 and
C24 variants embed the preprocessing expected by the .NET ObjectDetector.
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import onnx
import torch
import torch.nn as nn


YOLO_SCRIPTS = Path(__file__).resolve().parents[1] / "yolo"
sys.path.insert(0, str(YOLO_SCRIPTS))
from export_yolo26_detection import canonicalize, wrap_bw8, wrap_c24  # noqa: E402


class RTMDetExportLayout(nn.Module):
    """Adapt RTMDet's [B,N,4+C] output to LibreYOLO's NMS layout."""

    def __init__(self, detector: nn.Module):
        super().__init__()
        self.detector = detector

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.detector(images).permute(0, 2, 1)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="LibreYOLO RTMDet .pt checkpoint")
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--output", type=Path, default=Path("artifacts/rtmdet-detection.onnx"))
    parser.add_argument("--bw8-output", type=Path)
    parser.add_argument("--c24-output", type=Path)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--confidence", type=float, default=0.001, help="Embedded NMS confidence")
    parser.add_argument("--iou", type=float, default=0.70, help="Embedded NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--simplify", action=BooleanOptionalAction, default=True,
        help="Simplify the intermediate ONNX graph (default: enabled)",
    )
    parser.add_argument("--skip-embedded-preprocessing", action="store_true")
    parser.add_argument("--device", default="cpu", help='Export device, for example "cpu" or "0"')
    return parser.parse_args()


def simplify_onnx(path: Path) -> None:
    try:
        from onnxslim import slim
    except ImportError:
        print("warning: onnxslim is unavailable; skipping simplification")
        return
    simplified = slim(onnx.load(path))
    onnx.checker.check_model(simplified)
    onnx.save(simplified, path)


def main() -> None:
    args = parse_args()
    if args.resolution < 32 or args.resolution % 32:
        raise ValueError("--resolution must be at least 32 and divisible by 32")
    if not 0.0 <= args.confidence <= 1.0 or not 0.0 < args.iou <= 1.0:
        raise ValueError("--confidence must be in [0,1] and --iou in (0,1]")
    if args.max_det < 1:
        raise ValueError("--max-det must be positive")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    try:
        from libreyolo import LibreYOLO
        from libreyolo.export.nms import EmbeddedNMSDetector
    except ImportError as error:
        raise RuntimeError("LibreYOLO is not installed. Run: uv add libreyolo") from error

    model = LibreYOLO(str(checkpoint), device=args.device, task="detect")
    if model._get_model_name() != "rtmdet":
        raise ValueError(f"Expected an RTMDet checkpoint, received family {model._get_model_name()!r}")
    model.model.eval()
    model.model.head.export = True
    export_model = EmbeddedNMSDetector(
        RTMDetExportLayout(model.model),
        conf=args.confidence,
        iou=args.iou,
        max_det=args.max_det,
    ).eval()
    device = next(model.model.parameters()).device
    dummy = torch.zeros(1, 3, args.resolution, args.resolution, device=device)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate = output.with_name(output.stem + ".libreyolo.onnx")
    torch.onnx.export(
        export_model,
        dummy,
        str(intermediate),
        input_names=["images"],
        output_names=["output", "raw"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    if args.simplify:
        simplify_onnx(intermediate)

    names = {int(index): str(name) for index, name in model.names.items()}
    intermediate_graph = onnx.load(intermediate)
    if [item.name for item in intermediate_graph.graph.output] != ["output", "raw"]:
        raise ValueError(
            "Unexpected LibreYOLO export outputs: "
            f"{[item.name for item in intermediate_graph.graph.output]}"
        )
    del intermediate_graph.graph.output[1:]
    onnx.save(intermediate_graph, intermediate)
    canonicalize(intermediate, output, args.resolution, names)
    graph = onnx.load(output)
    metadata = {item.key: item.value for item in graph.metadata_props}
    metadata.update(
        {
            "source_model": "libreyolo-rtmdet",
            "nms": "true",
            "nms_required": "false",
            "nms_conf": str(args.confidence),
            "nms_iou": str(args.iou),
            "max_det": str(args.max_det),
            "libreyolo_names": json.dumps(names),
        }
    )
    del graph.metadata_props[:]
    for key, value in metadata.items():
        item = graph.metadata_props.add()
        item.key, item.value = key, value
    onnx.save(graph, output)
    print(f"created_onnx={output}")

    if not args.skip_embedded_preprocessing:
        bw8 = (args.bw8_output or output.with_name(output.stem + "-bw8.onnx")).resolve()
        c24 = (args.c24_output or output.with_name(output.stem + "-c24.onnx")).resolve()
        wrap_bw8(output, bw8, args.resolution)
        wrap_c24(output, c24, args.resolution)
        print(f"created_bw8={bw8}")
        print(f"created_c24={c24}")


if __name__ == "__main__":
    main()
