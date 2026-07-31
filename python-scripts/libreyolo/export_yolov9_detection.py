"""Export LibreYOLO YOLOv9 weights to the repository ONNX detection contract.

LibreYOLO's public exporter embeds class-aware NMS. The final model emits
normalized ``boxes`` [1,N,4], ``scores`` [1,N], and int64 ``class_ids`` [1,N].
Optional BW8 and C24 variants embed preprocessing for the .NET ObjectDetector.
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import onnx


YOLO_SCRIPTS = Path(__file__).resolve().parents[1] / "yolo"
sys.path.insert(0, str(YOLO_SCRIPTS))
from export_yolo26_detection import canonicalize, wrap_bw8, wrap_c24  # noqa: E402


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="LibreYOLO YOLOv9 .pt checkpoint")
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--output", type=Path, default=Path("artifacts/yolov9-detection.onnx"))
    parser.add_argument("--bw8-output", type=Path)
    parser.add_argument("--c24-output", type=Path)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--confidence", type=float, default=0.001, help="Embedded NMS confidence")
    parser.add_argument("--iou", type=float, default=0.70, help="Embedded NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--simplify", action=BooleanOptionalAction, default=True,
        help="Simplify the LibreYOLO ONNX graph (default: enabled)",
    )
    parser.add_argument("--skip-embedded-preprocessing", action="store_true")
    parser.add_argument("--device", default="cpu", help='Export device, for example "cpu" or "0"')
    return parser.parse_args()


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
    except ImportError as error:
        raise RuntimeError("LibreYOLO is not installed. Run: uv add libreyolo") from error

    model = LibreYOLO(str(checkpoint), device=args.device, task="detect")
    if model._get_model_name() != "yolo9":
        raise ValueError(f"Expected a YOLOv9 checkpoint, received family {model._get_model_name()!r}")
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.resolution,
            batch=1,
            dynamic=False,
            simplify=args.simplify,
            nms=True,
            conf=args.confidence,
            iou=args.iou,
            max_det=args.max_det,
            opset=args.opset,
            device=args.device,
        )
    ).resolve()

    exported_graph = onnx.load(exported)
    output_names = [item.name for item in exported_graph.graph.output]
    if output_names != ["output", "raw"]:
        raise ValueError(f"Unexpected LibreYOLO embedded-NMS outputs: {output_names}")
    del exported_graph.graph.output[1:]
    onnx.save(exported_graph, exported)

    output = args.output.resolve()
    names = {int(index): str(name) for index, name in model.names.items()}
    canonicalize(exported, output, args.resolution, names)
    graph = onnx.load(output)
    metadata = {item.key: item.value for item in graph.metadata_props}
    metadata.update(
        {
            "source_model": "libreyolo-yolov9",
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
    onnx.checker.check_model(graph)
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
