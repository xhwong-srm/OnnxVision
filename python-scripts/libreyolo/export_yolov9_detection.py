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
from typing import Any

import numpy as np
import onnx
from PIL import Image


YOLO_SCRIPTS = Path(__file__).resolve().parents[1] / "yolo"
sys.path.insert(0, str(YOLO_SCRIPTS))
from export_yolo26_detection import (  # noqa: E402
    Detection,
    canonicalize,
    dataset_names,
    dataset_root,
    image_files_for_split,
    load_dataset_yaml,
    match_detections,
    parity_metrics,
    print_quality,
    quality_metrics,
    read_ground_truth,
    wrap_bw8,
    wrap_c24,
)


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
    parser.add_argument(
        "--data", type=Path,
        help="YOLO data.yaml or dataset directory used for post-export agreement evaluation",
    )
    parser.add_argument("--validation-split", default="test", help="Dataset split to evaluate (default: test)")
    parser.add_argument(
        "--validation-limit",
        type=int,
        default=0,
        help="Maximum dataset images to evaluate; 0 evaluates the complete split",
    )
    parser.add_argument(
        "--validation-confidence",
        type=float,
        default=0.25,
        help="Confidence threshold for quality and PT-vs-ONNX agreement metrics",
    )
    parser.add_argument(
        "--validation-iou",
        type=float,
        default=0.50,
        help="IoU threshold for ground-truth and agreement matching",
    )
    parser.add_argument("--validation-report", type=Path, help="JSON agreement report path")
    return parser.parse_args()


def resolve_dataset_yaml(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "data.yaml"
    if not resolved.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {resolved}")
    return resolved


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return value.detach().cpu().numpy()


def run_native(
    model: Any,
    image_path: Path,
    resolution: int,
    confidence: float,
    nms_iou: float,
    grayscale: bool = False,
) -> tuple[Detection, ...]:
    # The existing BW8/C24 wrappers use stretch-resize. Feed the native model
    # the same square image so parity measures the deployed preprocessing path.
    with Image.open(image_path) as image:
        source_image = image.convert("L").convert("RGB") if grayscale else image.convert("RGB")
        source = np.asarray(source_image.resize((resolution, resolution)), dtype=np.uint8)
    result = model.predict(
        source=source,
        imgsz=resolution,
        conf=confidence,
        iou=nms_iou,
        max_det=300,
        color_format="rgb",
    )
    if isinstance(result, (list, tuple)):
        result = result[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return ()
    xyxy = _to_numpy(boxes.xyxy)
    scores = _to_numpy(boxes.conf)
    classes = _to_numpy(boxes.cls)
    scale = np.asarray([resolution, resolution, resolution, resolution], dtype=np.float32)
    return tuple(
        Detection(np.asarray(box, dtype=np.float32) / scale, float(score), int(class_id))
        for box, score, class_id in zip(xyxy, scores, classes)
        if np.isfinite(score) and float(score) >= confidence
    )


def run_onnx(
    session: Any,
    image_path: Path,
    resolution: int,
    confidence: float,
) -> tuple[Detection, ...]:
    input_info = session.get_inputs()[0]
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        if input_info.type == "tensor(float)":
            value = np.asarray(rgb.resize((resolution, resolution)), dtype=np.float32)
            value = value.transpose(2, 0, 1)[None] / 255.0
        elif input_info.type == "tensor(uint8)" and input_info.name.startswith("images_bw8"):
            value = np.asarray(rgb.convert("L"), dtype=np.uint8)[None, None]
        elif input_info.type == "tensor(uint8)":
            value = np.asarray(rgb, dtype=np.uint8)[..., ::-1].copy()[None]
        else:
            raise ValueError(f"Unsupported YOLOv9 ONNX input: {input_info.type} {input_info.shape}")
    outputs = {
        item.name: np.asarray(result)
        for item, result in zip(session.get_outputs(), session.run(None, {input_info.name: value}))
    }
    return tuple(
        Detection(np.asarray(box, dtype=np.float32), float(score), int(class_id))
        for box, score, class_id in zip(
            outputs["boxes"][0], outputs["scores"][0], outputs["class_ids"][0]
        )
        if np.isfinite(score) and float(score) >= confidence
    )


def evaluate(
    model: Any,
    checkpoint: Path,
    generated: list[Path],
    data_path: Path,
    split: str,
    limit: int,
    resolution: int,
    inference_floor: float,
    nms_iou: float,
    confidence: float,
    iou_threshold: float,
    report_path: Path,
) -> None:
    import onnxruntime as ort

    document = load_dataset_yaml(data_path)
    names = dataset_names(document)
    if split not in document:
        raise ValueError(f"Dataset YAML does not define split '{split}': {data_path}")
    images = image_files_for_split(dataset_root(data_path, document), document[split])
    if limit:
        images = images[:limit]
    if not images:
        raise ValueError(f"Dataset split '{split}' contains no supported images")
    samples = [read_ground_truth(path) for path in images]
    print(
        f"validation_dataset={data_path}; split={split}; images={len(samples)}; "
        f"confidence={confidence}; iou={iou_threshold}"
    )

    predictions: dict[str, dict[Path, tuple[Detection, ...]]] = {
        "native_pt": {},
        "native_pt_bw8": {},
    }
    references: dict[str, str] = {}
    for index, sample in enumerate(samples, 1):
        predictions["native_pt"][sample.image] = run_native(
            model, sample.image, resolution, inference_floor, nms_iou=nms_iou
        )
        predictions["native_pt_bw8"][sample.image] = run_native(
            model, sample.image, resolution, inference_floor, nms_iou=nms_iou, grayscale=True
        )
        print(f"validation_progress={index}/{len(samples)} image={sample.image.name}")
    for path in generated:
        session = ort.InferenceSession(str(path.resolve()), providers=["CPUExecutionProvider"])
        backend = path.name
        input_info = session.get_inputs()[0]
        references[backend] = (
            "native_pt_bw8"
            if input_info.type == "tensor(uint8)" and input_info.name.startswith("images_bw8")
            else "native_pt"
        )
        predictions[backend] = {
            sample.image: run_onnx(session, sample.image, resolution, inference_floor)
            for sample in samples
        }

    quality = {
        backend: quality_metrics(samples, values, names, iou_threshold, confidence)
        for backend, values in predictions.items()
    }
    thresholded = {
        backend: {
            path: tuple(item for item in values[path] if item.score >= confidence)
            for path in values
        }
        for backend, values in predictions.items()
    }
    parity = {
        backend: parity_metrics(thresholded[references[backend]], values, iou_threshold)
        for backend, values in thresholded.items()
        if backend in references
    }
    per_image: list[dict[str, Any]] = []
    for sample in samples:
        item: dict[str, Any] = {
            "image": str(sample.image),
            "annotation": str(sample.label),
            "ground_truth": len(sample.ground_truth),
            "backends": {},
        }
        for backend, values in thresholded.items():
            counts = match_detections(values[sample.image], sample.ground_truth, iou_threshold)
            item["backends"][backend] = {
                "predictions": len(values[sample.image]),
                "true_positive": counts[0],
                "false_positive": counts[1],
                "false_negative": counts[2],
                "mean_matched_iou": float(np.mean(counts[3])) if counts[3] else None,
            }
        item["native_vs_onnx"] = {
            backend: parity_metrics(
                {sample.image: thresholded[references[backend]][sample.image]},
                {sample.image: values[sample.image]},
                iou_threshold,
            )
            for backend, values in thresholded.items()
            if backend in references
        }
        per_image.append(item)
    for backend, values in quality.items():
        print_quality(backend, values)
    for backend, values in parity.items():
        print(
            f"parity/{backend} reference={references[backend]}: matched={values['matched_detections']} "
            f"native_unmatched={values['native_unmatched']} onnx_unmatched={values['onnx_unmatched']} "
            f"mean_IoU={values['mean_pair_iou']} mean_box_delta={values['mean_abs_box_coordinate_delta']} "
            f"mean_score_delta={values['mean_abs_score_delta']}"
        )
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "dataset": str(data_path),
        "split": split,
        "images": [str(sample.image) for sample in samples],
        "models": [str(path) for path in generated],
        "agreement_references": references,
        "settings": {
            "resolution": resolution,
            "inference_floor": inference_floor,
            "confidence": confidence,
            "nms_iou": nms_iou,
            "iou": iou_threshold,
            "native_preprocess": "libreyolo_yolov9_rgb_stretch",
            "onnx_provider": "CPUExecutionProvider",
        },
        "interpretation": {
            "quality": "Predictions compared with YOLO ground-truth annotations.",
            "parity": "Each ONNX result compared with native PyTorch on the same stretched test image and confidence floor.",
        },
        "quality": quality,
        "native_vs_onnx": parity,
        "per_image": per_image,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"validation_report={report_path}")


def main() -> None:
    args = parse_args()
    if args.resolution < 32 or args.resolution % 32:
        raise ValueError("--resolution must be at least 32 and divisible by 32")
    if not 0.0 <= args.confidence <= 1.0 or not 0.0 < args.iou <= 1.0:
        raise ValueError("--confidence must be in [0,1] and --iou in (0,1]")
    if args.max_det < 1 or args.validation_limit < 0:
        raise ValueError("--max-det must be positive and --validation-limit cannot be negative")
    if not 0.0 <= args.validation_confidence <= 1.0 or not 0.0 < args.validation_iou <= 1.0:
        raise ValueError("--validation-confidence must be in [0,1] and --validation-iou in (0,1]")
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

    generated = [output]
    if not args.skip_embedded_preprocessing:
        bw8 = (args.bw8_output or output.with_name(output.stem + "-bw8.onnx")).resolve()
        c24 = (args.c24_output or output.with_name(output.stem + "-c24.onnx")).resolve()
        wrap_bw8(output, bw8, args.resolution)
        wrap_c24(output, c24, args.resolution)
        print(f"created_bw8={bw8}")
        print(f"created_c24={c24}")

        generated = [output, bw8, c24]

    if args.data:
        data = resolve_dataset_yaml(args.data)
        validation_model = model
        if getattr(model.device, "type", "cpu") != "cpu":
            validation_model = LibreYOLO(str(checkpoint), device="cpu", task="detect")
        report_path = (
            args.validation_report.expanduser().resolve()
            if args.validation_report
            else output.with_suffix(".validation.json")
        )
        evaluate(
            model=validation_model,
            checkpoint=checkpoint,
            generated=generated,
            data_path=data,
            split=args.validation_split,
            limit=args.validation_limit,
            resolution=args.resolution,
            inference_floor=args.confidence,
            nms_iou=args.iou,
            confidence=args.validation_confidence,
            iou_threshold=args.validation_iou,
            report_path=report_path,
        )


if __name__ == "__main__":
    main()
