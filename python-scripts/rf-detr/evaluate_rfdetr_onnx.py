"""Evaluate one or more OnnxVision RF-DETR detectors on a COCO split."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    score: float
    class_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    return parser.parse_args()


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def nms(detections: list[Detection], threshold: float) -> list[Detection]:
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: item.score, reverse=True):
        if any(
            item.class_index == candidate.class_index
            and iou(item.box, candidate.box) > threshold
            for item in kept
        ):
            continue
        kept.append(candidate)
    return kept


def infer(
    session: ort.InferenceSession, image_path: Path, nms_iou: float
) -> tuple[list[Detection], int, int]:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        bgr = np.asarray(rgb, dtype=np.uint8)[..., ::-1].copy()[None]
    boxes, scores, class_ids = session.run(None, {session.get_inputs()[0].name: bgr})
    detections = [
        Detection(
            (
                float(box[0]) * width,
                float(box[1]) * height,
                float(box[2]) * width,
                float(box[3]) * height,
            ),
            float(score),
            int(class_index),
        )
        for box, score, class_index in zip(boxes[0], scores[0], class_ids[0])
    ]
    return nms(detections, nms_iou), width, height


def operational_metrics(
    predictions: dict[int, list[Detection]],
    annotations: dict,
    confidence: float,
) -> tuple[int, int, int, float, float, float]:
    ground_truth: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    category_to_class = {
        category["id"]: index for index, category in enumerate(annotations["categories"])
    }
    for item in annotations["annotations"]:
        x, y, width, height = item["bbox"]
        ground_truth.setdefault(item["image_id"], []).append(
            (category_to_class[item["category_id"]], (x, y, x + width, y + height))
        )

    true_positive = false_positive = false_negative = 0
    for image in annotations["images"]:
        expected = ground_truth.get(image["id"], [])
        matched: set[int] = set()
        candidates = [
            detection
            for detection in predictions[image["id"]]
            if detection.score >= confidence
        ]
        for detection in sorted(candidates, key=lambda item: item.score, reverse=True):
            best_index = -1
            best_iou = 0.5
            for index, (class_index, box) in enumerate(expected):
                if index in matched or class_index != detection.class_index:
                    continue
                overlap = iou(detection.box, box)
                if overlap >= best_iou:
                    best_iou = overlap
                    best_index = index
            if best_index >= 0:
                matched.add(best_index)
                true_positive += 1
            else:
                false_positive += 1
        false_negative += len(expected) - len(matched)

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return true_positive, false_positive, false_negative, precision, recall, f1


def evaluate(model_path: Path, dataset: Path, confidence: float, nms_iou: float) -> None:
    annotation_path = dataset / "_annotations.coco.json"
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    category_ids = [item["id"] for item in annotations["categories"]]
    session = ort.InferenceSession(str(model_path.resolve()), providers=["CPUExecutionProvider"])
    predictions: dict[int, list[Detection]] = {}
    coco_results: list[dict[str, object]] = []

    for image in annotations["images"]:
        detections, _, _ = infer(session, dataset / image["file_name"], nms_iou)
        predictions[image["id"]] = detections
        for detection in detections:
            if detection.score < 0.001:
                continue
            x1, y1, x2, y2 = detection.box
            coco_results.append({
                "image_id": image["id"],
                "category_id": category_ids[detection.class_index],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": detection.score,
            })

    with contextlib.redirect_stdout(io.StringIO()):
        coco_ground_truth = COCO(str(annotation_path))
        coco_predictions = coco_ground_truth.loadRes(coco_results)
        evaluator = COCOeval(coco_ground_truth, coco_predictions, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    tp, fp, fn, precision, recall, f1 = operational_metrics(
        predictions, annotations, confidence
    )
    print(f"model={model_path}")
    print(
        f"coco_ap={evaluator.stats[0]:.6f} coco_ap50={evaluator.stats[1]:.6f} "
        f"coco_ar100={evaluator.stats[8]:.6f}"
    )
    print(
        f"threshold={confidence:.3f} tp={tp} fp={fp} fn={fn} "
        f"precision={precision:.6f} recall={recall:.6f} f1={f1:.6f}"
    )


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    for model in args.models:
        evaluate(model.expanduser().resolve(), dataset, args.confidence, args.nms_iou)


if __name__ == "__main__":
    main()
