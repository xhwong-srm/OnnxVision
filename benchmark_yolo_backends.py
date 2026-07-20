from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = sorted(
        path
        for path in args.dataset.rglob("*")
        if path.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    )
    if not images:
        raise SystemExit(f"No images found below {args.dataset}")

    load_started = time.perf_counter()
    model = YOLO(str(args.model), task="classify")
    load_ms = (time.perf_counter() - load_started) * 1_000

    for index in range(args.warmup):
        model.predict(
            str(images[index % len(images)]), imgsz=224, batch=1, verbose=False, device=args.device
        )

    elapsed_runs: list[float] = []
    inference_runs: list[float] = []
    correct = 0
    predictions: list[tuple[str, str, float]] = []

    for run in range(args.runs):
        started = time.perf_counter()
        results = [
            model.predict(str(image), imgsz=224, batch=1, verbose=False, device=args.device)[0]
            for image in images
        ]
        elapsed = time.perf_counter() - started
        elapsed_runs.append(elapsed)
        inference_runs.append(sum(result.speed["inference"] for result in results) / len(results))

        if run == 0:
            for image, result in zip(images, results, strict=True):
                predicted = result.names[result.probs.top1]
                confidence = float(result.probs.top1conf)
                expected = image.parent.name
                correct += predicted == expected
                if predicted != expected:
                    predictions.append((str(image), predicted, confidence))

    per_image_ms = [elapsed * 1_000 / len(images) for elapsed in elapsed_runs]
    print(f"model={args.model}")
    print(
        f"images={len(images)} runs={args.runs} warmup={args.warmup} "
        f"batch=1 imgsz=224 device={args.device or 'default'}"
    )
    print(f"load_ms={load_ms:.3f}")
    print(f"accuracy={correct / len(images):.6f} ({correct}/{len(images)})")
    print(f"wall_ms_per_image_median={statistics.median(per_image_ms):.3f}")
    print(f"wall_ms_per_image_runs={','.join(f'{value:.3f}' for value in per_image_ms)}")
    print(f"reported_inference_ms_median={statistics.median(inference_runs):.3f}")
    for path, predicted, confidence in predictions:
        print(f"error={path}|predicted={predicted}|confidence={confidence:.6f}")


if __name__ == "__main__":
    main()
