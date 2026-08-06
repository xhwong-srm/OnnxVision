"""Measure ONNX Runtime throughput and shape behavior across batch sizes."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np


def _parse_batches(value: str) -> tuple[int, ...]:
    batches = tuple(sorted({int(item) for item in value.split(",")}))
    if not batches or any(item <= 0 for item in batches):
        raise ValueError("batch sizes must be positive integers")
    return batches


def _shape_dimension(value: int | str | None, fallback: int) -> int:
    return int(value) if isinstance(value, int) and value > 0 else fallback


def _input_shape(session, batch: int, height: int, width: int) -> tuple[int, ...]:
    input_info = session.get_inputs()[0]
    if len(input_info.shape) != 4:
        raise ValueError(f"expected one rank-4 input, got {input_info.shape}")
    if "bw8" in input_info.name.casefold():
        return batch, 1, height, width
    if "c24" in input_info.name.casefold() or input_info.shape[-1] == 3:
        return batch, height, width, 3
    return batch, 3, height, width


def _input_value(session, batch: int, height: int, width: int, seed: int) -> np.ndarray:
    input_info = session.get_inputs()[0]
    shape = _input_shape(session, batch, height, width)
    generator = np.random.default_rng(seed)
    if input_info.type == "tensor(uint8)":
        return generator.integers(0, 256, size=shape, dtype=np.uint8)
    if input_info.type == "tensor(float)":
        return generator.random(shape, dtype=np.float32)
    raise ValueError(f"unsupported input type: {input_info.type}")


def _providers(requested: str, ort) -> list[str]:
    available = set(ort.get_available_providers())
    if requested == "cpu":
        return ["CPUExecutionProvider"]
    if requested == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider is unavailable; available={sorted(available)}")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return (["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"])


def _measure(session, input_name: str, value: np.ndarray, warmups: int, repeats: int) -> dict[str, object]:
    for _ in range(warmups):
        session.run(None, {input_name: value})
    elapsed = []
    output_shapes = None
    for _ in range(repeats):
        started = time.perf_counter()
        outputs = session.run(None, {input_name: value})
        elapsed.append((time.perf_counter() - started) * 1000.0)
        output_shapes = [list(output.shape) for output in outputs]
    elapsed.sort()
    batch = value.shape[0]
    median_ms = statistics.median(elapsed)
    p95_ms = elapsed[min(len(elapsed) - 1, int(len(elapsed) * 0.95))]
    return {
        "batch": batch,
        "input_shape": list(value.shape),
        "output_shapes": output_shapes,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "images_per_second": batch * 1000.0 / median_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--batches", help="comma-separated batches; defaults to the fixed model batch or 1,2,4,8,16,32")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be non-negative and repeats must be positive")

    import onnxruntime as ort

    model = args.model.expanduser().resolve()
    providers = _providers(args.provider, ort)
    session = ort.InferenceSession(str(model), providers=providers)
    input_info = session.get_inputs()[0]
    declared_batch = input_info.shape[0]
    fixed_batch = int(declared_batch) if isinstance(declared_batch, int) and declared_batch > 0 else None
    requested_batches = _parse_batches(args.batches) if args.batches else (
        (fixed_batch,) if fixed_batch is not None else _parse_batches("1,2,4,8,16,32")
    )
    result: dict[str, object] = {
        "model": str(model),
        "providers": session.get_providers(),
        "input_name": input_info.name,
        "input_type": input_info.type,
        "declared_input_shape": list(input_info.shape),
        "batch_contract": {"mode": "fixed", "size": fixed_batch} if fixed_batch is not None else {"mode": "dynamic"},
        "batches": [],
    }
    for batch in requested_batches:
        value = _input_value(session, batch, args.height, args.width, args.seed + batch)
        try:
            measurement = _measure(session, input_info.name, value, args.warmups, args.repeats)
            result["batches"].append({"status": "passed", **measurement})
        except Exception as error:
            result["batches"].append({
                "status": "failed",
                "batch": batch,
                "input_shape": list(value.shape),
                "error": str(error),
            })

    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
