"""Benchmark timm training configurations and select the fastest safe CUDA setup.

The tuner measures the real training path (image loading, host-to-device transfer,
forward pass, backward pass, and optimizer step). Each trial runs in a separate
process so a CUDA out-of-memory failure cannot affect later trials.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from train_timm_classification import MODEL_NAME, image_transform, make_dataset


RESULT_PREFIX = "TIMM_GPU_TUNER_RESULT="
SELECTED_CONFIG_PATH = Path(__file__).resolve().with_name("tune_timm_gpu.selected.json")


@dataclass(frozen=True)
class TrialConfig:
    batch: int
    workers: int
    prefetch_factor: int | None
    persistent_workers: bool
    pin_memory: bool
    amp: bool
    amp_dtype: str
    channels_last: bool
    compile: bool


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("cannot be negative")
    return parsed


def fraction(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return parsed


def parse_worker_candidates(value: str) -> list[int]:
    if value.lower() == "auto":
        cpu_count = os.cpu_count() or 1
        worker_limit = min(16, max(1, cpu_count - 2))
        workers = [0]
        candidate = 1
        while candidate <= worker_limit:
            workers.append(candidate)
            candidate *= 2
        if workers[-1] != worker_limit:
            workers.append(worker_limit)
        return workers
    try:
        workers = sorted({int(item.strip()) for item in value.split(",")})
    except ValueError as error:
        raise argparse.ArgumentTypeError("use 'auto' or comma-separated integers") from error
    if not workers or workers[0] < 0:
        raise argparse.ArgumentTypeError("worker values must be non-negative")
    return workers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("images/seal_dataset_v2"),
        help="Dataset root containing a train/<class> folder structure",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-batch", type=positive_int, default=16)
    parser.add_argument("--max-batch", type=positive_int, default=512)
    parser.add_argument(
        "--workers",
        type=parse_worker_candidates,
        default=parse_worker_candidates("auto"),
        help="DataLoader worker candidates: 'auto' or comma-separated values",
    )
    parser.add_argument(
        "--prefetch-factors",
        type=parse_worker_candidates,
        default=None,
        help=(
            "Prefetch candidates used when workers > 0; default derives 1,2,4,8 "
            "within a host-RAM budget"
        ),
    )
    parser.add_argument(
        "--max-prefetch-ram-fraction",
        type=fraction,
        default=0.25,
        help="Maximum fraction of currently available RAM budgeted for prefetched batches",
    )
    parser.add_argument("--warmup-steps", type=non_negative_int, default=3)
    parser.add_argument("--measure-steps", type=positive_int, default=10)
    parser.add_argument(
        "--final-measure-steps",
        type=positive_int,
        default=100,
        help="Measured steps used to validate the strongest short-run candidates",
    )
    parser.add_argument(
        "--final-candidates",
        type=positive_int,
        default=3,
        help="Number of short-run winners remeasured with --final-measure-steps",
    )
    parser.add_argument(
        "--max-vram-fraction",
        type=fraction,
        default=0.90,
        help="Reject configurations whose peak reserved VRAM exceeds this fraction",
    )
    parser.add_argument(
        "--top-compute-configs",
        type=positive_int,
        default=3,
        help="Number of compute/batch candidates advanced to DataLoader tuning",
    )
    parser.add_argument(
        "--test-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include torch.compile for the strongest uncompiled candidate",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "gpu-tuning",
        help="Directory for JSON and CSV benchmark reports",
    )

    # Internal options used by isolated child trials.
    parser.add_argument("--_trial", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_batch", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_workers", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_prefetch-factor", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_persistent-workers", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    parser.add_argument("--_pin-memory", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    parser.add_argument("--_amp", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    parser.add_argument("--_amp-dtype", choices=("float16", "bfloat16"), help=argparse.SUPPRESS)
    parser.add_argument("--_channels-last", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    parser.add_argument("--_compile", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    return parser


def emit_trial_result(result: dict) -> None:
    print(f"{RESULT_PREFIX}{json.dumps(result, separators=(',', ':'))}")


def is_cuda_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return isinstance(error, torch.OutOfMemoryError) or (
        "cuda" in message and ("out of memory" in message or "memory allocation" in message)
    )


def run_training_trial(args: argparse.Namespace) -> int:
    config = TrialConfig(
        batch=args._batch,
        workers=args._workers,
        prefetch_factor=args._prefetch_factor if args._workers > 0 else None,
        persistent_workers=bool(args._persistent_workers) and args._workers > 0,
        pin_memory=bool(args._pin_memory),
        amp=bool(args._amp),
        amp_dtype=args._amp_dtype,
        channels_last=bool(args._channels_last),
        compile=bool(args._compile),
    )
    result: dict[str, Any] = {"config": asdict(config)}
    try:
        if config.batch is None or config.workers is None or config.amp_dtype is None:
            raise ValueError("internal trial arguments are incomplete")

        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("GPU tuning requires a CUDA device")

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

        base_model: Any = timm.create_model(args.model, pretrained=True, num_classes=0)
        train_transform, data_config = image_transform(base_model, train=True)
        train_set = make_dataset(args.data.resolve() / "train", train_transform)
        required_samples = config.batch * (args.warmup_steps + args.measure_steps)
        sampler = RandomSampler(train_set, replacement=True, num_samples=required_samples)
        loader_args = {
            "batch_size": config.batch,
            "sampler": sampler,
            "num_workers": config.workers,
            "pin_memory": config.pin_memory,
            "persistent_workers": config.persistent_workers,
        }
        if config.workers > 0:
            loader_args["prefetch_factor"] = config.prefetch_factor
        loader = DataLoader(train_set, **loader_args)

        base_model.reset_classifier(len(train_set.classes))
        if config.channels_last:
            base_model.to(device=device, memory_format=torch.channels_last)
        else:
            base_model.to(device)
        model: Any = torch.compile(base_model) if config.compile else base_model
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        amp_dtype = torch.float16 if config.amp_dtype == "float16" else torch.bfloat16
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=config.amp and amp_dtype == torch.float16,
        )
        non_blocking = config.pin_memory

        def step(images: torch.Tensor, targets: torch.Tensor) -> None:
            images = images.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            if config.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=config.amp,
            ):
                loss = criterion(model(images), targets)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        iterator = iter(loader)
        for _ in range(args.warmup_steps):
            step(*next(iterator))
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        started = time.perf_counter()
        measured_samples = 0
        for _ in range(args.measure_steps):
            images, targets = next(iterator)
            step(images, targets)
            measured_samples += targets.size(0)
        torch.cuda.synchronize(device)
        elapsed_seconds = time.perf_counter() - started

        properties = torch.cuda.get_device_properties(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        total_memory = properties.total_memory
        result.update(
            {
                "status": "success",
                "samples_per_second": measured_samples / elapsed_seconds,
                "milliseconds_per_batch": elapsed_seconds * 1000 / args.measure_steps,
                "peak_allocated_mb": peak_allocated / (1024**2),
                "peak_reserved_mb": peak_reserved / (1024**2),
                "total_vram_mb": total_memory / (1024**2),
                "vram_fraction": peak_reserved / total_memory,
                "measured_samples": measured_samples,
                "elapsed_seconds": elapsed_seconds,
                "input_size": data_config["input_size"],
            }
        )
    except BaseException as error:
        result.update(
            {
                "status": "oom" if is_cuda_oom(error) else "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        )
    emit_trial_result(result)
    return 0 if result["status"] in {"success", "oom"} else 1


def trial_command(
    args: argparse.Namespace,
    config: TrialConfig,
    measure_steps: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_trial",
        "--data",
        str(args.data),
        "--model",
        args.model,
        "--device",
        args.device,
        "--warmup-steps",
        str(args.warmup_steps),
        "--measure-steps",
        str(args.measure_steps if measure_steps is None else measure_steps),
        "--seed",
        str(args.seed),
        "--_batch",
        str(config.batch),
        "--_workers",
        str(config.workers),
        "--_amp-dtype",
        config.amp_dtype,
    ]
    if config.prefetch_factor is not None:
        command.extend(["--_prefetch-factor", str(config.prefetch_factor)])
    for name, enabled in (
        ("persistent-workers", config.persistent_workers),
        ("pin-memory", config.pin_memory),
        ("amp", config.amp),
        ("channels-last", config.channels_last),
        ("compile", config.compile),
    ):
        command.extend([f"--_{name}", str(int(enabled))])
    return command


def run_isolated_trial(
    args: argparse.Namespace,
    config: TrialConfig,
    *,
    measure_steps: int | None = None,
    benchmark_stage: str = "screening",
) -> dict:
    print(
        "  "
        f"batch={config.batch:<4} workers={config.workers} "
        f"amp={'off' if not config.amp else config.amp_dtype:<8} "
        f"channels_last={str(config.channels_last):<5} "
        f"pin={str(config.pin_memory):<5} "
        f"prefetch={config.prefetch_factor or '-':<2} "
        f"compile={config.compile}",
        flush=True,
    )
    completed = subprocess.run(
        trial_command(args, config, measure_steps),
        capture_output=True,
        text=True,
        check=False,
    )
    result_line = next(
        (line for line in reversed(completed.stdout.splitlines()) if line.startswith(RESULT_PREFIX)),
        None,
    )
    if result_line is None:
        error = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        result = {"config": asdict(config), "status": "failed", "error": error}
    else:
        result = json.loads(result_line[len(RESULT_PREFIX) :])
    result["benchmark_stage"] = benchmark_stage

    if result["status"] == "success":
        print(
            f"    {result['samples_per_second']:.1f} samples/s; "
            f"peak reserved={result['peak_reserved_mb']:.0f} MiB "
            f"({result['vram_fraction']:.1%})"
        )
    else:
        print(f"    {result['status']}: {result.get('error', 'unknown error')}")
    return result


def batch_candidates(minimum: int, maximum: int) -> list[int]:
    batches = []
    batch = minimum
    while batch <= maximum:
        batches.append(batch)
        batch *= 2
    if batches[-1] != maximum and maximum > batches[-1]:
        batches.append(maximum)
    return batches


def compute_profiles(bfloat16_supported: bool) -> list[dict]:
    profiles = [
        {"amp": True, "amp_dtype": "float16", "channels_last": True},
        {"amp": True, "amp_dtype": "float16", "channels_last": False},
        {"amp": False, "amp_dtype": "float16", "channels_last": True},
        {"amp": False, "amp_dtype": "float16", "channels_last": False},
    ]
    if bfloat16_supported:
        profiles.extend(
            [
                {"amp": True, "amp_dtype": "bfloat16", "channels_last": True},
                {"amp": True, "amp_dtype": "bfloat16", "channels_last": False},
            ]
        )
    return profiles


def available_system_memory_bytes() -> int | None:
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.available_physical)
        return None
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError):
            return None
    return None


def automatic_prefetch_factors(
    *,
    batch: int,
    workers: int,
    input_size: tuple[int, ...],
    available_memory: int | None,
    max_ram_fraction: float,
) -> list[int]:
    if workers <= 0:
        return []
    candidates = [1, 2, 4, 8]
    if available_memory is None:
        return candidates[:3]

    elements_per_sample = 1
    for dimension in input_size:
        elements_per_sample *= dimension
    batch_bytes = batch * elements_per_sample * 4
    budget = available_memory * max_ram_fraction
    maximum = max(1, int(budget // (workers * batch_bytes)))
    allowed = [factor for factor in candidates if factor <= maximum]
    return allowed or [1]


def compute_mode_key(result: dict) -> tuple:
    config = result["config"]
    return (
        config["amp"],
        config["amp_dtype"] if config["amp"] else None,
        config["channels_last"],
    )


def select_compute_finalists(results: list[dict], top_count: int) -> list[dict]:
    ranked = sorted(results, key=lambda item: item["samples_per_second"], reverse=True)
    selected = ranked[:top_count]
    selected_keys = {
        config_key(TrialConfig(**result["config"])) for result in selected
    }
    covered_modes = {compute_mode_key(result) for result in selected}

    # workers=0 can hide a compute mode that becomes much faster once loading is
    # overlapped. Always advance the best batch from every compute mode so the
    # DataLoader phase, rather than synchronous decoding, makes that decision.
    for result in ranked:
        mode = compute_mode_key(result)
        key = config_key(TrialConfig(**result["config"]))
        if mode not in covered_modes and key not in selected_keys:
            selected.append(result)
            selected_keys.add(key)
            covered_modes.add(mode)
    return selected


def successful_and_safe(result: dict, max_vram_fraction: float) -> bool:
    return result["status"] == "success" and result["vram_fraction"] <= max_vram_fraction


def config_key(config: TrialConfig) -> tuple:
    return tuple(asdict(config).values())


def append_trial(results: list[dict], seen: set[tuple], result: dict) -> None:
    key = config_key(TrialConfig(**result["config"]))
    if key not in seen:
        seen.add(key)
        results.append(result)


def format_training_command(args: argparse.Namespace, config: TrialConfig) -> str:
    command = [
        "uv run python",
        "python-scripts\\timm\\train_timm_classification.py",
        "--data",
        f'"{args.data}"',
        "--model",
        f'"{args.model}"',
        "--batch",
        str(config.batch),
        "--workers",
        str(config.workers),
        "--prefetch-factor",
        str(config.prefetch_factor or 2),
        "--persistent-workers" if config.persistent_workers else "--no-persistent-workers",
        "--pin-memory" if config.pin_memory else "--no-pin-memory",
        "--amp" if config.amp else "--no-amp",
        "--amp-dtype",
        config.amp_dtype,
        "--channels-last" if config.channels_last else "--no-channels-last",
        "--compile" if config.compile else "--no-compile",
        "--device",
        args.device,
    ]
    return " ".join(command)


def write_reports(
    args: argparse.Namespace,
    results: list[dict],
    selected: dict,
    gpu: dict,
) -> tuple[Path, Path, Path]:
    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = args.output / f"gpu-tuning-{timestamp}.json"
    csv_path = args.output / f"gpu-tuning-{timestamp}.csv"
    selected_config = TrialConfig(**selected["config"])
    selected_config_path = SELECTED_CONFIG_PATH
    selected_config_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().astimezone().isoformat(),
                "source": str(Path(__file__).resolve()),
                "config": asdict(selected_config),
                "benchmark": {
                    "samples_per_second": selected["samples_per_second"],
                    "milliseconds_per_batch": selected["milliseconds_per_batch"],
                    "peak_reserved_mb": selected["peak_reserved_mb"],
                    "vram_fraction": selected["vram_fraction"],
                    "device": str(args.device),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "gpu": gpu,
        "search": {
            "data": str(args.data.resolve()),
            "model": args.model,
            "min_batch": args.min_batch,
            "max_batch": args.max_batch,
            "workers": args.workers,
            "prefetch_factors": args.prefetch_factors or "auto",
            "max_prefetch_ram_fraction": args.max_prefetch_ram_fraction,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "final_measure_steps": args.final_measure_steps,
            "final_candidates": args.final_candidates,
            "max_vram_fraction": args.max_vram_fraction,
            "test_compile": args.test_compile,
        },
        "selected": selected,
        "training_command": format_training_command(args, selected_config),
        "trials": results,
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    fieldnames = [
        "status",
        "batch",
        "workers",
        "prefetch_factor",
        "persistent_workers",
        "pin_memory",
        "amp",
        "amp_dtype",
        "channels_last",
        "compile",
        "benchmark_stage",
        "samples_per_second",
        "milliseconds_per_batch",
        "peak_allocated_mb",
        "peak_reserved_mb",
        "vram_fraction",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {**result["config"], **{key: value for key, value in result.items() if key != "config"}}
            writer.writerow({key: row.get(key) for key in fieldnames})
    return json_path, csv_path, selected_config_path


def main() -> int:
    args = build_parser().parse_args()
    if args._trial:
        return run_training_trial(args)
    if args.min_batch > args.max_batch:
        raise ValueError("--min-batch cannot exceed --max-batch")
    if args.prefetch_factors is not None and (
        not args.prefetch_factors or args.prefetch_factors[0] < 1
    ):
        raise ValueError("--prefetch-factors must contain positive integers")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "GPU tuning requires CUDA. Run this script in the environment and machine used for training."
        )
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    gpu = {
        "name": properties.name,
        "device": str(device),
        "total_vram_mb": properties.total_memory / (1024**2),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "timm_version": getattr(timm, "__version__", "unknown"),
        "bfloat16_supported": torch.cuda.is_bf16_supported(),
    }
    print(
        f"GPU: {gpu['name']} ({gpu['total_vram_mb']:.0f} MiB); "
        f"CUDA {gpu['cuda_version']}; PyTorch {gpu['torch_version']}"
    )
    print(
        f"Safe VRAM limit: {args.max_vram_fraction:.0%}; "
        f"warmup={args.warmup_steps}, screening steps={args.measure_steps}, "
        f"final steps={args.final_measure_steps}"
    )

    results: list[dict] = []
    seen: set[tuple] = set()
    compute_results: list[dict] = []
    print("\nPhase 1: compute mode and batch-size search")
    for profile in compute_profiles(gpu["bfloat16_supported"]):
        for batch in batch_candidates(args.min_batch, args.max_batch):
            config = TrialConfig(
                batch=batch,
                workers=0,
                prefetch_factor=None,
                persistent_workers=False,
                pin_memory=True,
                compile=False,
                **profile,
            )
            result = run_isolated_trial(args, config)
            append_trial(results, seen, result)
            if result["status"] == "success":
                if successful_and_safe(result, args.max_vram_fraction):
                    compute_results.append(result)
                else:
                    print("    rejected: exceeds configured VRAM safety limit")
                    break
            else:
                break

    if not compute_results:
        raise RuntimeError(
            "No safe GPU configuration completed. Lower --min-batch or increase "
            "--max-vram-fraction after checking other GPU workloads."
        )

    input_size = tuple(compute_results[0]["input_size"])
    available_memory = available_system_memory_bytes()
    print(f"Model input size: {input_size}")
    if available_memory is not None:
        print(f"Available host RAM: {available_memory / (1024**3):.1f} GiB")

    finalists = select_compute_finalists(compute_results, args.top_compute_configs)
    print(
        f"Advancing {len(finalists)} candidates: the global top "
        f"{args.top_compute_configs} plus the best batch from each compute mode."
    )
    print("\nPhase 2: DataLoader search on the strongest compute configurations")
    loader_results: list[dict] = []
    for finalist in finalists:
        base = TrialConfig(**finalist["config"])
        previous_worker_throughput: float | None = None
        low_gain_count = 0
        for workers in args.workers:
            if workers == 0:
                loader_variants = [(None, False, False), (None, False, True)]
            else:
                prefetch_factors = args.prefetch_factors or automatic_prefetch_factors(
                    batch=base.batch,
                    workers=workers,
                    input_size=input_size,
                    available_memory=available_memory,
                    max_ram_fraction=args.max_prefetch_ram_fraction,
                )
                loader_variants = [
                    (prefetch, True, True) for prefetch in prefetch_factors
                ]
            worker_results = []
            for prefetch, persistent, pin_memory in loader_variants:
                config = TrialConfig(
                    batch=base.batch,
                    workers=workers,
                    prefetch_factor=prefetch,
                    persistent_workers=persistent,
                    pin_memory=pin_memory,
                    amp=base.amp,
                    amp_dtype=base.amp_dtype,
                    channels_last=base.channels_last,
                    compile=False,
                )
                key = config_key(config)
                if key in seen:
                    existing = next(
                        item for item in results if config_key(TrialConfig(**item["config"])) == key
                    )
                    result = existing
                else:
                    result = run_isolated_trial(args, config)
                    append_trial(results, seen, result)
                if successful_and_safe(result, args.max_vram_fraction):
                    loader_results.append(result)
                    worker_results.append(result)

            if workers > 0 and worker_results:
                worker_throughput = max(
                    result["samples_per_second"] for result in worker_results
                )
                if previous_worker_throughput is not None:
                    gain = worker_throughput / previous_worker_throughput - 1
                    low_gain_count = low_gain_count + 1 if gain < 0.05 else 0
                previous_worker_throughput = worker_throughput
                if low_gain_count >= 2:
                    print(
                        "    stopping worker expansion: throughput improved by "
                        "less than 5% at two consecutive worker counts"
                    )
                    break

    candidates = loader_results or compute_results
    selected = max(candidates, key=lambda item: item["samples_per_second"])

    if args.test_compile:
        print("\nPhase 3: torch.compile check on the current winner")
        compile_config = TrialConfig(**{**selected["config"], "compile": True})
        compile_result = run_isolated_trial(args, compile_config)
        append_trial(results, seen, compile_result)
        if successful_and_safe(compile_result, args.max_vram_fraction):
            candidates.append(compile_result)

    print("\nFinal validation: sustained throughput on the short-run winners")
    validation_configs: list[TrialConfig] = []
    validation_keys: set[tuple] = set()
    for candidate in sorted(
        candidates,
        key=lambda item: item["samples_per_second"],
        reverse=True,
    ):
        config = TrialConfig(**candidate["config"])
        key = config_key(config)
        if key not in validation_keys:
            validation_configs.append(config)
            validation_keys.add(key)
        if len(validation_configs) >= args.final_candidates:
            break

    validation_results = []
    for config in validation_configs:
        queue_depth = (
            config.workers * (config.prefetch_factor or 0)
            if config.workers > 0
            else 0
        )
        final_measure_steps = max(args.final_measure_steps, 2 * queue_depth)
        result = run_isolated_trial(
            args,
            config,
            measure_steps=final_measure_steps,
            benchmark_stage="final_validation",
        )
        results.append(result)
        if successful_and_safe(result, args.max_vram_fraction):
            validation_results.append(result)
    if validation_results:
        selected = max(
            validation_results,
            key=lambda item: item["samples_per_second"],
        )

    selected_config = TrialConfig(**selected["config"])
    json_path, csv_path, selected_config_path = write_reports(args, results, selected, gpu)
    print("\nSelected configuration")
    print(json.dumps(selected_config.__dict__, indent=2))
    print(
        f"{selected['samples_per_second']:.1f} samples/s; "
        f"peak reserved={selected['peak_reserved_mb']:.0f} MiB "
        f"({selected['vram_fraction']:.1%})"
    )
    print("\nTraining command")
    print(format_training_command(args, selected_config))
    print(f"\nReports:\n  {json_path}\n  {csv_path}\nSelected config:\n  {selected_config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
