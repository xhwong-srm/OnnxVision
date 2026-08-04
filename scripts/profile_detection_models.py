"""Profile inference parameters and theoretical FLOPs for local detectors.

The reported FLOPs are a graph-level estimate for one batch-1 image.  They
include the neural-network forward pass exposed by each framework's PyTorch
model, but not image decoding, host/device copies, or framework-specific
post-processing such as NMS.  Use ``--resolution`` to compare models at the
same input size; otherwise each adapter uses the training script's default or
the checkpoint's saved input size.

Model specifications are passed as ``family-variant::checkpoint``.  The
checkpoint part is optional when only the architecture is needed.  Examples::

    uv run python python-scripts/profile_detection_models.py
    uv run python python-scripts/profile_detection_models.py \
        --model timm-v2::python-scripts/timm/runs/.../best.pt \
        --model rtmdet-t::python-scripts/libreyolo/runs/.../best.pt \
        --model yolo26-n::yolo26n.pt --resolution 640

The default model list profiles the local final checkpoints when available
and uses the matching two-class architecture for families without a saved
checkpoint yet.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
TIMM_LEGACY_SCRIPT = ROOT / "python-scripts" / "timm" / "train_timm_detection.py"
TIMM_V2_SCRIPT = ROOT / "python-scripts" / "timm" / "train_timm_detection_v2.py"

DEFAULT_SPECS = (
    "timm-v2",
    "timm-retinanet::python-scripts/timm/runs/mobilenetv4_small_retinanet_v1/best.pt",
    "rtmdet-t::python-scripts/libreyolo/runs/rtmdet-t-v1/best.pt",
    "picodet-s::python-scripts/libreyolo/runs/picodet-s-v1/best.pt",
    "yolov9-t",
    "yolo26-n",
    "rfdetr-nano",
)


@dataclass
class LoadedModel:
    """A framework-independent model ready for a tensor forward pass."""

    name: str
    model: nn.Module
    resolution: int
    source: str
    metadata: dict[str, Any]


def load_module(path: Path, module_name: str) -> ModuleType:
    """Import one of the existing script modules without changing the scripts."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import local script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def checkpoint_dict(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a dictionary checkpoint: {path}")
    return value


def checkpoint_classes(checkpoint: dict[str, Any], requested: int) -> list[str]:
    classes = checkpoint.get("classes") or checkpoint.get("names")
    if isinstance(classes, dict):
        classes = [classes[key] for key in sorted(classes, key=lambda item: int(item))]
    if isinstance(classes, (list, tuple)) and classes:
        return [str(value) for value in classes]
    return [f"class_{index}" for index in range(requested)]


def int_from_config(config: dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if value is None:
        return default
    return int(value)


def load_timm_detector(
    variant: str,
    checkpoint: Path | None,
    requested_classes: int,
    resolution_override: int | None,
    v2: bool,
) -> LoadedModel:
    script = TIMM_V2_SCRIPT if v2 else TIMM_LEGACY_SCRIPT
    module_name = "profile_timm_v2" if v2 else "profile_timm_legacy"
    module = load_module(script, module_name)
    checkpoint_value: dict[str, Any] | None = None
    if checkpoint is not None:
        checkpoint_value = checkpoint_dict(checkpoint)

    if checkpoint_value is not None:
        model_config = checkpoint_value.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError(f"Checkpoint is missing model_config: {checkpoint}")
        data_config = checkpoint_value.get("data_config") or {}
        classes = checkpoint_classes(checkpoint_value, requested_classes)
        model_name = str(
            checkpoint_value.get("model_name")
            or model_config.get("model_name")
            or variant
        )
        resolution = int_from_config(
            model_config,
            "imgsz",
            resolution_override or 384,
        )
        if resolution_override is not None:
            resolution = resolution_override
        if v2:
            model, _, _ = module.build_detector(
                model_name=model_name,
                num_classes=len(classes),
                fpn_channels=int_from_config(model_config, "fpn_channels", 256),
                num_queries=int_from_config(model_config, "num_queries", 100),
                decoder_layers=int_from_config(model_config, "decoder_layers", 3),
                attention_heads=int_from_config(model_config, "attention_heads", 8),
                pretrained=False,
                image_mean=data_config.get("mean"),
                image_std=data_config.get("std"),
                bbox_loss_weight=float(model_config.get("bbox_loss_weight", 5.0)),
                giou_loss_weight=float(model_config.get("giou_loss_weight", 2.0)),
                no_object_weight=float(model_config.get("no_object_weight", 0.1)),
            )
        else:
            model, _, _ = module.build_detector(
                model_name=model_name,
                num_classes=len(classes),
                imgsz=resolution,
                fpn_channels=int_from_config(model_config, "fpn_channels", 256),
                anchor_sizes=tuple(int(value) for value in model_config.get("anchor_sizes", (32, 64, 128, 256, 512))),
                anchor_ratios=tuple(float(value) for value in model_config.get("anchor_ratios", (0.5, 1.0, 2.0))),
                pretrained=False,
                image_mean=data_config.get("mean"),
                image_std=data_config.get("std"),
            )
        model.load_state_dict(checkpoint_value["model_state_dict"], strict=True)
        source = str(checkpoint)
        metadata = {
            "checkpoint_epoch": checkpoint_value.get("epoch"),
            "classes": classes,
            "model_name": model_name,
            "architecture": "nms_free_query" if v2 else "retinanet",
        }
    else:
        model_name = variant or ("mobilenetv4_conv_small" if not v2 else "mobilenetv4_conv_small")
        classes = [f"class_{index}" for index in range(requested_classes)]
        resolution = resolution_override or 384
        if v2:
            model, _, _ = module.build_detector(
                model_name=model_name,
                num_classes=requested_classes,
                fpn_channels=256,
                num_queries=100,
                decoder_layers=3,
                attention_heads=8,
                pretrained=False,
            )
        else:
            model, _, _ = module.build_detector(
                model_name=model_name,
                num_classes=requested_classes,
                imgsz=resolution,
                pretrained=False,
            )
        source = "architecture only (random weights; FLOPs and parameter count are unchanged)"
        metadata = {
            "classes": classes,
            "model_name": model_name,
            "architecture": "nms_free_query" if v2 else "retinanet",
        }
    return LoadedModel(
        name="timm-v2" if v2 else "timm-retinanet",
        model=model,
        resolution=resolution,
        source=source,
        metadata=metadata,
    )


def load_libreyolo_detector(
    family: str,
    variant: str,
    checkpoint: Path | None,
    requested_classes: int,
    resolution_override: int | None,
    device: torch.device,
) -> LoadedModel:
    from libreyolo import LibrePICODET, LibreRTMDet, LibreYOLO, LibreYOLO9

    if checkpoint is not None:
        wrapper = LibreYOLO(str(checkpoint), device=str(device), task="detect")
        source = str(checkpoint)
    elif family == "rtmdet":
        wrapper = LibreRTMDet(
            model_path=None,
            size=variant or "t",
            nb_classes=requested_classes,
            device=str(device),
            task="detect",
        )
        source = "architecture only (random weights; FLOPs and parameter count are unchanged)"
    elif family == "picodet":
        wrapper = LibrePICODET(
            model_path=None,
            size=variant or "s",
            nb_classes=requested_classes,
            device=str(device),
            task="detect",
        )
        source = "architecture only (random weights; FLOPs and parameter count are unchanged)"
    else:
        wrapper = LibreYOLO9(
            model_path=None,
            size=variant or "t",
            nb_classes=requested_classes,
            device=str(device),
            task="detect",
        )
        source = "architecture only (random weights; FLOPs and parameter count are unchanged)"

    model = wrapper.model
    resolution = resolution_override or int(getattr(wrapper, "input_size", 640))
    return LoadedModel(
        name=f"{family}-{variant or getattr(wrapper, 'size', '')}".rstrip("-"),
        model=model,
        resolution=resolution,
        source=source,
        metadata={
            "model_family": family,
            "variant": variant or getattr(wrapper, "size", None),
            "classes": requested_classes,
        },
    )


def load_yolo26_detector(
    variant: str,
    checkpoint: Path | None,
    requested_classes: int,
    resolution_override: int | None,
) -> LoadedModel:
    from ultralytics import YOLO

    if checkpoint is None:
        default_name = f"yolo26{variant or 'n'}.pt"
        default_path = ROOT / default_name
        source_value = default_path if default_path.is_file() else Path(default_name)
    else:
        source_value = checkpoint
    yolo = YOLO(str(source_value))
    model = yolo.model
    source = str(source_value)
    metadata: dict[str, Any] = {"classes": int(getattr(model.model[-1], "nc", requested_classes))}

    # The repository's yolo26n.pt is an 80-class pretrained detector.  Build
    # the same architecture with the local two-class head when no final
    # project checkpoint was supplied, so the estimate matches training.
    if checkpoint is None and requested_classes != int(getattr(model.model[-1], "nc", requested_classes)):
        from ultralytics.nn.tasks import DetectionModel

        config = copy.deepcopy(model.yaml)
        model = DetectionModel(config, ch=3, nc=requested_classes, verbose=False)
        source = f"{source_value} architecture, rebuilt for {requested_classes} classes"
        metadata["classes"] = requested_classes

    resolution = resolution_override or int(getattr(model, "args", {}).get("imgsz", 640))
    return LoadedModel(
        name=f"yolo26-{variant or 'n'}",
        model=model,
        resolution=resolution,
        source=source,
        metadata=metadata,
    )


def load_rfdetr_detector(
    variant: str,
    checkpoint: Path | None,
    requested_classes: int,
    resolution_override: int | None,
) -> LoadedModel:
    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

    classes = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }
    model_class = classes.get(variant or "nano")
    if model_class is None:
        raise ValueError(f"Unsupported RF-DETR variant: {variant!r}")
    if checkpoint is not None:
        detector = model_class.from_checkpoint(str(checkpoint))
        source = str(checkpoint)
    else:
        detector = model_class(num_classes=requested_classes, resolution=resolution_override or 384)
        source = "pretrained architecture (RF-DETR starter weights; no local final checkpoint)"
    model = detector.model.model
    resolution = resolution_override or int(detector.model.resolution)
    return LoadedModel(
        name=f"rfdetr-{variant or 'nano'}",
        model=model,
        resolution=resolution,
        source=source,
        metadata={
            "classes": requested_classes,
            "variant": variant or "nano",
        },
    )


def parse_family(value: str) -> tuple[str, str]:
    family = value.strip().lower()
    aliases = {
        "timm-v2": ("timm-v2", "mobilenetv4_conv_small"),
        "timm": ("timm-retinanet", "mobilenetv4_conv_small"),
        "timm-retinanet": ("timm-retinanet", "mobilenetv4_conv_small"),
        "timm-legacy": ("timm-retinanet", "mobilenetv4_conv_small"),
    }
    if family in aliases:
        return aliases[family]
    for prefix, canonical in (("rtmdet", "rtmdet"), ("picodet", "picodet"), ("yolov9", "yolov9"), ("yolo26", "yolo26"), ("rfdetr", "rfdetr")):
        if family == prefix:
            defaults = {"rtmdet": "t", "picodet": "s", "yolov9": "t", "yolo26": "n", "rfdetr": "nano"}
            return canonical, defaults[canonical]
        if family.startswith(prefix + "-"):
            return canonical, family[len(prefix) + 1 :]
    raise ValueError(
        f"Unknown model family {value!r}; use timm-v2, timm-retinanet, rtmdet-t, "
        "picodet-s, yolov9-t, yolo26-n, or rfdetr-nano"
    )


def parse_model_spec(value: str) -> tuple[str, str, Path | None]:
    family_value, separator, source_value = value.partition("::")
    family, variant = parse_family(family_value)
    checkpoint = Path(source_value).expanduser() if separator and source_value else None
    if checkpoint is not None:
        if not checkpoint.is_absolute():
            checkpoint = (Path.cwd() / checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    return family, variant, checkpoint


def load_spec(
    value: str,
    requested_classes: int,
    resolution_override: int | None,
    device: torch.device,
) -> LoadedModel:
    family, variant, checkpoint = parse_model_spec(value)
    if family == "timm-v2":
        return load_timm_detector(variant, checkpoint, requested_classes, resolution_override, v2=True)
    if family == "timm-retinanet":
        return load_timm_detector(variant, checkpoint, requested_classes, resolution_override, v2=False)
    if family in {"rtmdet", "picodet", "yolov9"}:
        return load_libreyolo_detector(
            family, variant, checkpoint, requested_classes, resolution_override, device
        )
    if family == "yolo26":
        return load_yolo26_detector(variant, checkpoint, requested_classes, resolution_override)
    if family == "rfdetr":
        return load_rfdetr_detector(variant, checkpoint, requested_classes, resolution_override)
    raise AssertionError(f"Unhandled model family: {family}")


def profile_model(loaded: LoadedModel, device: torch.device) -> dict[str, Any]:
    from calflops import calculate_flops

    model = loaded.model.to(device).eval()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    input_shape = (1, 3, loaded.resolution, loaded.resolution)
    with torch.no_grad():
        flops, macs, _ = calculate_flops(
            model=model,
            input_shape=input_shape,
            print_results=False,
            print_detailed=False,
            output_as_string=False,
        )
    flops = int(flops)
    macs = int(macs)
    pixels = loaded.resolution * loaded.resolution
    return {
        "name": loaded.name,
        "status": "ok",
        "source": loaded.source,
        "input_shape": list(input_shape),
        "parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "parameters_m": total_parameters / 1_000_000,
        "flops": flops,
        "flops_g": flops / 1_000_000_000,
        "macs": macs,
        "macs_g": macs / 1_000_000_000,
        "flops_per_pixel": flops / pixels,
        "parameter_storage_mb_fp32": total_parameters * 4 / (1024 * 1024),
        "metadata": loaded.metadata,
    }


def print_results(results: list[dict[str, Any]]) -> None:
    print("\nInference compute estimate (batch=1; graph-level PyTorch forward)\n")
    print(f"{'model':<18} {'status':<8} {'params':>10} {'GFLOPs':>10} {'GMACs':>10} {'input':>11}")
    print("-" * 73)
    for result in results:
        if result["status"] == "ok":
            shape = result["input_shape"]
            print(
                f"{result['name']:<18} {'ok':<8} "
                f"{result['parameters_m']:>9.2f}M "
                f"{result['flops_g']:>9.2f} "
                f"{result['macs_g']:>9.2f} "
                f"{shape[2]}x{shape[3]:<6}"
            )
        else:
            print(f"{result['name']:<18} {'error':<8} {result['error']}")
    print()
    for result in results:
        if result["status"] == "ok":
            print(
                f"{result['name']}: {result['flops_g']:.3f} GFLOPs, "
                f"{result['parameters_m']:.3f} M parameters, "
                f"{result['flops_per_pixel']:.1f} FLOPs/pixel; source={result['source']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help="Repeatable family-variant::checkpoint specification; defaults to all local families",
    )
    parser.add_argument("--classes", type=int, default=2, help="Class count for architecture-only profiles")
    parser.add_argument("--resolution", type=int, help="Override every model's square input resolution")
    parser.add_argument(
        "--device",
        default="cpu",
        help='Torch device for model construction and profiling (default: "cpu")',
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path; parent directories are created",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.classes < 1:
        raise ValueError("--classes must be positive")
    if args.resolution is not None and (args.resolution < 32 or args.resolution % 32):
        raise ValueError("--resolution must be at least 32 and divisible by 32")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    results: list[dict[str, Any]] = []
    for spec in args.models or DEFAULT_SPECS:
        try:
            loaded = load_spec(spec, args.classes, args.resolution, device)
            result = profile_model(loaded, device)
        except Exception as error:  # Keep one unsupported operator from hiding the other models.
            result = {
                "name": spec.split("::", 1)[0],
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
        results.append(result)

    print_results(results)
    if args.output is not None:
        output = args.output.expanduser()
        if not output.is_absolute():
            output = (Path.cwd() / output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"JSON report: {output}")
    return 0 if all(result["status"] == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
