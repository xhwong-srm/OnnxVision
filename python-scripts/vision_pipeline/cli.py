"""Unified command-line frontend for training and ONNX export workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .models import ExportRequest, TrainRequest, as_args
from .registry import descriptors, export, train


def _common_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("timm", "ultralytics", "libreyolo"), required=True)
    parser.add_argument("--task", choices=("classification", "detection"), required=True)
    parser.add_argument("--model", required=True, help="Model family or timm model name")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    train_parser = subcommands.add_parser("train", help="Train a registered model workflow")
    _common_model_arguments(train_parser)
    train_parser.add_argument("--data", required=True, type=Path)
    train_parser.add_argument("--output", type=Path)
    train_parser.add_argument("--weights", type=Path)
    train_parser.add_argument("--imgsz", type=int)
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--batch", type=int)
    train_parser.add_argument("--lr", type=float)
    train_parser.add_argument("--workers", type=int)
    train_parser.add_argument("--patience", type=int)
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--device")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    train_parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None)
    train_parser.add_argument("--run-test", action=argparse.BooleanOptionalAction, default=None)
    train_parser.add_argument("backend_args", nargs=argparse.REMAINDER, help="Backend-specific flags after --")

    export_parser = subcommands.add_parser("export", help="Export a registered model workflow")
    _common_model_arguments(export_parser)
    export_parser.add_argument("--checkpoint", required=True, type=Path)
    export_parser.add_argument("--output", type=Path)
    export_parser.add_argument("--imgsz", type=int)
    export_parser.add_argument("--opset", type=int)
    export_parser.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=None)
    export_parser.add_argument("--embedded-preprocessing", action=argparse.BooleanOptionalAction, default=None)
    export_parser.add_argument("--device")
    export_parser.add_argument("--data", type=Path)
    export_parser.add_argument("--validation-split")
    export_parser.add_argument("--validation-limit", type=int)
    export_parser.add_argument("--validation-report", type=Path)
    export_parser.add_argument("backend_args", nargs=argparse.REMAINDER, help="Backend-specific flags after --")

    list_parser = subcommands.add_parser("list-models", help="List registered workflows")
    list_parser.set_defaults(list_models=True)
    return parser


def _strip_separator(values: Sequence[str]) -> tuple[str, ...]:
    return as_args(values[1:] if values and values[0] == "--" else values)


def _run_train(args: argparse.Namespace) -> int:
    if args.backend == "timm" and args.task == "classification":
        unsupported = {
            "--imgsz": args.imgsz,
            "--weights": args.weights,
            "--resume": args.resume if args.resume else None,
            "--pretrained": args.pretrained,
            "--run-test": args.run_test,
        }
        selected = [name for name, value in unsupported.items() if value is not None]
        if selected:
            raise ValueError(
                "timm classification does not expose these unified training options: "
                + ", ".join(selected)
            )
    if args.backend == "timm" and args.task == "detection":
        if args.weights is not None or args.resume:
            raise ValueError("timm detection does not expose --weights or --resume")
    result = train(TrainRequest(
        backend=args.backend,
        task=args.task,
        model=args.model,
        data=args.data,
        output=args.output,
        weights=args.weights,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
        deterministic=args.deterministic,
        pretrained=args.pretrained,
        run_test=args.run_test,
        backend_args=_strip_separator(args.backend_args),
    ))
    return result.returncode


def _run_export(args: argparse.Namespace) -> int:
    if args.backend == "timm" and args.task == "classification" and args.validation_split is not None:
        raise ValueError("timm classification export uses --dataset validation rather than --validation-split")
    if args.backend == "timm" and args.task == "detection" and (
        args.validation_split is not None or args.validation_limit is not None or args.validation_report is not None
    ):
        raise ValueError("timm detection export uses --data and --max-test-images for validation")
    result = export(ExportRequest(
        backend=args.backend,
        task=args.task,
        model=args.model,
        checkpoint=args.checkpoint,
        output=args.output,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        embedded_preprocessing=args.embedded_preprocessing,
        device=args.device,
        data=args.data,
        validation_split=args.validation_split,
        validation_limit=args.validation_limit,
        validation_report=args.validation_report,
        backend_args=_strip_separator(args.backend_args),
    ))
    return result.returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "list_models", False):
        for item in descriptors():
            print(f"{item.backend}/{item.task}/{item.model_family}: {item.description}")
        return 0
    try:
        if args.command == "train":
            return _run_train(args)
        if args.command == "export":
            return _run_export(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    raise AssertionError(f"Unhandled command: {args.command}")
