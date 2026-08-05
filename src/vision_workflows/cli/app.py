from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from ..api import (
    ConvertDatasetRequest,
    DatasetService,
    ExportRequest,
    ExportService,
    MergeDatasetRequest,
    SplitDatasetRequest,
    TestRequest,
    TestService,
    TrainRequest,
    TrainService,
    ValidateDatasetRequest,
    ValidateRequest,
    ValidationService,
)
from ..backends.registry import descriptors
from ..domain.datasets import DatasetFormat, MaterializationMode, SplitPolicy
from ..domain.models import ModelRef


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _format(value: str) -> DatasetFormat:
    try:
        return DatasetFormat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"unsupported dataset format: {value}") from error


def _model(value: str) -> ModelRef:
    try:
        return ModelRef.parse(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _common_dataset(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", type=_format)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--materialization", choices=[item.value for item in MaterializationMode], default="hardlink")


def _split_policy(args: argparse.Namespace) -> SplitPolicy:
    return SplitPolicy(args.train, args.val, args.test, args.seed, args.grouping, not args.no_stratify)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vision-workflows", description="Dataset and model workflows")
    commands = parser.add_subparsers(dest="command", required=True)

    dataset = commands.add_parser("dataset", help="dataset management")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    inspect = dataset_commands.add_parser("inspect")
    inspect.add_argument("source", type=_path)
    inspect.add_argument("--format", type=_format)
    validate = dataset_commands.add_parser("validate")
    validate.add_argument("source", type=_path)
    validate.add_argument("--format", type=_format)
    validate.add_argument("--require-train-val", action="store_true")
    convert = dataset_commands.add_parser("convert")
    convert.add_argument("source", type=_path)
    convert.add_argument("output", type=_path)
    convert.add_argument("--output-format", type=_format, required=True)
    convert.add_argument("--input-format", type=_format)
    _common_dataset(convert)
    merge = dataset_commands.add_parser("merge")
    merge.add_argument("sources", nargs="+", type=_path)
    merge.add_argument("--output", type=_path, required=True)
    merge.add_argument("--output-format", type=_format)
    merge.add_argument("--split", action="store_true")
    merge.add_argument("--train", type=float, default=0.7)
    merge.add_argument("--val", type=float, default=0.2)
    merge.add_argument("--test", type=float, default=0.1)
    merge.add_argument("--seed", type=int, default=42)
    merge.add_argument("--grouping", default="sample")
    merge.add_argument("--no-stratify", action="store_true")
    merge.add_argument("--balance", choices=("none", "undersample", "oversample"), default="none")
    _common_dataset(merge)
    split = dataset_commands.add_parser("split")
    split.add_argument("source", type=_path)
    split.add_argument("output", type=_path)
    split.add_argument("--output-format", type=_format)
    split.add_argument("--train", type=float, default=0.7)
    split.add_argument("--val", type=float, default=0.2)
    split.add_argument("--test", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--grouping", default="sample")
    split.add_argument("--no-stratify", action="store_true")
    _common_dataset(split)

    model = commands.add_parser("model", help="model registry")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    model_commands.add_parser("list")

    train = commands.add_parser("train")
    train.add_argument("--model", type=_model, required=True)
    train.add_argument("--data", type=_path, required=True)
    train.add_argument("--output", type=_path, required=True)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--image-size", type=int, default=640)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--workers", type=int, default=-1)
    train.add_argument("--patience", type=int, default=20)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="auto")
    train.add_argument("--weights", type=_path)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--allow-experimental", action="store_true", help="allow backends with gated experimental training")
    train.add_argument("--overwrite", action="store_true", help="delete the requested run directory before training")

    export = commands.add_parser("export")
    export.add_argument("--model", type=_model, required=True)
    export.add_argument("--checkpoint", type=_path, required=True)
    export.add_argument("--output", type=_path, required=True)
    export.add_argument("--data", type=_path)
    export.add_argument("--image-size", type=int, default=640)
    export.add_argument("--opset", type=int, default=18)
    export.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True)
    export.add_argument("--device", default="auto")
    export.add_argument("--embedded-preprocessing", action="store_true")

    for name in ("validate", "test"):
        command = commands.add_parser(name)
        command.add_argument("--model", type=_model, required=True)
        command.add_argument("--target", type=_path, required=True)
        command.add_argument("--data", type=_path, required=name == "test")
        command.add_argument("--split", default="test" if name == "test" else "val")
        command.add_argument("--device", default="cpu" if name == "validate" else "auto")
    return parser


def _print(value) -> None:
    print(json.dumps(value, indent=2, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    args = build_parser().parse_args(argv)
    service = DatasetService()
    materialization = MaterializationMode(args.materialization) if hasattr(args, "materialization") else MaterializationMode.HARDLINK
    if args.command == "model":
        for item in descriptors():
            _print({"backend": item.backend, "family": item.family, "task": item.task, "variants": item.variants, "capabilities": sorted(capability.value for capability in item.capabilities), "description": item.description})
        return 0
    if args.command == "dataset":
        if args.dataset_command == "inspect":
            _print(asdict(service.inspect(args.source, args.format)))
        elif args.dataset_command == "validate":
            result = service.validate(ValidateDatasetRequest(args.source, args.format, args.require_train_val))
            _print(asdict(result))
            return 0 if result.valid else 1
        elif args.dataset_command == "convert":
            _print(asdict(service.convert(ConvertDatasetRequest(args.source, args.output, args.output_format, args.input_format, materialization, args.overwrite))))
        elif args.dataset_command == "split":
            result = service.split(SplitDatasetRequest(args.source, args.output, _split_policy(args), args.format, args.output_format, materialization, args.overwrite))
            _print(asdict(result))
        elif args.dataset_command == "merge":
            policy = _split_policy(args) if args.split else None
            from ..domain.datasets import BalanceMode
            result = service.merge(MergeDatasetRequest(tuple(args.sources), args.output, args.output_format, policy, materialization, args.overwrite, BalanceMode(args.balance)))
            _print(asdict(result))
        return 0
    if args.command == "train":
        options = {"allow_experimental": True} if args.allow_experimental else {}
        result = TrainService().run(TrainRequest(args.model, args.data, args.output, args.epochs, args.batch, args.image_size, args.learning_rate, args.workers, args.patience, args.seed, args.device, args.weights, args.resume, args.pretrained, args.deterministic, args.overwrite, options))
        _print(asdict(result))
        return 0
    if args.command == "export":
        result = ExportService().run(ExportRequest(args.model, args.checkpoint, args.output, args.data, args.image_size, args.opset, args.simplify, args.device, args.embedded_preprocessing))
        _print(asdict(result))
        return 0
    if args.command == "validate":
        result = ValidationService().run(ValidateRequest(args.model, args.target, args.data, args.split, args.device))
        _print(asdict(result))
        return 0
    result = TestService().run(TestRequest(args.model, args.target, args.data, args.split, args.device))
    _print(asdict(result))
    return 0
