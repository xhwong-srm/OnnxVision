from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

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
from ..backends.registry import frameworks, models_for, plugin_for
from ..domain.datasets import BalanceMode, DatasetFormat, MaterializationMode, SplitPolicy, TaskKind
from ..domain.models import ModelSelection, Operation, ParameterSchema


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _format(value: str) -> DatasetFormat:
    try:
        return DatasetFormat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"unsupported dataset format: {value}") from error


def _common_dataset(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", type=_format)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--materialization", choices=[item.value for item in MaterializationMode], default="hardlink")


def _split_policy(args: argparse.Namespace) -> SplitPolicy:
    return SplitPolicy(args.train, args.val, args.test, args.seed, args.grouping, not args.no_stratify)


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", choices=[item.value for item in TaskKind], required=True)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--model", required=True)


def _selection(args: argparse.Namespace) -> ModelSelection:
    return ModelSelection(TaskKind(args.task), args.framework, args.model)


def _selected_schema(argv: Sequence[str]) -> tuple[Operation, ModelSelection, ParameterSchema] | None:
    if not argv or argv[0] not in {item.value for item in Operation}:
        return None
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("command")
    probe.add_argument("--task")
    probe.add_argument("--framework")
    probe.add_argument("--model")
    values, _ = probe.parse_known_args(argv)
    if not all((values.task, values.framework, values.model)):
        return None
    operation = Operation(values.command)
    selection = ModelSelection(TaskKind(values.task), values.framework, values.model)
    plugin = plugin_for(selection)
    model = plugin.catalog.resolve(selection.model)
    return operation, selection, plugin.handlers[operation].schema(model)


def _add_parameters(parser: argparse.ArgumentParser, schema: ParameterSchema) -> None:
    groups: dict[str, argparse._ArgumentGroup] = {}
    for spec in schema.parameters:
        group = groups.setdefault(spec.origin.value, parser.add_argument_group(f"{spec.origin.value} parameters"))
        kwargs: dict[str, Any] = {"dest": spec.name, "help": spec.help, "default": argparse.SUPPRESS}
        if spec.choices:
            kwargs["choices"] = spec.choices
        if spec.value_type is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = spec.value_type
        if spec.required:
            kwargs["required"] = True
        group.add_argument(spec.cli_flag, **kwargs)


def build_parser(argv: Sequence[str] | None = None) -> argparse.ArgumentParser:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    selected = _selected_schema(effective_argv)
    parser = argparse.ArgumentParser(prog="vision-workflows", description="Dataset and model workflows")
    commands = parser.add_subparsers(dest="command", required=True)

    dataset = commands.add_parser("dataset", help="dataset management")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    inspect = dataset_commands.add_parser("inspect")
    inspect.add_argument("source", type=_path)
    inspect.add_argument("--format", type=_format)
    validate_dataset = dataset_commands.add_parser("validate")
    validate_dataset.add_argument("source", type=_path)
    validate_dataset.add_argument("--format", type=_format)
    validate_dataset.add_argument("--require-train-val", action="store_true")
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
    merge.add_argument("--balance", choices=[item.value for item in BalanceMode], default="none")
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

    framework = commands.add_parser("framework", help="framework providers")
    framework.add_subparsers(dest="framework_command", required=True).add_parser("list")
    model = commands.add_parser("model", help="model catalogs")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    model_list = model_commands.add_parser("list")
    model_list.add_argument("--task", choices=[item.value for item in TaskKind], required=True)
    model_list.add_argument("--framework", required=True)
    model_list.add_argument("--pattern")
    describe = model_commands.add_parser("describe")
    _add_selection(describe)
    describe.add_argument("--operation", choices=[item.value for item in Operation], required=True)

    workflow_parsers: dict[Operation, argparse.ArgumentParser] = {}
    train = workflow_parsers[Operation.TRAIN] = commands.add_parser("train")
    _add_selection(train)
    train.add_argument("--data", type=_path, required=True)
    train.add_argument("--output", type=_path, required=True)
    train.add_argument("--weights", type=_path)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--overwrite", action="store_true")
    export = workflow_parsers[Operation.EXPORT] = commands.add_parser("export")
    _add_selection(export)
    export.add_argument("--checkpoint", type=_path, required=True)
    export.add_argument("--output", type=_path, required=True)
    export.add_argument("--data", type=_path)
    for operation in (Operation.VALIDATE, Operation.TEST):
        command = workflow_parsers[operation] = commands.add_parser(operation.value)
        _add_selection(command)
        command.add_argument("--target", type=_path, required=True)
        command.add_argument("--data", type=_path, required=operation is Operation.TEST)
        command.add_argument("--split", default="test" if operation is Operation.TEST else "val")
    if selected is not None:
        operation, _, schema = selected
        _add_parameters(workflow_parsers[operation], schema)
    return parser


def _print(value) -> None:
    print(json.dumps(value, indent=2, default=str))


def _parameters(args: argparse.Namespace, operation: Operation) -> dict:
    selection = _selection(args)
    plugin = plugin_for(selection)
    model = plugin.catalog.resolve(selection.model)
    schema = plugin.handlers[operation].schema(model)
    return {spec.name: getattr(args, spec.name) for spec in schema.parameters if hasattr(args, spec.name)}


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser(effective_argv).parse_args(effective_argv)
    if args.command == "framework":
        for item in frameworks():
            _print({"framework": item.framework, "task": item.task.value, "operations": sorted(item.operations), "description": item.description, "optional_dependency": item.optional_dependency})
        return 0
    if args.command == "model":
        if args.model_command == "list":
            for item in models_for(TaskKind(args.task), args.framework, args.pattern):
                _print(asdict(item))
        else:
            selection = _selection(args)
            plugin = plugin_for(selection)
            model_info = plugin.catalog.resolve(selection.model)
            handler = plugin.handlers[Operation(args.operation)]
            _print({"selection": str(selection), "resolved_model": asdict(model_info), "parameters": handler.schema(model_info).describe(), "dataset": asdict(handler.dataset) if handler.dataset else None})
        return 0
    service = DatasetService()
    materialization = MaterializationMode(args.materialization) if hasattr(args, "materialization") else MaterializationMode.HARDLINK
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
            _print(asdict(service.split(SplitDatasetRequest(args.source, args.output, _split_policy(args), args.format, args.output_format, materialization, args.overwrite))))
        else:
            policy = _split_policy(args) if args.split else None
            _print(asdict(service.merge(MergeDatasetRequest(tuple(args.sources), args.output, args.output_format, policy, materialization, args.overwrite, BalanceMode(args.balance)))))
        return 0
    operation = Operation(args.command)
    selection = _selection(args)
    parameters = _parameters(args, operation)
    if operation is Operation.TRAIN:
        TrainService().run(TrainRequest(selection, args.data, args.output, args.weights, args.resume, args.overwrite, parameters))
        return 0
    elif operation is Operation.EXPORT:
        ExportService().run(ExportRequest(selection, args.checkpoint, args.output, args.data, parameters))
        return 0
    elif operation is Operation.VALIDATE:
        result = ValidationService().run(ValidateRequest(selection, args.target, args.data, args.split, parameters))
    else:
        result = TestService().run(TestRequest(selection, args.target, args.data, args.split, parameters))
    _print(asdict(result))
    return 0
