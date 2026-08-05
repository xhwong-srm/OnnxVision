from __future__ import annotations

from vision_workflows.cli.app import build_parser


def test_cli_has_separate_dataset_and_workflow_commands() -> None:
    parser = build_parser()
    assert parser.prog == "vision-workflows"
    assert parser.parse_args(["dataset", "convert", "source", "output", "--output-format", "yolo"]).dataset_command == "convert"
    assert parser.parse_args(["train", "--model", "timm/classification/resnet18", "--data", "data", "--output", "run"]).command == "train"


def test_cli_exposes_explicit_experimental_training_opt_in() -> None:
    args = build_parser().parse_args([
        "train", "--model", "libreyolo/picodet/s", "--data", "data.yaml", "--output", "run",
        "--allow-experimental", "--overwrite",
    ])
    assert args.allow_experimental is True
    assert args.overwrite is True
