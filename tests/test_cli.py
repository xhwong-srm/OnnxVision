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


def test_cli_uses_model_image_size_when_training_size_is_omitted() -> None:
    args = build_parser().parse_args([
        "train", "--model", "timm/classification/mobilenetv4_conv_small_050.e3000_r224_in1k",
        "--data", "data", "--output", "run", "--worker", "4",
    ])
    assert args.image_size is None
    assert args.workers == 4


def test_cli_exposes_timm_loader_and_amp_options() -> None:
    args = build_parser().parse_args([
        "train", "--model", "timm/classification/resnet18", "--data", "data", "--output", "run",
        "--workers", "4", "--prefetch-factor", "3", "--persistent-workers", "--pin-memory",
        "--amp", "--amp-dtype", "bfloat16",
    ])
    assert args.prefetch_factor == 3
    assert args.persistent_workers is True
    assert args.pin_memory is True
    assert args.amp is True
    assert args.amp_dtype == "bfloat16"
