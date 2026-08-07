from __future__ import annotations

import pytest

from vision_workflows.cli.app import build_parser


def _parse(argv: list[str]):
    return build_parser(argv).parse_args(argv)


def test_cli_has_explicit_task_framework_and_model_selectors() -> None:
    dataset = ["dataset", "convert", "source", "output", "--output-format", "yolo"]
    assert build_parser(dataset).parse_args(dataset).dataset_command == "convert"
    train = [
        "train", "--task", "classification", "--framework", "ultralytics",
        "--model", "yolo26n", "--data", "data", "--output", "run",
    ]
    args = _parse(train)
    assert (args.task, args.framework, args.model) == ("classification", "ultralytics", "yolo26n")


def test_cli_generates_ultralytics_classification_parameters() -> None:
    argv = [
        "train", "--task", "classification", "--framework", "ultralytics",
        "--model", "yolo26n", "--data", "data", "--output", "run",
        "--workers", "2", "--dropout", "0.1", "--no-amp",
    ]
    args = _parse(argv)
    assert args.workers == 2
    assert args.dropout == 0.1
    assert args.amp is False


def test_cli_exposes_libreyolo_amp_and_cache_options() -> None:
    argv = [
        "train", "--task", "object-detection", "--framework", "libreyolo",
        "--model", "yolov9t", "--data", "data.yaml", "--output", "run",
        "--no-amp", "--cache", "ram",
    ]
    args = _parse(argv)
    assert args.amp is False
    assert args.cache == "ram"


def test_cli_does_not_expose_timm_parameters_to_ultralytics() -> None:
    argv = [
        "train", "--task", "classification", "--framework", "ultralytics",
        "--model", "yolo26n", "--data", "data", "--output", "run",
        "--validate-every", "2",
    ]
    with pytest.raises(SystemExit):
        _parse(argv)


def test_cli_exposes_timm_loader_and_amp_options() -> None:
    argv = [
        "train", "--task", "classification", "--framework", "timm",
        "--model", "mobilenetv4_conv_small_050.e3000_r224_in1k", "--data", "data", "--output", "run",
        "--workers", "4", "--val-workers", "2", "--prefetch-factor", "3", "--persistent-workers", "--cache", "disk",
        "--pin-memory", "--amp", "--amp-dtype", "bfloat16",
    ]
    args = _parse(argv)
    assert args.prefetch_factor == 3
    assert args.persistent_workers is True
    assert args.pin_memory is True
    assert args.amp is True
    assert args.amp_dtype == "bfloat16"
    assert args.cache == "disk"
    assert args.val_workers == 2


def test_cli_exposes_timm_albumentations_and_optuna_tuning() -> None:
    argv = [
        "tune", "--task", "classification", "--framework", "timm",
        "--model", "mobilenetv4_conv_small_050.e3000_r224_in1k", "--data", "data", "--output", "run",
        "--augmentation-backend", "albumentations", "--augmentation-policy", "robust", "--augmentation",
        "--trials", "3", "--learning-rate-min", "0.00001",
        "--learning-rate-max", "0.001", "--label-smoothing-min", "0.01", "--label-smoothing-max", "0.1",
    ]
    args = _parse(argv)
    assert args.augmentation is True
    assert args.augmentation_backend == "albumentations"
    assert args.augmentation_policy == "robust"
    assert args.trials == 3
    assert args.learning_rate_min == 0.00001
    assert args.learning_rate_max == 0.001
    assert args.label_smoothing_min == 0.01
    assert args.label_smoothing_max == 0.1


def test_cli_can_disable_timm_augmentation() -> None:
    argv = [
        "train", "--task", "classification", "--framework", "timm",
        "--model", "mobilenetv4_conv_small_050.e3000_r224_in1k", "--data", "data", "--output", "run",
        "--no-augmentation",
    ]
    assert _parse(argv).augmentation is False


def test_cli_accepts_timm_pretrained_configuration_name() -> None:
    model = "mobilenetv4_conv_small_050.e3000_r224_in1k"
    argv = [
        "train", "--task", "classification", "--framework", "timm",
        "--model", model, "--data", "data", "--output", "run",
    ]
    args = _parse(argv)
    assert args.model == model


def test_export_batch_size_is_optional_and_positive() -> None:
    base = [
        "export", "--task", "classification", "--framework", "timm",
        "--model", "mobilenetv4_conv_small_050.e3000_r224_in1k",
        "--checkpoint", "model.pt", "--output", "model.onnx",
    ]
    assert getattr(_parse(base), "batch_size", None) is None
    assert _parse(base + ["--batch-size", "4"]).batch_size == 4
