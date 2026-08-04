"""Fine-tune a pretrained timm MobileNetV3 classifier on class-folder data."""

from __future__ import annotations

import csv
import gc
import json
import os
import random
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import numpy as np
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision import datasets, transforms


MODEL_NAME = "mobilenetv3_small_100.lamb_in1k"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SELECTED_CONFIG_PATH = Path(__file__).resolve().with_name("tune_timm_gpu.selected.json")


def load_selected_config() -> dict:
    if not SELECTED_CONFIG_PATH.is_file():
        return {}
    try:
        document = json.loads(SELECTED_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read selected GPU config {SELECTED_CONFIG_PATH}: {error}") from error
    config = document.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Selected GPU config must contain an object named 'config': {SELECTED_CONFIG_PATH}")
    return config


def parse_args():
    selected_config = load_selected_config()
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("images/seal_dataset_v2"),
        help="Dataset root containing train/val/test class folders",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Stop after this many epochs without a better validation checkpoint; 0 disables early stopping",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=20,
        help="Train at least this many epochs before early stopping is allowed",
    )
    parser.add_argument(
        "--min-loss-delta",
        type=float,
        default=1e-4,
        help="Minimum validation-loss decrease required when the primary metric and accuracy are tied",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("accuracy", "macro_f1"),
        default="macro_f1",
        help="Primary validation metric used for best-checkpoint selection and early stopping",
    )
    parser.add_argument("--batch", type=int, default=selected_config.get("batch", 32))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--workers",
        type=int,
        default=selected_config.get("workers", -1),
        help="DataLoader worker processes; -1 selects automatically, 0 loads in the main process",
    )
    parser.add_argument("--prefetch-factor", type=int, default=selected_config.get("prefetch_factor", 2) or 2)
    parser.add_argument(
        "--persistent-workers",
        action=BooleanOptionalAction,
        default=selected_config.get("persistent_workers", True),
        help="Keep DataLoader workers alive between epochs when workers are enabled",
    )
    parser.add_argument(
        "--pin-memory",
        action=BooleanOptionalAction,
        default=selected_config.get("pin_memory", True),
        help="Use pinned host memory and asynchronous transfers on accelerators",
    )
    parser.add_argument(
        "--amp",
        action=BooleanOptionalAction,
        default=selected_config.get("amp", True),
        help="Use automatic mixed precision on CUDA",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16"),
        default=selected_config.get("amp_dtype", "float16"),
        help="CUDA automatic mixed-precision dtype",
    )
    parser.add_argument(
        "--channels-last",
        action=BooleanOptionalAction,
        default=selected_config.get("channels_last", True),
        help="Use channels-last tensors on CUDA for convolution-heavy models",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action=BooleanOptionalAction,
        default=True,
        help="Let cuDNN benchmark fixed input shapes and select faster convolution algorithms",
    )
    parser.add_argument(
        "--deterministic",
        action=BooleanOptionalAction,
        default=False,
        help="Favor reproducibility over speed with deterministic algorithms and seeded data loaders",
    )
    parser.add_argument(
        "--compile",
        action=BooleanOptionalAction,
        default=selected_config.get("compile", False),
        help="Use torch.compile; opt in after benchmarking compatibility on the target machine",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
        help="Float32 matrix-multiplication precision; high enables faster hardware paths where supported",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-percent",
        type=float,
        default=100.0,
        help=(
            "Percentage of each class to use from the training split. "
            "For example, 10 uses 50 of 500 images from every class."
        ),
    )
    parser.add_argument(
        "--balance-train",
        action=BooleanOptionalAction,
        default=False,
        help=(
            "Draw an equal number of training samples from each class per epoch, "
            "using replacement as needed. "
            "Validation and test data are never balanced."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "mobilenetv3",
        help="Directory in which checkpoints and metrics are saved",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Training device, for example "cuda", "cuda:0", or "cpu"',
    )
    return parser.parse_args()


def image_transform(model, *, train: bool):
    config = dict(timm.data.resolve_data_config(model.pretrained_cfg))
    input_size = tuple(config["input_size"])
    height, width = input_size[-2:]
    interpolation = getattr(transforms.InterpolationMode, config.get("interpolation", "bicubic").upper())
    # This trainer deliberately resizes the complete image instead of applying
    # timm's pretrained evaluation crop. Record the transform we actually use
    # so checkpoints, exports, and metadata do not advertise an unused crop.
    config.pop("crop_pct", None)
    config.pop("crop_mode", None)
    config["resize_mode"] = "stretch_to_input_size"
    config["antialias"] = True
    operations = [transforms.Resize((height, width), interpolation=interpolation, antialias=True)]
    if train:
        operations.extend(
            [
                transforms.RandomApply(
                    [
                        transforms.RandomAffine(
                            degrees=3,
                            translate=(0.03, 0.03),
                            scale=(0.97, 1.03),
                        ),
                        transforms.ColorJitter(brightness=0.10, contrast=0.10),
                    ],
                    p=0.5,
                ),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(config["mean"], config["std"]),
        ]
    )
    return transforms.Compose(operations), config


def make_dataset(root: Path, transform):
    if not root.is_dir():
        raise FileNotFoundError(f"Missing dataset split: {root}")
    dataset = datasets.ImageFolder(root, transform=transform, is_valid_file=lambda p: Path(p).suffix.lower() in IMAGE_EXTENSIONS)
    if not dataset.classes:
        raise ValueError(f"No class folders found in {root}")
    return dataset


def class_counts(targets, class_count):
    counts = [0] * class_count
    for target in targets:
        counts[target] += 1
    return counts


def stratified_subset_indices(targets, class_count, percent, seed):
    """Select the requested percentage independently and deterministically per class."""
    rng = random.Random(seed)
    by_class = [[] for _ in range(class_count)]
    for index, target in enumerate(targets):
        by_class[target].append(index)
    selected = []
    for indices in by_class:
        rng.shuffle(indices)
        selected_count = max(1, int(len(indices) * percent / 100))
        selected.extend(indices[:selected_count])
    selected.sort()
    return selected


class ClassBalancedSampler(Sampler[int]):
    """Draw an equal number of samples per class, with replacement as needed."""

    def __init__(self, targets, seed):
        self.class_indices = []
        for class_index in range(max(targets) + 1):
            indices = [index for index, target in enumerate(targets) if target == class_index]
            if indices:
                self.class_indices.append(torch.tensor(indices, dtype=torch.int64))
        self.num_samples = len(targets)
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        base_count, remainder = divmod(self.num_samples, len(self.class_indices))
        selected = []
        for index, indices in enumerate(self.class_indices):
            count = base_count + (index < remainder)
            choices = torch.randint(len(indices), (count,), generator=self.generator)
            selected.append(indices[choices])
        combined = torch.cat(selected)
        order = torch.randperm(len(combined), generator=self.generator)
        yield from combined[order].tolist()


def balanced_sampler(targets, seed):
    return ClassBalancedSampler(targets, seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def classification_metrics(confusion_matrix, classes):
    per_class = {}
    f1_scores = []
    for index, class_name in enumerate(classes):
        true_positive = confusion_matrix[index][index]
        actual_count = sum(confusion_matrix[index])
        predicted_count = sum(row[index] for row in confusion_matrix)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / actual_count if actual_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_scores.append(f1)
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual_count,
        }
    return {
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
    }


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    classes=None,
    amp_enabled=False,
    amp_dtype=torch.float16,
    channels_last=False,
    non_blocking=False,
):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_items = 0
    confusion_matrix = [[0 for _ in classes] for _ in classes] if classes is not None else None
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images = images.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=autocast_device,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * targets.size(0)
            predictions = logits.argmax(1)
            total_correct += (predictions == targets).sum().item()
            total_items += targets.size(0)
            if confusion_matrix is not None:
                for target, prediction in zip(targets.tolist(), predictions.tolist()):
                    confusion_matrix[target][prediction] += 1
    metrics = classification_metrics(confusion_matrix, classes) if confusion_matrix is not None else None
    return total_loss / total_items, total_correct / total_items, metrics


def is_better_checkpoint(score, val_accuracy, val_loss, best_score, best_accuracy, best_loss, min_loss_delta):
    if score > best_score:
        return True, "higher primary validation metric"
    if score == best_score and val_accuracy > best_accuracy:
        return True, "equal primary metric with higher validation accuracy"
    if score == best_score and val_accuracy == best_accuracy and val_loss < best_loss - min_loss_delta:
        return True, "equal primary metric and accuracy with lower validation loss"
    return False, None


def save_checkpoint(path, model, optimizer, epoch, val_accuracy, val_loss, val_metrics, classes, config, model_name):
    torch.save(
        {
            "model_name": model_name,
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "classes": classes,
            "class_to_idx": {name: index for index, name in enumerate(classes)},
            "data_config": config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.patience < 0:
        raise ValueError("--patience cannot be negative")
    if args.min_epochs < 0:
        raise ValueError("--min-epochs cannot be negative")
    if args.min_loss_delta < 0:
        raise ValueError("--min-loss-delta cannot be negative")
    if args.workers < -1:
        raise ValueError("--workers must be -1 or greater")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be at least 1")
    if not 0 < args.train_percent <= 100:
        raise ValueError("--train-percent must be greater than 0 and at most 100")

    if args.deterministic:
        # This must be set before CUDA creates a cuBLAS workspace.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = args.cudnn_benchmark and device.type == "cuda"

    cpu_count = os.cpu_count() or 1
    workers = args.workers
    if workers == -1:
        workers = min(8, max(1, cpu_count // 2)) if device.type == "cuda" else 0
    pin_memory = args.pin_memory and device.type == "cuda"
    persistent_workers = args.persistent_workers and workers > 0
    channels_last = args.channels_last and device.type == "cuda"
    amp_enabled = args.amp and device.type == "cuda"
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    non_blocking = pin_memory and device.type == "cuda"

    data = args.data.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    model = timm.create_model(args.model, pretrained=True, num_classes=0)
    train_transform, config = image_transform(model, train=True)
    eval_transform, _ = image_transform(model, train=False)
    full_train_set = make_dataset(data / "train", train_transform)
    val_set = make_dataset(data / "val", eval_transform)
    if full_train_set.classes != val_set.classes:
        raise ValueError(f"Train/val classes differ: {full_train_set.classes} != {val_set.classes}")
    test_set = make_dataset(data / "test", eval_transform) if (data / "test").is_dir() else None
    if test_set is not None and full_train_set.classes != test_set.classes:
        raise ValueError(f"Train/test classes differ: {full_train_set.classes} != {test_set.classes}")

    train_indices = stratified_subset_indices(
        full_train_set.targets,
        len(full_train_set.classes),
        args.train_percent,
        args.seed,
    )
    train_set = Subset(full_train_set, train_indices)
    train_targets = [full_train_set.targets[index] for index in train_indices]
    original_train_counts = class_counts(full_train_set.targets, len(full_train_set.classes))
    selected_train_counts = class_counts(train_targets, len(full_train_set.classes))
    train_sampler = balanced_sampler(train_targets, args.seed) if args.balance_train else None

    model.reset_classifier(len(full_train_set.classes))
    model.to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    training_model = torch.compile(model) if args.compile else model

    loader_args = {
        "batch_size": args.batch,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if workers > 0:
        loader_args["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(
        train_set,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=torch.Generator().manual_seed(args.seed),
        worker_init_fn=seed_worker,
        **loader_args,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        generator=torch.Generator().manual_seed(args.seed + 1),
        worker_init_fn=seed_worker,
        **loader_args,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(training_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16,
    )

    print(
        f"device={device}; workers={workers}; pin_memory={pin_memory}; "
        f"amp={amp_enabled} ({args.amp_dtype}); channels_last={channels_last}; "
        f"compile={args.compile}; deterministic={args.deterministic}; "
        f"cudnn_benchmark={torch.backends.cudnn.benchmark}; "
        f"train_percent={args.train_percent:g}; balance_train={args.balance_train}"
    )
    print(
        "train samples by class: "
        + ", ".join(
            f"{class_name}={selected}/{original}"
            for class_name, selected, original in zip(
                full_train_set.classes, selected_train_counts, original_train_counts
            )
        )
    )

    # Keep the effective configuration alongside every run so it can be reproduced
    # even when the command used to start training is no longer available.
    parameters = {
        "command": [sys.executable, *sys.argv],
        "data": str(data),
        "model": args.model,
        "epochs": args.epochs,
        "patience": args.patience,
        "min_epochs": args.min_epochs,
        "min_loss_delta": args.min_loss_delta,
        "selection_metric": args.selection_metric,
        "batch": args.batch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "requested_workers": args.workers,
        "workers": workers,
        "prefetch_factor": args.prefetch_factor if workers > 0 else None,
        "persistent_workers": persistent_workers,
        "pin_memory": pin_memory,
        "non_blocking_transfers": non_blocking,
        "amp": amp_enabled,
        "amp_dtype": args.amp_dtype if amp_enabled else None,
        "channels_last": channels_last,
        "compile": args.compile,
        "deterministic": args.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_precision": args.matmul_precision,
        "seed": args.seed,
        "train_percent": args.train_percent,
        "balance_train": args.balance_train,
        "balance_method": "exact class-balanced sampling with replacement" if args.balance_train else None,
        "output": str(output),
        "requested_device": args.device,
        "device": str(device),
        "pretrained": True,
        "optimizer": "AdamW",
        "scheduler": {"name": "CosineAnnealingLR", "T_max": args.epochs},
        "loss": "CrossEntropyLoss",
        "mixed_precision": amp_enabled,
        "num_classes": len(full_train_set.classes),
        "classes": full_train_set.classes,
        "original_train_samples": len(full_train_set),
        "train_samples": len(train_set),
        "original_train_samples_per_class": dict(zip(full_train_set.classes, original_train_counts)),
        "train_samples_per_class": dict(zip(full_train_set.classes, selected_train_counts)),
        "val_samples": len(val_set),
        "test_samples": len(test_set) if test_set is not None else 0,
        "torch_version": torch.__version__,
        "timm_version": getattr(timm, "__version__", "unknown"),
        "data_config": config,
    }
    best_score = -1.0
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_val_metrics = None
    epochs_without_improvement = 0
    history = []
    stop_reason = "maximum epochs reached"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, _ = run_epoch(
            training_model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
            non_blocking=non_blocking,
        )
        val_loss, val_accuracy, val_metrics = run_epoch(
            training_model,
            val_loader,
            criterion,
            device,
            classes=full_train_set.classes,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
            non_blocking=non_blocking,
        )
        selection_score = val_accuracy if args.selection_metric == "accuracy" else val_metrics["macro_f1"]
        scheduler.step()
        improved, improvement_reason = is_better_checkpoint(
            selection_score,
            val_accuracy,
            val_loss,
            best_score,
            best_accuracy,
            best_loss,
            args.min_loss_delta,
        )
        if improved:
            best_score = selection_score
            best_accuracy = val_accuracy
            best_loss = val_loss
            best_epoch = epoch
            best_val_metrics = val_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_metrics["macro_f1"],
            "selection_score": selection_score,
            "is_best": improved,
            "epochs_without_improvement": epochs_without_improvement,
        }
        history.append(row)
        print(
            "epoch {epoch:03d}: train loss={train_loss:.4f} acc={train_accuracy:.4f}; "
            "val loss={val_loss:.4f} acc={val_accuracy:.4f} macro_f1={val_macro_f1:.4f}".format(**row)
        )
        save_checkpoint(
            output / "last.pt",
            model,
            optimizer,
            epoch,
            val_accuracy,
            val_loss,
            val_metrics,
            full_train_set.classes,
            config,
            args.model,
        )
        if improved:
            save_checkpoint(
                output / "best.pt",
                model,
                optimizer,
                epoch,
                val_accuracy,
                val_loss,
                val_metrics,
                full_train_set.classes,
                config,
                args.model,
            )
            print(f"  saved best.pt: {improvement_reason}")

        can_stop = epoch >= args.min_epochs
        patience_exhausted = args.patience > 0 and epochs_without_improvement >= args.patience
        if can_stop and patience_exhausted:
            stop_reason = (
                f"early stopping after {args.patience} epochs without a better validation checkpoint"
            )
            print(
                f"{stop_reason}; best epoch={best_epoch}, "
                f"{args.selection_metric}={best_score:.4f}, "
                f"val loss={best_loss:.4f}, val acc={best_accuracy:.4f}"
            )
            break

    with (output / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    # Persistent train/validation workers and optimizer state are no longer
    # needed. Release them before final evaluation so Windows does not have to
    # start another group of CUDA-importing worker processes at peak commit.
    del train_loader, val_loader, optimizer, scheduler, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    metadata = {
        "model_name": args.model,
        "classes": full_train_set.classes,
        "data_config": config,
        "parameters": parameters,
        "training_result": {
            "completed_epochs": len(history),
            "best_epoch": best_epoch,
            "selection_metric": args.selection_metric,
            "best_selection_score": best_score,
            "best_val_loss": best_loss,
            "best_val_accuracy": best_accuracy,
            "best_val_metrics": best_val_metrics,
            "stop_reason": stop_reason,
        },
    }
    if test_set is not None:
        # Test evaluation is a one-shot pass, so extra worker processes provide
        # little benefit and can exhaust the Windows pagefile after training.
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
            persistent_workers=False,
            generator=torch.Generator().manual_seed(args.seed + 2),
            worker_init_fn=seed_worker,
        )
        best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        checkpoint_epoch = best["epoch"]
        model.load_state_dict(best["model_state_dict"])
        del best
        gc.collect()
        test_loss, test_accuracy, test_metrics = run_epoch(
            training_model,
            test_loader,
            criterion,
            device,
            classes=full_train_set.classes,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
            non_blocking=non_blocking,
        )
        print(
            f"test loss={test_loss:.4f} acc={test_accuracy:.4f} "
            f"macro_f1={test_metrics['macro_f1']:.4f}"
        )
        metadata["test_result"] = {
            "checkpoint_epoch": checkpoint_epoch,
            "loss": test_loss,
            "accuracy": test_accuracy,
            **test_metrics,
        }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
