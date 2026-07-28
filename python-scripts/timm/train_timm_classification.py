"""Fine-tune a pretrained timm MobileNetV3 classifier on class-folder data."""

from __future__ import annotations

import csv
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import timm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


MODEL_NAME = "mobilenetv3_small_100.lamb_in1k"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
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
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
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
    config = timm.data.resolve_data_config(model.pretrained_cfg)
    input_size = tuple(config["input_size"])
    height, width = input_size[-2:]
    interpolation = getattr(transforms.InterpolationMode, config.get("interpolation", "bicubic").upper())
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


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, classes=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_items = 0
    confusion_matrix = [[0 for _ in classes] for _ in classes] if classes is not None else None
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=autocast_device, enabled=device.type == "cuda"):
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

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data = args.data.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    model = timm.create_model(args.model, pretrained=True, num_classes=0)
    train_transform, config = image_transform(model, train=True)
    eval_transform, _ = image_transform(model, train=False)
    train_set = make_dataset(data / "train", train_transform)
    val_set = make_dataset(data / "val", eval_transform)
    if train_set.classes != val_set.classes:
        raise ValueError(f"Train/val classes differ: {train_set.classes} != {val_set.classes}")
    test_set = make_dataset(data / "test", eval_transform) if (data / "test").is_dir() else None
    if test_set is not None and train_set.classes != test_set.classes:
        raise ValueError(f"Train/test classes differ: {train_set.classes} != {test_set.classes}")

    model.reset_classifier(len(train_set.classes))
    model.to(device)
    loader_args = {"batch_size": args.batch, "num_workers": args.workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args) if test_set else None
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

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
        "workers": args.workers,
        "seed": args.seed,
        "output": str(output),
        "requested_device": args.device,
        "device": str(device),
        "pretrained": True,
        "optimizer": "AdamW",
        "scheduler": {"name": "CosineAnnealingLR", "T_max": args.epochs},
        "loss": "CrossEntropyLoss",
        "mixed_precision": device.type == "cuda",
        "num_classes": len(train_set.classes),
        "classes": train_set.classes,
        "train_samples": len(train_set),
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
        train_loss, train_accuracy, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_loss, val_accuracy, val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            classes=train_set.classes,
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
            train_set.classes,
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
                train_set.classes,
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
    metadata = {
        "model_name": args.model,
        "classes": train_set.classes,
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
    if test_loader is not None:
        best = torch.load(output / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best["model_state_dict"])
        test_loss, test_accuracy, test_metrics = run_epoch(
            model,
            test_loader,
            criterion,
            device,
            classes=train_set.classes,
        )
        print(
            f"test loss={test_loss:.4f} acc={test_accuracy:.4f} "
            f"macro_f1={test_metrics['macro_f1']:.4f}"
        )
        metadata["test_result"] = {
            "checkpoint_epoch": best["epoch"],
            "loss": test_loss,
            "accuracy": test_accuracy,
            **test_metrics,
        }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
