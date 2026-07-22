"""Fine-tune a pretrained timm MobileNetV3 classifier on class-folder data."""

from __future__ import annotations

import csv
import json
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
    parser.add_argument("--epochs", type=int, default=50)
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


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_items = 0
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
            total_correct += (logits.argmax(1) == targets).sum().item()
            total_items += targets.size(0)
    return total_loss / total_items, total_correct / total_items


def save_checkpoint(path, model, optimizer, epoch, val_accuracy, classes, config, model_name):
    torch.save(
        {
            "model_name": model_name,
            "epoch": epoch,
            "val_accuracy": val_accuracy,
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
    best_accuracy = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy, "val_loss": val_loss, "val_accuracy": val_accuracy}
        history.append(row)
        print("epoch {epoch:03d}: train loss={train_loss:.4f} acc={train_accuracy:.4f}; val loss={val_loss:.4f} acc={val_accuracy:.4f}".format(**row))
        save_checkpoint(output / "last.pt", model, optimizer, epoch, val_accuracy, train_set.classes, config, args.model)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(output / "best.pt", model, optimizer, epoch, val_accuracy, train_set.classes, config, args.model)

    with (output / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    (output / "metadata.json").write_text(json.dumps({"model_name": args.model, "classes": train_set.classes, "data_config": config}, indent=2), encoding="utf-8")
    if test_loader is not None:
        best = torch.load(output / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best["model_state_dict"])
        test_loss, test_accuracy = run_epoch(model, test_loader, criterion, device)
        print(f"test loss={test_loss:.4f} acc={test_accuracy:.4f}")


if __name__ == "__main__":
    main()
