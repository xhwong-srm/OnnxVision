"""Train a MobileNetV4-backed RetinaNet detector on a COCO-format dataset.

The dataset layout is the COCO export produced by ``dataset_builder.py``::

    dataset/
      images/{train,val,test}/...
      annotations/instances_{train,val,test}.json

The Timm encoder exposes stride-8, stride-16, and stride-32 feature maps. A
256-channel FPN adds P6/P7 before the RetinaNet prediction heads. Images are
stretched to a fixed square by default so the training geometry matches the
embedded BW8/C24 ONNX preprocessing used by the deployment exporter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import OrderedDict, defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.ops import FeaturePyramidNetwork, box_iou
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.transforms.functional import pil_to_tensor


MODEL_NAME = "mobilenetv4_conv_small"
OUT_INDICES = (2, 3, 4)
DEFAULT_FPN_CHANNELS = 256
DEFAULT_ANCHOR_SIZES = (32, 64, 128, 256, 512)
DEFAULT_ANCHOR_RATIOS = (0.5, 1.0, 2.0)
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_number_list(value: str, name: str, positive: bool = True) -> tuple[float, ...]:
    try:
        numbers = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated list of numbers: {value!r}") from error
    if not numbers or (positive and any(number <= 0 for number in numbers)):
        raise ValueError(f"{name} must contain positive numbers: {value!r}")
    return numbers


def parse_int_list(value: str, name: str) -> tuple[int, ...]:
    numbers = parse_number_list(value, name)
    if any(number != int(number) for number in numbers):
        raise ValueError(f"{name} must contain integers: {value!r}")
    return tuple(int(number) for number in numbers)


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def collate_detection_batch(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def resolve_image_path(root: Path, split: str, file_name: str) -> Path:
    relative = Path(file_name)
    candidates = [
        root / relative,
        root / "images" / split / relative.name,
        root / relative.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"COCO image {file_name!r} from {split!r} was not found below {root}; "
        f"checked: {', '.join(str(path) for path in candidates)}"
    )


class CocoDetectionDataset(Dataset):
    """Small dependency-free COCO detection reader for the local dataset layout."""

    def __init__(self, root: Path, split: str, imgsz: int, train: bool, stretch: bool):
        self.root = root.resolve()
        self.split = split
        self.imgsz = imgsz
        self.train = train
        self.stretch = stretch
        annotation_path = self.root / "annotations" / f"instances_{split}.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing COCO annotation file: {annotation_path}")
        try:
            document = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read COCO annotations {annotation_path}: {error}") from error

        categories = document.get("categories")
        if not isinstance(categories, list) or not categories:
            raise ValueError(f"COCO annotations must contain non-empty categories: {annotation_path}")
        ordered_categories = sorted(categories, key=lambda item: int(item["id"]))
        self.class_names = [str(item["name"]) for item in ordered_categories]
        self.category_to_label = {
            int(item["id"]): index for index, item in enumerate(ordered_categories)
        }

        images = document.get("images")
        if not isinstance(images, list) or not images:
            raise ValueError(f"COCO annotations must contain non-empty images: {annotation_path}")
        self.images = []
        for item in images:
            image_id = int(item["id"])
            file_name = str(item["file_name"])
            path = resolve_image_path(self.root, split, file_name)
            self.images.append({"id": image_id, "path": path, "file_name": file_name})

        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in document.get("annotations", []):
            annotations_by_image[int(annotation["image_id"])].append(annotation)
        self.annotations_by_image = annotations_by_image

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        record = self.images[index]
        with Image.open(record["path"]) as source:
            image = source.convert("RGB")
        width, height = image.size
        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        for annotation in self.annotations_by_image.get(record["id"], []):
            category_id = int(annotation["category_id"])
            if category_id not in self.category_to_label:
                continue
            x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
            x1 = max(0.0, min(float(width), x))
            y1 = max(0.0, min(float(height), y))
            x2 = max(0.0, min(float(width), x + box_width))
            y2 = max(0.0, min(float(height), y + box_height))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.category_to_label[category_id])
            areas.append((x2 - x1) * (y2 - y1))

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        if self.stretch:
            scale_x = self.imgsz / width
            scale_y = self.imgsz / height
            image = image.resize((self.imgsz, self.imgsz), Image.Resampling.BILINEAR)
            if boxes_tensor.numel():
                boxes_tensor *= torch.tensor([scale_x, scale_y, scale_x, scale_y])
            areas = [area * scale_x * scale_y for area in areas]
            canvas_size = (self.imgsz, self.imgsz)
        else:
            canvas_size = (height, width)

        if self.train and random.random() < 0.5:
            image = ImageOps.mirror(image)
            if boxes_tensor.numel():
                current_width = image.width
                left = boxes_tensor[:, 0].clone()
                boxes_tensor[:, 0] = current_width - boxes_tensor[:, 2]
                boxes_tensor[:, 2] = current_width - left

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor(record["id"], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }
        image_tensor = pil_to_tensor(image).float().div(255.0)
        return image_tensor, target


class TimmFpnBackbone(nn.Module):
    """Adapt a Timm multi-scale encoder to the Torchvision FPN backbone contract."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        fpn_channels: int,
        out_indices: tuple[int, ...] = OUT_INDICES,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        channels = list(self.encoder.feature_info.channels())
        reductions = list(self.encoder.feature_info.reduction())
        if reductions != [8, 16, 32]:
            raise ValueError(
                f"{model_name!r} must expose stride 8/16/32 features for this RetinaNet adapter; "
                f"received reductions={reductions}"
            )
        self.fpn = FeaturePyramidNetwork(
            channels,
            out_channels=fpn_channels,
            extra_blocks=LastLevelP6P7(fpn_channels, fpn_channels),
        )
        self.out_channels = fpn_channels
        self.model_name = model_name
        self.feature_channels = channels
        self.feature_reductions = reductions

    def forward(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        features = self.encoder(images)
        return self.fpn(OrderedDict((str(index), value) for index, value in enumerate(features)))


def model_data_config(backbone: TimmFpnBackbone) -> dict[str, Any]:
    config = dict(timm.data.resolve_model_data_config(backbone.encoder))
    config["mean"] = [float(value) for value in config.get("mean", (0.485, 0.456, 0.406))]
    config["std"] = [float(value) for value in config.get("std", (0.229, 0.224, 0.225))]
    config["input_size"] = [int(value) for value in config.get("input_size", (3, 224, 224))]
    config["interpolation"] = str(config.get("interpolation", "bilinear"))
    return config


def build_detector(
    model_name: str,
    num_classes: int,
    imgsz: int,
    fpn_channels: int = DEFAULT_FPN_CHANNELS,
    anchor_sizes: tuple[int, ...] = DEFAULT_ANCHOR_SIZES,
    anchor_ratios: tuple[float, ...] = DEFAULT_ANCHOR_RATIOS,
    pretrained: bool = True,
    image_mean: list[float] | None = None,
    image_std: list[float] | None = None,
) -> tuple[RetinaNet, TimmFpnBackbone, dict[str, Any]]:
    backbone = TimmFpnBackbone(model_name, pretrained, fpn_channels)
    config = model_data_config(backbone)
    mean = image_mean or config["mean"]
    std = image_std or config["std"]
    anchor_generator = AnchorGenerator(
        sizes=tuple((int(size),) for size in anchor_sizes),
        aspect_ratios=tuple(tuple(float(ratio) for ratio in anchor_ratios) for _ in anchor_sizes),
    )
    head = RetinaNetHead(
        fpn_channels,
        anchor_generator.num_anchors_per_location()[0],
        num_classes,
        norm_layer=partial(nn.GroupNorm, 32),
    )
    model = RetinaNet(
        backbone,
        num_classes=num_classes,
        min_size=imgsz,
        max_size=imgsz,
        image_mean=mean,
        image_std=std,
        anchor_generator=anchor_generator,
        head=head,
        score_thresh=0.001,
        nms_thresh=0.5,
        detections_per_img=300,
        topk_candidates=1000,
    )
    config.update({"mean": mean, "std": std})
    return model, backbone, config


def move_targets(targets: list[dict[str, torch.Tensor]], device: torch.device):
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def amp_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def train_epoch(
    model: RetinaNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for images, targets in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        targets = move_targets(targets, device)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, amp_enabled, amp_dtype):
            losses = model(images, targets)
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite detection loss: {float(loss.detach().cpu())}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu())
        batches += 1
    return total_loss / max(1, batches)


def average_precision(scores_and_matches: list[tuple[float, bool]], ground_truth_count: int) -> float:
    if ground_truth_count <= 0:
        return float("nan")
    if not scores_and_matches:
        return 0.0
    ordered = sorted(scores_and_matches, key=lambda item: item[0], reverse=True)
    true_positive = np.cumsum([int(item[1]) for item in ordered], dtype=np.float64)
    false_positive = np.cumsum([int(not item[1]) for item in ordered], dtype=np.float64)
    recall = true_positive / max(1, ground_truth_count)
    precision = true_positive / np.maximum(1.0, true_positive + false_positive)
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_points = np.concatenate(([0.0], recall))
    precision_points = np.concatenate(([precision_envelope[0]], precision_envelope))
    return float(np.sum((recall_points[1:] - recall_points[:-1]) * precision_points[1:]))


def detection_metrics(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    class_names: list[str],
    score_threshold: float,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    matched_detections: list[list[tuple[float, bool]]] = [[] for _ in class_names]
    ground_truth_counts = [0 for _ in class_names]
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for prediction, target in zip(predictions, targets):
        predicted_boxes = prediction["boxes"].detach().cpu()
        predicted_scores = prediction["scores"].detach().cpu()
        predicted_labels = prediction["labels"].detach().cpu()
        target_boxes = target["boxes"].detach().cpu()
        target_labels = target["labels"].detach().cpu()
        for class_id in range(len(class_names)):
            gt_indices = torch.where(target_labels == class_id)[0]
            ground_truth_counts[class_id] += int(gt_indices.numel())
            prediction_indices = torch.where(
                (predicted_labels == class_id) & (predicted_scores >= score_threshold)
            )[0]
            if prediction_indices.numel():
                prediction_indices = prediction_indices[
                    torch.argsort(predicted_scores[prediction_indices], descending=True)
                ]
            used = torch.zeros(gt_indices.numel(), dtype=torch.bool)
            for prediction_index in prediction_indices.tolist():
                score = float(predicted_scores[prediction_index])
                if not gt_indices.numel():
                    matched = False
                else:
                    overlaps = box_iou(
                        predicted_boxes[prediction_index : prediction_index + 1],
                        target_boxes[gt_indices],
                    )[0]
                    overlaps[used] = -1.0
                    best_overlap, best_index = overlaps.max(dim=0)
                    matched = bool(best_overlap >= iou_threshold)
                    if matched:
                        used[best_index] = True
                matched_detections[class_id].append((score, matched))
                true_positive += int(matched)
                false_positive += int(not matched)
            false_negative += int((~used).sum())

    aps = [average_precision(items, count) for items, count in zip(matched_detections, ground_truth_counts)]
    valid_aps = [value for value in aps if math.isfinite(value)]
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "map50": float(np.mean(valid_aps)) if valid_aps else 0.0,
        "precision50": precision,
        "recall50": recall,
        "f1_50": 2 * precision * recall / max(1e-12, precision + recall),
        "per_class_ap50": {
            name: (None if not math.isfinite(ap) else float(ap))
            for name, ap in zip(class_names, aps)
        },
        "ground_truth_boxes": sum(ground_truth_counts),
        "predicted_true_positive": true_positive,
        "predicted_false_positive": false_positive,
        "missed_ground_truth": false_negative,
    }


@torch.no_grad()
def evaluate(
    model: RetinaNet,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    score_threshold: float,
) -> dict[str, Any]:
    model.eval()
    predictions = []
    targets = []
    for images, batch_targets in loader:
        device_images = [image.to(device, non_blocking=True) for image in images]
        predictions.extend(model(device_images))
        targets.extend(batch_targets)
    return detection_metrics(predictions, targets, class_names, score_threshold)


def save_checkpoint(
    path: Path,
    model: RetinaNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    classes: list[str],
    model_name: str,
    config: dict[str, Any],
    model_config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "task": "object_detection",
            "model_name": model_name,
            "classes": classes,
            "class_to_idx": {name: index for index, name in enumerate(classes)},
            "model_config": model_config,
            "data_config": config,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "torch_version": torch.__version__,
            "timm_version": getattr(timm, "__version__", "unknown"),
        },
        path,
    )


def make_loader(dataset: Dataset, batch: int, workers: int, pin_memory: bool, seed: int, shuffle: bool):
    kwargs: dict[str, Any] = {
        "batch_size": batch,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": workers > 0,
        "collate_fn": collate_detection_batch,
        "generator": torch.Generator().manual_seed(seed),
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="COCO dataset root")
    parser.add_argument("--model", default=MODEL_NAME, help="Timm MobileNetV4 model name")
    parser.add_argument("--imgsz", type=int, default=384, help="Fixed square input size")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("python-scripts/timm/runs/mobilenetv4-retinanet"))
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stretch-to-input-size", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fpn-channels", type=int, default=DEFAULT_FPN_CHANNELS)
    parser.add_argument("--anchor-sizes", default=",".join(str(value) for value in DEFAULT_ANCHOR_SIZES))
    parser.add_argument("--anchor-ratios", default=",".join(str(value) for value in DEFAULT_ANCHOR_RATIOS))
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--run-test", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.imgsz < 32 or args.imgsz % 32:
        raise ValueError("--imgsz must be at least 32 and divisible by 32")
    if args.batch < 1 or args.epochs < 1 or args.patience < 0 or args.min_epochs < 0:
        raise ValueError("--batch/--epochs must be positive and --patience/--min-epochs cannot be negative")
    if not 0 <= args.score_threshold <= 1:
        raise ValueError("--score-threshold must be between 0 and 1")
    anchor_sizes = parse_int_list(args.anchor_sizes, "--anchor-sizes")
    anchor_ratios = parse_number_list(args.anchor_ratios, "--anchor-ratios")
    if len(anchor_sizes) != 5:
        raise ValueError("--anchor-sizes must contain five values for P3-P7")

    set_seed(args.seed, args.deterministic)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    workers = args.workers
    if workers == -1:
        workers = min(8, max(1, (os.cpu_count() or 1) // 2)) if device.type == "cuda" else 0
    amp_enabled = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    pin_memory = device.type == "cuda"

    data = args.data.expanduser().resolve()
    output = args.output.expanduser().resolve()
    train_set = CocoDetectionDataset(data, "train", args.imgsz, train=True, stretch=args.stretch_to_input_size)
    val_set = CocoDetectionDataset(data, "val", args.imgsz, train=False, stretch=args.stretch_to_input_size)
    if train_set.class_names != val_set.class_names:
        raise ValueError(f"Train/val categories differ: {train_set.class_names} != {val_set.class_names}")
    test_set = None
    if args.run_test and (data / "annotations" / "instances_test.json").is_file():
        test_set = CocoDetectionDataset(data, "test", args.imgsz, train=False, stretch=args.stretch_to_input_size)
        if train_set.class_names != test_set.class_names:
            raise ValueError(f"Train/test categories differ: {train_set.class_names} != {test_set.class_names}")

    model, backbone, data_config = build_detector(
        args.model,
        len(train_set.class_names),
        args.imgsz,
        fpn_channels=args.fpn_channels,
        anchor_sizes=anchor_sizes,
        anchor_ratios=anchor_ratios,
        pretrained=args.pretrained,
    )
    data_config = dict(data_config)
    data_config["input_size"] = [3, args.imgsz, args.imgsz]
    data_config["interpolation"] = "bilinear"
    data_config["resize_mode"] = "stretch_to_input_size" if args.stretch_to_input_size else "retinanet_aspect_preserving"
    data_config["stretch_to_input_size"] = bool(args.stretch_to_input_size)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    train_loader = make_loader(train_set, args.batch, workers, pin_memory, args.seed, shuffle=True)
    val_loader = make_loader(val_set, args.batch, workers, pin_memory, args.seed + 1, shuffle=False)

    model_config = {
        "model_name": args.model,
        "out_indices": list(OUT_INDICES),
        "fpn_channels": args.fpn_channels,
        "anchor_sizes": list(anchor_sizes),
        "anchor_ratios": list(anchor_ratios),
        "imgsz": args.imgsz,
        "stretch_to_input_size": args.stretch_to_input_size,
        "feature_channels": backbone.feature_channels,
        "feature_reductions": backbone.feature_reductions,
    }
    parameters = {
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "workers": workers,
        "amp": amp_enabled,
        "amp_dtype": args.amp_dtype if amp_enabled else None,
        "device": str(device),
        "seed": args.seed,
        "deterministic": args.deterministic,
        "pretrained": args.pretrained,
        "score_threshold": args.score_threshold,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps({"model_config": model_config, "data_config": data_config, "parameters": parameters}, indent=2),
        encoding="utf-8",
    )

    best_score = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    print(
        f"training {args.model}; classes={train_set.class_names}; images="
        f"{len(train_set)}/{len(val_set)}; device={device}; imgsz={args.imgsz}"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, amp_enabled, amp_dtype)
        metrics = evaluate(model, val_loader, train_set.class_names, device, args.score_threshold)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, **metrics, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(
            f"epoch={epoch:03d} loss={train_loss:.5f} map50={metrics['map50']:.5f} "
            f"precision50={metrics['precision50']:.5f} recall50={metrics['recall50']:.5f}"
        )
        score = float(metrics["map50"])
        if score > best_score + 1e-8:
            best_score = score
            epochs_without_improvement = 0
            save_checkpoint(
                output / "best.pt", model, optimizer, epoch, train_set.class_names,
                args.model, data_config, model_config, metrics,
            )
        else:
            epochs_without_improvement += 1
        save_checkpoint(
            output / "last.pt", model, optimizer, epoch, train_set.class_names,
            args.model, data_config, model_config, metrics,
        )
        if args.patience and epoch >= args.min_epochs and epochs_without_improvement >= args.patience:
            print(f"early_stop=patience({args.patience})")
            break

    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        if history:
            fieldnames = sorted({key for row in history for key in row if not isinstance(row[key], dict)})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({key: value for key, value in row.items() if key in fieldnames} for row in history)

    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    best_metrics = evaluate(model, val_loader, train_set.class_names, device, args.score_threshold)
    print(f"best_epoch={best['epoch']} best_map50={best_metrics['map50']:.5f}")
    if test_set is not None:
        test_loader = make_loader(test_set, args.batch, 0, pin_memory, args.seed + 2, shuffle=False)
        test_metrics = evaluate(model, test_loader, train_set.class_names, device, args.score_threshold)
        print(
            f"test_map50={test_metrics['map50']:.5f} test_precision50={test_metrics['precision50']:.5f} "
            f"test_recall50={test_metrics['recall50']:.5f}"
        )
        (output / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    metadata = {
        "task": "object_detection",
        "model_name": args.model,
        "classes": train_set.class_names,
        "model_config": model_config,
        "data_config": data_config,
        "training_result": {"best_epoch": int(best["epoch"]), "best_val_metrics": best_metrics},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
