from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from vision_workflows.backends.timm_classification import TimmClassificationBackend


class RobustTransform:
    def __init__(self, image_size: int):
        import torchvision.transforms as transforms

        self.augmentation = TimmClassificationBackend._albumentations_augmentation(True, True, "robust")
        self.preprocessing = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, image):
        return self.preprocessing(self.augmentation(image))


class ImageCacheDataset(Dataset):
    def __init__(self, paths: tuple[Path, ...], mode: str, image_size: int, transform, cache_dir: Path | None):
        self.paths = paths
        self.mode = mode
        self.image_size = image_size
        self.transform = transform
        self.cache_dir = cache_dir
        self.images = None
        if mode == "ram_decoded":
            self.images = tuple(self._read(path) for path in paths)
        elif mode == "ram_resized":
            self.images = tuple(self._read(path).resize((image_size, image_size), Image.Resampling.BILINEAR) for path in paths)
        elif mode == "disk_resized_npy":
            if cache_dir is None:
                raise ValueError("disk cache requires a cache directory")
            cache_dir.mkdir(parents=True, exist_ok=True)
            for index, path in enumerate(paths):
                target = cache_dir / f"{index:06d}.npy"
                if not target.exists():
                    image = self._read(path).resize((image_size, image_size), Image.Resampling.BILINEAR)
                    np.save(target, np.asarray(image, dtype=np.uint8))
        elif mode != "raw":
            raise ValueError(f"unsupported cache mode: {mode}")

    @staticmethod
    def _read(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def _base_image(self, index: int) -> Image.Image:
        if self.mode in {"ram_decoded", "ram_resized"}:
            return self.images[index]
        if self.mode == "disk_resized_npy":
            return Image.fromarray(np.load(self.cache_dir / f"{index:06d}.npy"), mode="RGB")
        return self._read(self.paths[index])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        return self.transform(self._base_image(index)), 0


def _measure(dataset: Dataset, workers: int, batch_size: int, epochs: int, device: torch.device) -> dict[str, float | int]:
    options = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
    }
    if workers > 0:
        options.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(dataset, **options)
    times: list[float] = []
    sample_count = 0
    for _ in range(epochs):
        started = time.perf_counter()
        samples = 0
        for images, _ in loader:
            if device.type == "cuda":
                images = images.to(device, non_blocking=True)
            samples += len(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - started)
        sample_count = samples
    del loader
    warm = times[1:] or times
    warm_seconds = sum(warm) / len(warm)
    return {
        "first_epoch_seconds": times[0],
        "warm_epoch_seconds": warm_seconds,
        "warm_images_per_second": sample_count / warm_seconds,
        "images": sample_count,
        "epochs": epochs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare image decode/cache modes for classification input throughput")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    paths = tuple(sorted(args.data.glob("*.png")))
    if not paths:
        raise FileNotFoundError(f"no PNG images found under {args.data}")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    transform = RobustTransform(args.image_size)
    output = {
        "data": str(args.data),
        "images": len(paths),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "device": str(device),
        "cases": [],
    }
    with tempfile.TemporaryDirectory(prefix="vision-cache-benchmark-") as temporary:
        temporary_path = Path(temporary)
        for mode in ("raw", "ram_decoded", "ram_resized", "disk_resized_npy"):
            cache_dir = temporary_path / mode if mode == "disk_resized_npy" else None
            started = time.perf_counter()
            dataset = ImageCacheDataset(paths, mode, args.image_size, transform, cache_dir)
            build_seconds = time.perf_counter() - started
            cache_bytes = sum(path.stat().st_size for path in cache_dir.glob("*.npy")) if cache_dir else 0
            for workers in args.workers:
                result = _measure(dataset, workers, args.batch_size, args.epochs, device)
                output["cases"].append({
                    "mode": mode,
                    "workers": workers,
                    "cache_build_seconds": build_seconds,
                    "cache_bytes": cache_bytes,
                    **result,
                })
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
