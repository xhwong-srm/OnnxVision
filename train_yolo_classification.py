"""Train a YOLO26 image-classification model with Ultralytics."""

from argparse import ArgumentParser
from pathlib import Path

import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer


class FullFrameClassificationDataset(ClassificationDataset):
    """Resize without cropping so off-center objects remain visible."""

    def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
        super().__init__(root, args, augment, prefix)

        preprocessing = [
            transforms.Resize((args.imgsz, args.imgsz), antialias=True),
        ]
        if augment:
            preprocessing.extend(
                [
                    transforms.RandomHorizontalFlip(p=args.fliplr),
                    transforms.RandomVerticalFlip(p=args.flipud),
                    transforms.ColorJitter(
                        brightness=args.hsv_v,
                        contrast=args.hsv_v,
                        saturation=args.hsv_s,
                        hue=args.hsv_h,
                    ),
                ]
            )
        preprocessing.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor((0.485, 0.456, 0.406)),
                    std=torch.tensor((0.229, 0.224, 0.225)),
                ),
            ]
        )
        self.torch_transforms = transforms.Compose(preprocessing)


class FullFrameClassificationTrainer(ClassificationTrainer):
    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        return FullFrameClassificationDataset(
            root=img_path,
            args=self.args,
            augment=mode == "train",
            prefix=mode,
        )


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("images/seal_dataset"),
        help="Dataset root containing train/val/test class folders",
    )
    parser.add_argument("--model", default="yolo26n-cls.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "classify",
        help="Directory in which training runs are saved",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Training device, for example 0, "cpu", or "mps"',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = args.data.resolve()
    if not (data / "train").is_dir():
        raise FileNotFoundError(f"Missing training directory: {data / 'train'}")

    model = YOLO(args.model)
    model.train(
        trainer=FullFrameClassificationTrainer,
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project.resolve()),
        name="yolo26-seal",
    )


if __name__ == "__main__":
    main()
