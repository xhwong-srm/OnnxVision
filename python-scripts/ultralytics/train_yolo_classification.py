"""Train a YOLO26 image-classification model with Ultralytics."""

from argparse import ArgumentParser
from pathlib import Path

import torchvision.transforms as transforms
from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer


class FullFrameClassificationDataset(ClassificationDataset):
    """Resize without cropping so off-center objects remain visible."""

    def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
        super().__init__(root, args, augment, prefix)

        self.torch_transforms = transforms.Compose(
            [
                transforms.Resize((args.imgsz, args.imgsz), antialias=True),
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
                )
                if augment
                else transforms.Lambda(lambda image: image),
                transforms.ToTensor(),
            ]
        )


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
        default=Path("images/seal_dataset_v2"),
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
        name="yolo26-seal-260721",
    )


if __name__ == "__main__":
    main()
