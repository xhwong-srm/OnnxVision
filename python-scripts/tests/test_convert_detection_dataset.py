from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convert_detection_dataset import (  # noqa: E402
    load_coco,
    load_rfdetr,
    load_yolo,
    write_coco,
    write_rfdetr,
    write_yolo,
)


class DetectionDatasetConverterTests(unittest.TestCase):
    def test_coco_yolo_rfdetr_coco_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coco = root / "coco"
            self._write_coco_source(coco)

            loaded_coco = load_coco(coco)
            self.assertEqual(loaded_coco.classes, ["seal", "defect"])
            self.assertEqual([len(loaded_coco.splits[split]) for split in ("train", "val", "test")], [1, 1, 1])

            yolo = root / "yolo"
            yolo.mkdir()
            write_yolo(loaded_coco, yolo, "copy")
            loaded_yolo = load_yolo(yolo)

            rfdetr = root / "rfdetr"
            rfdetr.mkdir()
            write_rfdetr(loaded_yolo, rfdetr, "copy")
            loaded_rfdetr = load_rfdetr(rfdetr)

            output_coco = root / "output-coco"
            output_coco.mkdir()
            write_coco(loaded_rfdetr, output_coco, "copy")
            round_trip = load_coco(output_coco)

            self.assertEqual(round_trip.classes, ["seal", "defect"])
            self.assertEqual(
                {split: len(images) for split, images in round_trip.splits.items()},
                {"train": 1, "val": 1, "test": 1},
            )
            train_detection = round_trip.splits["train"][0].detections[0]
            self.assertEqual(train_detection.class_id, 0)
            self.assertAlmostEqual(train_detection.x1, 10.0)
            self.assertAlmostEqual(train_detection.y1, 20.0)
            self.assertAlmostEqual(train_detection.x2, 50.0)
            self.assertAlmostEqual(train_detection.y2, 60.0)

    @staticmethod
    def _write_coco_source(root: Path) -> None:
        categories = {
            "train": [{"id": 20, "name": "defect"}, {"id": 10, "name": "seal"}],
            "val": [{"id": 9, "name": "defect"}, {"id": 3, "name": "seal"}],
            "test": [{"id": 1, "name": "seal"}, {"id": 2, "name": "defect"}],
        }
        for split in ("train", "val", "test"):
            image_dir = root / "images" / split
            annotation_dir = root / "annotations"
            image_dir.mkdir(parents=True, exist_ok=True)
            annotation_dir.mkdir(parents=True, exist_ok=True)
            image_name = f"{split}.png"
            Image.new("RGB", (100, 80), color=(20, 30, 40)).save(image_dir / image_name)
            if split == "train":
                annotations = [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 10,
                        "bbox": [10, 20, 40, 40],
                    }
                ]
            else:
                annotations = []
            document = {
                "images": [
                    {
                        "id": 1,
                        "file_name": f"images/{split}/{image_name}",
                        "width": 100,
                        "height": 80,
                    }
                ],
                "annotations": annotations,
                "categories": categories[split],
            }
            (annotation_dir / f"instances_{split}.json").write_text(
                json.dumps(document), encoding="utf-8"
            )


if __name__ == "__main__":
    unittest.main()
