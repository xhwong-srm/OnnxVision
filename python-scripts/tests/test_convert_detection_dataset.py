from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convert_detection_dataset import (  # noqa: E402
    build_parser,
    detect_input_format,
    load_coco,
    load_neurocle,
    load_rfdetr,
    load_yolo,
    main,
    write_coco,
    write_neurocle,
    write_rfdetr,
    write_yolo,
)


class DetectionDatasetConverterTests(unittest.TestCase):
    def test_coco_yolo_rfdetr_coco_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coco = root / "coco"
            self._write_coco_source(coco)
            self.assertEqual(detect_input_format(coco), "coco")

            loaded_coco = load_coco(coco)
            self.assertEqual(loaded_coco.classes, ["seal", "defect"])
            self.assertEqual([len(loaded_coco.splits[split]) for split in ("train", "val", "test")], [1, 1, 1])

            neurocle = root / "neurocle"
            neurocle.mkdir()
            write_neurocle(loaded_coco, neurocle, "copy")
            neurocle_document = json.loads(
                (neurocle / "neurocle_labeling.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [record["set"] for record in neurocle_document["data"]],
                ["train", "train", "test"],
            )

            yolo = root / "yolo"
            yolo.mkdir()
            write_yolo(loaded_coco, yolo, "copy")
            self.assertEqual(detect_input_format(yolo), "yolo")
            self.assertEqual(detect_input_format(yolo / "data.yaml"), "yolo")
            loaded_yolo = load_yolo(yolo)

            rfdetr = root / "rfdetr"
            rfdetr.mkdir()
            write_rfdetr(loaded_yolo, rfdetr, "copy")
            self.assertEqual(detect_input_format(rfdetr), "rfdetr")
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

    def test_neurocle_json_and_zip_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            neurocle = root / "neurocle"
            neurocle.mkdir()
            source_images = root / "source-images"
            source_images.mkdir()
            records = []
            for split, image_name, x in (
                ("train", "train-image.png", 10),
                ("test", "test-image.png", 20),
            ):
                image_path = source_images / image_name
                Image.new("RGB", (100, 80), color=(20, 30, 40)).save(image_path)
                records.append(
                    {
                        "fileName": image_name,
                        "set": split,
                        "classLabel": "",
                        "regionLabel": [
                            {
                                "className": "Core",
                                "type": "Rect",
                                "x": x,
                                "y": 20,
                                "width": 40,
                                "height": 30,
                            }
                        ],
                        "width": 100,
                        "height": 80,
                    }
                )

            (neurocle / "labels.json").write_text(
                json.dumps({"classes": {"name": "Core"}, "data": records}),
                encoding="utf-8",
            )
            with zipfile.ZipFile(neurocle / "images.zip", "w") as archive:
                for record in records:
                    archive.write(source_images / record["fileName"], arcname=record["fileName"])

            dataset = load_neurocle(neurocle)
            try:
                self.assertEqual(dataset.classes, ["Core"])
                self.assertEqual(
                    {split: len(images) for split, images in dataset.splits.items()},
                    {"train": 1, "test": 1},
                )
                detection = dataset.splits["train"][0].detections[0]
                self.assertEqual((detection.x1, detection.y1, detection.x2, detection.y2), (10, 20, 50, 50))
                neurocle_output = root / "neurocle-output"
                neurocle_output.mkdir()
                write_neurocle(dataset, neurocle_output, "copy")
                output = root / "coco"
                output.mkdir()
                write_coco(dataset, output, "copy")
            finally:
                dataset.cleanup()

            train_document = json.loads(
                (output / "annotations" / "instances_train.json").read_text(encoding="utf-8")
            )
            self.assertEqual(train_document["categories"], [{"id": 1, "name": "Core", "supercategory": ""}])
            self.assertEqual(train_document["annotations"][0]["bbox"], [10, 20, 40, 30])
            self.assertTrue((output / "images" / "train" / "train-image.png").is_file())

            written_document = json.loads(
                (neurocle_output / "neurocle_labeling.json").read_text(encoding="utf-8")
            )
            self.assertEqual(written_document["label_type"], "obd")
            self.assertEqual(written_document["classes"][0]["name"], "Core")
            self.assertEqual(written_document["data"][0]["regionLabel"][0]["width"], 40)
            self.assertFalse(any(neurocle_output.glob("*.zip")))
            self.assertTrue((neurocle_output / "images" / "train-image.png").is_file())
            unzipped = load_neurocle(neurocle_output)
            self.assertEqual(detect_input_format(neurocle_output), "neurocle")
            self.assertEqual(
                {split: len(images) for split, images in unzipped.splits.items()},
                {"train": 1, "test": 1},
            )
            unzipped.cleanup()

    def test_input_format_is_optional_when_it_can_be_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coco = root / "coco"
            self._write_coco_source(coco)
            output = root / "yolo"

            arguments = build_parser().parse_args(
                [
                    "--output-format",
                    "yolo",
                    "--data",
                    str(coco),
                    "--output",
                    str(output),
                    "--image-mode",
                    "copy",
                ]
            )
            self.assertIsNone(arguments.input_format)

            main(
                [
                    "--output-format",
                    "yolo",
                    "--data",
                    str(coco),
                    "--output",
                    str(output),
                    "--image-mode",
                    "copy",
                ]
            )
            self.assertTrue((output / "data.yaml").is_file())

    def test_input_format_detection_reports_unknown_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            unknown = Path(temporary) / "unknown"
            unknown.mkdir()
            with self.assertRaisesRegex(ValueError, "Could not detect"):
                detect_input_format(unknown)

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
