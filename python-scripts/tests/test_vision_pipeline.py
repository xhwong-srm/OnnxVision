from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from vision_pipeline.cli import main
from vision_pipeline.models import ExportRequest, TrainRequest
from vision_pipeline.registry import export, train
from vision_pipeline.runtime import CommandResult


ROOT = Path(__file__).resolve().parents[2]


class VisionPipelineRegistryTests(unittest.TestCase):
    @patch("vision_pipeline.registry.run_script")
    def test_yolo_training_request_maps_to_legacy_cli(self, run_script):
        run_script.return_value = CommandResult(("python", "script"), 0)

        result = train(TrainRequest(
            backend="ultralytics",
            task="detection",
            model="yolo26n",
            data=ROOT / "data.yaml",
            output=ROOT / "runs" / "example",
            imgsz=640,
            epochs=2,
            deterministic=False,
            run_test=False,
        ))

        self.assertEqual(result.returncode, 0)
        script, arguments = run_script.call_args.args
        self.assertEqual(script, "python-scripts/yolo/train_yolo26_detection.py")
        self.assertEqual(arguments[:6], ("--data", str(ROOT / "data.yaml"), "--model", "n", "--output", str(ROOT / "runs" / "example")))
        self.assertIn("--resolution", arguments)
        self.assertIn("--no-deterministic", arguments)
        self.assertIn("--no-run-test", arguments)

    @patch("vision_pipeline.registry.run_script")
    def test_libreyolo_export_maps_family_and_embedded_preprocessing(self, run_script):
        run_script.return_value = CommandResult(("python", "script"), 0)

        export(ExportRequest(
            backend="libreyolo",
            task="detection",
            model="picodet-s",
            checkpoint=ROOT / "best.pt",
            output=ROOT / "detector.onnx",
            imgsz=320,
            embedded_preprocessing=False,
            validation_split="test",
        ))

        script, arguments = run_script.call_args.args
        self.assertEqual(script, "python-scripts/libreyolo/export_picodet_detection.py")
        self.assertEqual(arguments[:6], ("--checkpoint", str(ROOT / "best.pt"), "--output", str(ROOT / "detector.onnx"), "--resolution", "320"))
        self.assertIn("--skip-embedded-preprocessing", arguments)
        self.assertIn("--validation-split", arguments)

    def test_direct_cli_lists_all_current_workflows(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["list-models"]), 0)
        text = output.getvalue()
        self.assertIn("timm/classification/classification", text)
        self.assertIn("ultralytics/detection/yolo26", text)
        self.assertIn("libreyolo/detection/picodet", text)


if __name__ == "__main__":
    unittest.main()
