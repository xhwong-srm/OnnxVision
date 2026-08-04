from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classification_dataset import merge_datasets, parse_group_selection  # noqa: E402
from classification_dataset.cli import build_parser  # noqa: E402


class ClassificationDatasetMergerTests(unittest.TestCase):
    def test_group_selection_parser_and_cli_shape(self) -> None:
        self.assertEqual(parse_group_selection("1-3,7"), {1, 2, 3, 7})

        args = build_parser().parse_args(["source-a", "source-b", "output", "--balance", "none"])
        self.assertEqual(args.source, [Path("source-a"), Path("source-b")])
        self.assertEqual(args.output, Path("output"))
        self.assertEqual(args.balance, "none")

    def test_merge_preserves_existing_splits_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_a = root / "source-a"
            source_b = root / "source-b"
            self._write_sample(source_a / "train" / "seal" / "a.png")
            self._write_sample(source_a / "val" / "seal" / "b.png")
            self._write_sample(source_b / "train" / "seal" / "c.png")
            self._write_sample(source_b / "test" / "seal" / "d.png")

            result = merge_datasets(
                [source_a, source_b], root / "merged", balance="none"
            )

            self.assertEqual(result.split_mode, "preserve")
            self.assertEqual(result.split_counts[("train", "seal")], 2)
            self.assertEqual(result.split_counts[("val", "seal")], 1)
            self.assertEqual(result.split_counts[("test", "seal")], 1)
            self.assertTrue((result.output / "train" / "seal" / "a.png").is_file())
            with result.manifest.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)

    def test_group_by_image_keeps_cross_class_crops_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_a = root / "source-a"
            source_b = root / "source-b"
            self._write_sample(
                source_a / "seal" / "0123456789ab_r0000_g0000_n000.png"
            )
            self._write_sample(
                source_a / "defect" / "0123456789ab_r0001_g0001_n000.png"
            )
            self._write_sample(
                source_b / "seal" / "fedcba987654_r0000_g0000_n000.png"
            )
            self._write_sample(
                source_b / "defect" / "fedcba987654_r0001_g0001_n000.png"
            )

            result = merge_datasets(
                [source_a, source_b],
                root / "merged",
                train=50,
                val=50,
                test=0,
                group_by_image=True,
                balance="none",
            )

            with result.manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for image_id in ("0123456789ab", "fedcba987654"):
                splits = {row["split"] for row in rows if image_id in row["source"]}
                self.assertEqual(len(splits), 1)

    def test_train_groups_are_restricted_and_not_undersampled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_a = root / "source-a"
            source_b = root / "source-b"
            self._write_sample(source_a / "seal" / "aaaaaaaaaaaa_r0000_g0000_n000.png")
            self._write_sample(source_a / "seal" / "aaaaaaaaaaaa_r0001_g0001_n000.png")
            self._write_sample(source_b / "seal" / "bbbbbbbbbbbb_r0000_g0002_n000.png")

            result = merge_datasets(
                [source_a, source_b],
                root / "merged",
                train=50,
                val=50,
                test=0,
                train_groups={1},
            )

            self.assertEqual(result.balance, "none")
            with result.manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            selected_group = [row for row in rows if "_g0001_" in row["source"]]
            other_groups = [row for row in rows if "_g0001_" not in row["source"]]
            self.assertEqual({row["split"] for row in selected_group}, {"train"})
            self.assertTrue({row["split"] for row in other_groups} <= {"val", "test"})

    @staticmethod
    def _write_sample(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"classification sample")


if __name__ == "__main__":
    unittest.main()
