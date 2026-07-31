"""Convert a Dataset Builder COCO detection export to YOLO format.

LibreYOLO and Ultralytics consume the same YOLO dataset layout, so this entry
point deliberately delegates to the repository's single converter
implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path


YOLO_SCRIPTS = Path(__file__).resolve().parents[1] / "yolo"
sys.path.insert(0, str(YOLO_SCRIPTS))

from convert_coco_to_yolo import main  # noqa: E402


if __name__ == "__main__":
    main()
