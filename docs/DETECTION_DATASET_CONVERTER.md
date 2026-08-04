# Object-detection dataset conversion

`python-scripts/convert_detection_dataset.py` converts between the four
dataset layouts used in this repository:

- `coco`: `images/{train,val,test}` plus
  `annotations/instances_{split}.json`
- `yolo`: `data.yaml` plus `images/{train,val,test}` and
  `labels/{train,val,test}`
- `rfdetr`: `train/`, `valid/`, and optional `test/`, each containing
  `_annotations.coco.json` and its images
- `neurocle`: a labeling JSON plus an image ZIP or loose images under
  `images/`

The converter reads one format into a canonical bounding-box representation,
then writes the requested format. Class IDs are zero-based in YOLO and
one-based in generated COCO/RF-DETR JSON. Class names, split membership, image
dimensions, and bounding boxes are preserved. Segmentation polygons,
keypoints, and other format-specific fields are intentionally not converted.

## Usage

```powershell
uv run python python-scripts\convert_detection_dataset.py `
  --output-format yolo `
  --data images\seal_dataset --output artifacts\seal-yolo

uv run python python-scripts\convert_detection_dataset.py `
  --input-format yolo --output-format rfdetr `
  --data artifacts\seal-yolo --output artifacts\seal-rfdetr

uv run python python-scripts\convert_detection_dataset.py `
  --input-format rfdetr --output-format coco `
  --data artifacts\seal-rfdetr --output artifacts\seal-coco
```

Images are hard-linked by default. Use `--image-mode copy` when the source and
destination are on different volumes.

## Reusable API

The conversion logic is in `python-scripts\detection_dataset`, separate from
the CLI frontend. Scripts running from `python-scripts` can call it directly:

```python
from pathlib import Path

from detection_dataset import convert_dataset, detect_input_format

data = Path("images/seal_dataset")
print(detect_input_format(data))
result = convert_dataset(data, Path("artifacts/seal-yolo"), "yolo", image_mode="copy")
print(result.split_image_counts)
```

The package exports `detect_input_format`, `load_dataset`, the format-specific
loaders and writers, and `convert_dataset`. The original script remains a
compatibility CLI entry point.

The existing `python-scripts\yolo\convert_coco_to_yolo.py` and
`python-scripts\libreyolo\convert_coco_to_yolo.py` entry points remain
compatible, as does `python-scripts\rf-detr\convert_coco_to_rfdetr.py`; they
now delegate to the unified implementation.
