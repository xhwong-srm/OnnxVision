# Object-detection dataset conversion

`python-scripts/convert_detection_dataset.py` converts between the three
dataset layouts used in this repository:

- `coco`: `images/{train,val,test}` plus
  `annotations/instances_{split}.json`
- `yolo`: `data.yaml` plus `images/{train,val,test}` and
  `labels/{train,val,test}`
- `rfdetr`: `train/`, `valid/`, and optional `test/`, each containing
  `_annotations.coco.json` and its images

The converter reads one format into a canonical bounding-box representation,
then writes the requested format. Class IDs are zero-based in YOLO and
one-based in generated COCO/RF-DETR JSON. Class names, split membership, image
dimensions, and bounding boxes are preserved. Segmentation polygons,
keypoints, and other format-specific fields are intentionally not converted.

## Usage

```powershell
uv run python python-scripts\convert_detection_dataset.py `
  --input-format coco --output-format yolo `
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

The existing `python-scripts\yolo\convert_coco_to_yolo.py` and
`python-scripts\libreyolo\convert_coco_to_yolo.py` entry points remain
compatible, as does `python-scripts\rf-detr\convert_coco_to_rfdetr.py`; they
now delegate to the unified implementation.
