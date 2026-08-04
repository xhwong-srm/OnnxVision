# Unified workflows

The package separates workflow logic from interaction. The CLI is one adapter;
other applications can import the same typed services.

```powershell
uv run vision-workflows model list
uv run vision-workflows dataset --help
uv run vision-workflows train --help
```

## Dataset management

```powershell
uv run vision-workflows dataset validate images\dataset --require-train-val
uv run vision-workflows dataset convert images\dataset artifacts\yolo --output-format yolo --materialization copy
uv run vision-workflows dataset split images\classification artifacts\classification-split --grouping image
uv run vision-workflows dataset merge source-a source-b --output artifacts\merged --split
```

Supported formats currently cover image-folder classification, COCO, YOLO,
RF-DETR, and Neurocle rectangular detection. Segmentation is a first-class
task kind in the domain contract and can be added with task-specific mask or
polygon formats and backends.

## Training and export

Models use `BACKEND/FAMILY/VARIANT` identifiers:

```powershell
uv run vision-workflows train `
  --model ultralytics/yolo26/n `
  --data images\dataset\data.yaml `
  --output runs\yolo26-n `
  --epochs 100 `
  --image-size 640 `
  --device 0

uv run vision-workflows export `
  --model libreyolo/yolov9/t `
  --checkpoint runs\yolov9-t\best.pt `
  --output artifacts\yolov9.onnx `
  --image-size 640
```

Validation and held-out testing are separate operations:

```powershell
uv run vision-workflows validate --model ultralytics/yolo26/n --target runs\yolo26-n\best.pt --data images\dataset\data.yaml
uv run vision-workflows test --model ultralytics/yolo26/n --target runs\yolo26-n\best.pt --data images\dataset\data.yaml --split test
```

Every workflow writes a JSON manifest, configuration, event log, metrics, and
artifact references in its run directory.
