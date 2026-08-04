# Unified model workflows

The repository provides one dispatching entrypoint for the current training
and export scripts. `uv run` synchronizes the local project environment; no
global package installation is required.

```powershell
uv run seal-vision list-models
uv run seal-vision train --help
uv run seal-vision export --help
```

The direct-file form is equivalent:

```powershell
uv run python python-scripts\vision_cli.py list-models
```

## Training

The common options are normalized by the dispatcher. Backend-specific options
can be passed after `--` and are forwarded unchanged to the native script.

```powershell
uv run seal-vision train `
  --backend ultralytics `
  --task detection `
  --model yolo26n `
  --data images\seal_dataset\data.yaml `
  --output runs\yolo26-n `
  --epochs 100 `
  --imgsz 640 `
  --device 0
```

For a timm classifier, use a class-folder dataset and the timm model name:

```powershell
uv run seal-vision train `
  --backend timm `
  --task classification `
  --model mobilenetv3_small_100.lamb_in1k `
  --data images\seal_dataset_v2
```

## Export

```powershell
uv run seal-vision export `
  --backend libreyolo `
  --task detection `
  --model yolov9-t `
  --checkpoint runs\yolov9-t\weights\best.pt `
  --output artifacts\yolov9.onnx `
  --imgsz 640 `
  --data images\seal_dataset\data.yaml
```

The dispatcher preserves each backend's existing defaults and output contract.
Model-specific export and validation flags remain available after `--`.
Legacy script paths continue to work unchanged.
