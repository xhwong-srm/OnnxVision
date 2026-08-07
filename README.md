# Vision Workflows

This project provides reusable dataset and model workflow services for computer
vision. The `vision_workflows` package is the logic layer; the CLI is only one
adapter over those services. The architecture is task-oriented and can grow
from classification and rectangular detection to segmentation and other vision
tasks without coupling the core services to a particular interface.

```powershell
uv run vision-workflows framework list
uv run vision-workflows model list --task classification --framework ultralytics
uv run vision-workflows dataset --help
uv run vision-workflows train --help
uv run vision-workflows export --help
```

Dataset operations can be used directly from Python:

```python
from pathlib import Path

from vision_workflows.api import ConvertDatasetRequest, DatasetService
from vision_workflows.domain.datasets import DatasetFormat, MaterializationMode

result = DatasetService().convert(ConvertDatasetRequest(
    source=Path("images/source"),
    output=Path("artifacts/yolo"),
    output_format=DatasetFormat.YOLO,
    materialization=MaterializationMode.COPY,
))
print(result.output)
```

Framework integrations are imported lazily. Install only the extras needed by
the host, for example `uv sync --extra test --extra timm`.

Models are selected explicitly by task, framework, and canonical model ID. The
framework/task plugin resolves that ID to its native checkpoint or architecture:

```powershell
uv run vision-workflows train --task classification --framework ultralytics `
  --model yolo26n --data images/seal_pocket_v1 `
  --output runs/ultralytics/yolo26-cls-n-v1
```

Operation parameters are generated from the selected plugin. Use all three
selectors with `--help` to see only the flags accepted by that operation:

```powershell
uv run vision-workflows train --task classification --framework ultralytics --model yolo26n --help
uv run vision-workflows model describe --task classification --framework ultralytics --model yolo26n --operation train
```

Timm classification and detection training support DataLoader and AMP options:

```powershell
uv run vision-workflows train --task classification --framework timm --model mobilenetv4_conv_small_050.e3000_r224_in1k `
  --data data --output run --workers 4 --prefetch-factor 2 `
  --persistent-workers --pin-memory --amp --amp-dtype float16
```

`prefetch_factor` and `persistent_workers` require a positive `--workers` value.
The same settings can be supplied through `TrainRequest.parameters` when using
the Python API. Unknown parameters are rejected before a run starts.

ONNX exports are self-describing. Classification artifacts use the
`onnx-vision-classification` contract, while object-detection artifacts use
`onnx-vision-object-detection`. Each deployment export emits separate
`-bw8.onnx` and `-c24.onnx` artifacts. Both use embedded preprocessing: BW8 is
`uint8[B,1,H,W]` and C24 is `uint8[B,H,W,3]` raw BGR. Batch is dynamic by
default; pass `--batch-size N` to export a fixed-batch model. Classification
outputs are categorical probabilities in `float32[B,C]` (finite values in
`[0,1]`, with each row summing to 1); detection outputs are normalized
`xyxy` `boxes float32[B,Q,4]`, `scores float32[B,Q]`, and `class_ids int64[B,Q]`.
Each artifact embeds `vision_task`, `contract_name`, `contract_version`,
`inputs`, `outputs`, `input_variant`, and `names`; detection artifacts also
embed `nms_required`. Detection rows with score `0` are padding and are ignored;
positive-score rows must contain valid class IDs and ordered normalized boxes.
When `export --data DATASET` is supplied, the export also evaluates the `val`
split with the native checkpoint and both wrapped artifacts. Classification
reports accuracy, loss, prediction agreement, and probability error; detection
reports native provider metrics plus contract-based BW8/C24 mAP50, precision,
recall, F1, and prediction agreement in `dataset-validation.json`.
Provider-owned exports use confidence `0` and IoU `0.7`; consumer-side
class-aware NMS runs at IoU `0.7` only when `nms_required=true`.
LibreYOLO PicoDet exports use a family-specific embedded-NMS adapter and
require `--batch-size 1`; PicoDet-s/m/l default to their native 320/416/640
export sizes.
Contract consumers accept any valid `2.x.y` version and ignore additive unknown
metadata; incompatible semantic changes require contract major version 3.
