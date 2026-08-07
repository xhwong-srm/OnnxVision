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
the host, for example `uv sync --extra test --extra timm`. Timm classification
can opt into the official `albumentations` package with
`uv sync --extra timm --extra albumentations`; this project intentionally does
not install or import AlbumentationsX. Hyperparameter tuning is available with
`uv sync --extra timm --extra optuna`.

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

Timm classification separates deterministic preprocessing from random
augmentation. Augmentation is enabled by default with the existing standard
flip/rotation policy. Select Albumentations and the stronger lighting/camera
policy explicitly when desired:

```powershell
uv run --extra timm --extra albumentations vision-workflows train `
  --task classification --framework timm `
  --model mobilenetv4_conv_small_050.e3000_r224_in1k `
  --data data --output runs/timm-robust `
  --augmentation-backend albumentations --augmentation-policy robust
```

Use `--no-augmentation` to disable random transforms. The robust policy adds
moderate brightness/contrast, gamma, sensor-noise, and blur variation while
keeping small flips and rotations. Resize is common preprocessing before
random training augmentation; tensor conversion and normalization follow the
augmentation. Validation and evaluation never apply random augmentation. For
timm classification, `--cache disk` or `--cache ram` caches
the resized RGB training images before augmentation; `--cache disk` is the
safer choice for larger datasets. The cache is keyed by image size and
source-file metadata and is rebuilt when the sources change. Ultralytics and
LibreYOLO retain their provider-native augmentation controls.

```powershell
uv run --extra timm vision-workflows train `
  --task classification --framework timm `
  --model mobilenetv4_conv_small_050.e3000_r224_in1k `
  --data data --output runs/timm-cached --cache disk --workers 4
```

Optuna tuning is a first-class timm classification operation. It searches
learning rate and weight decay, reports validation accuracy each validation
epoch for pruning, stores trial summaries in `optuna.json`, and copies the
best trial checkpoints to the requested output. Ultralytics `tune` delegates
to its native tuning API; LibreYOLO does not advertise tuning without a
verified provider implementation.

```powershell
uv run --extra timm --extra optuna vision-workflows tune `
  --task classification --framework timm `
  --model mobilenetv4_conv_small_050.e3000_r224_in1k `
  --data data --output runs/timm-tune --trials 10
```

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
split with the native checkpoint, the exported float ONNX core, and both wrapped
artifacts. `dataset-validation.json` reports these as peer `native`,
`native-export`, `bw8`, and `c24` sections. Classification sections contain
accuracy/loss and parity metrics; detection sections contain contract-based
mAP50, mAP50-95, precision, recall, F1, and IoU metrics. The top-level
`bw8_c24_*` fields compare the two wrapped variants directly. For detection,
`bw8_c24_agreement50` and its current alias `bw8_c24_agreement` are matched
BW8/C24 predictions divided by the larger prediction count per image, where
zero-area, non-finite, and below-confidence candidates are excluded, class IDs
must match, and IoU must be at least 0.5. `bw8_c24_score_mae` is the mean
absolute confidence-score difference for those matched pairs only. The report
also includes explicit `_raw_*` agreement fields with score threshold `0` and
`_deployment_*` fields with score threshold `0.5`, matching the ONNX consumer's
default detection threshold. This applies to both `bw8_c24_*` and the
`bw8_native_export_*`/`c24_native_export_*` comparisons; the unsuffixed fields
remain aliases of the raw agreement.
The export command saves its complete JSON result beside the requested output
using the same stem (`model.onnx` produces `model.json`) and logs that path
instead of printing the result.
Provider-owned exports use confidence `0` and IoU `0.7`; consumer-side
class-aware NMS runs at IoU `0.7` only when `nms_required=true`.
LibreYOLO PicoDet exports use a family-specific embedded-NMS adapter and
require `--batch-size 1`; PicoDet-s/m/l default to their native 320/416/640
export sizes.
Contract consumers accept any valid `2.x.y` version and ignore additive unknown
metadata; incompatible semantic changes require contract major version 3.
