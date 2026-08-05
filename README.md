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
`onnx-vision-object-detection`. Each artifact embeds `vision_task`,
`contract_name`, `contract_version`, and `names`; detection artifacts also
embed `nms_required`. Contract versions use `major.minor.micro` format, for
example `1.0.0`.
