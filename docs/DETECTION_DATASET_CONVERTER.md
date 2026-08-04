# Dataset conversion

Dataset conversion is implemented by `vision_workflows.datasets.DatasetService` and
is available through the CLI or direct Python calls. The canonical detection
representation contains rectangular pixel-space XYXY boxes and zero-based
class IDs.

```powershell
uv run vision-workflows dataset convert `
  images\dataset `
  artifacts\yolo `
  --output-format yolo `
  --materialization copy
```

The input format is detected from structural markers, or can be supplied with
`--input-format`. Supported detection formats are COCO, YOLO, RF-DETR, and
Neurocle. Classification image-folder datasets can be converted only to
image-folder output.

```python
from pathlib import Path

from vision_workflows.api import ConvertDatasetRequest, DatasetService
from vision_workflows.domain.datasets import DatasetFormat, MaterializationMode

result = DatasetService().convert(ConvertDatasetRequest(
    source=Path("images/dataset"),
    output=Path("artifacts/yolo"),
    output_format=DatasetFormat.YOLO,
    materialization=MaterializationMode.COPY,
))
print(result.split_counts)
```

Conversion is staged and finalized atomically. The output contains a
`dataset_manifest.json` with classes, split counts, source provenance, and
materialization settings. Unsupported annotation types are rejected instead of
being silently discarded.
