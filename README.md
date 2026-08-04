# Vision Workflows

This project provides reusable dataset and model workflow services for computer
vision. The `vision_workflows` package is the logic layer; the CLI is only one
adapter over those services. The architecture is task-oriented and can grow
from classification and rectangular detection to segmentation and other vision
tasks without coupling the core services to a particular interface.

```powershell
uv run vision-workflows model list
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

Model backends are imported lazily. Install only the backend extras needed by
the host, for example `uv sync --extra test --extra timm`.
