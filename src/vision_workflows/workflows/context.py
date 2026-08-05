from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..domain.errors import BackendUnavailableError


EventSink = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class WorkflowContext:
    run_dir: Path
    emit: EventSink
    device: str

    def write_json(self, name: str, value: Any) -> Path:
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
        return path


def optional_import(module: str):
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise BackendUnavailableError(
            f"optional dependency for {module!r} is not installed"
        ) from error
