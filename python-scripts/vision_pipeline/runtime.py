"""Runtime helpers shared by adapters without importing ML libraries."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int


def repository_root() -> Path:
    """Return the checkout root when the package is used from this repository."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError(
            "The unified workflow CLI must run from the repository package; "
            f"could not find pyproject.toml below {root}."
        )
    return root


def legacy_script(relative_path: str) -> Path:
    path = repository_root() / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Registered workflow script does not exist: {path}")
    return path


def run_script(relative_path: str, arguments: Sequence[str]) -> CommandResult:
    """Run one legacy backend script in the current uv-managed environment.

    Keeping this boundary as a subprocess is intentional for the first migration:
    it isolates incompatible ML libraries and preserves each backend's current
    lifecycle, logging, and checkpoint behavior while the shared package owns
    dispatch and request normalization.
    """
    script = legacy_script(relative_path)
    command = (sys.executable, str(script), *tuple(str(value) for value in arguments))
    environment = os.environ.copy()
    completed = subprocess.run(command, cwd=repository_root(), env=environment, check=False)
    return CommandResult(command, completed.returncode)
