from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..domain.results import ArtifactRef, RunManifest, RunStatus
from .context import WorkflowContext


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink() or path.is_junction()


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_junction():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _assert_safe_overwrite_target(path: Path) -> None:
    current = _absolute_path(Path.cwd())
    if path == path.parent or current.is_relative_to(path):
        raise ValueError(f"refusing to overwrite the current directory or one of its ancestors: {path}")


class RunStore:
    def __init__(self, output: Path):
        self.output = output.expanduser().resolve()
        self.run_dir: Path | None = None

    def start(self, operation: str, config: dict[str, Any], *, device: str = "auto", run_dir: Path | None = None, overwrite: bool = False) -> tuple[WorkflowContext, str]:
        self.output.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex
        self.run_dir = _absolute_path(run_dir or self.output / f"{operation}-{run_id[:12]}")
        if _path_exists(self.run_dir) and overwrite:
            _assert_safe_overwrite_target(self.run_dir)
            _remove_path(self.run_dir)
        if _path_exists(self.run_dir) and (self.run_dir.is_symlink() or self.run_dir.is_junction() or not self.run_dir.is_dir() or any(self.run_dir.iterdir())):
            raise FileExistsError(f"run directory is not empty: {self.run_dir}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        events = self.run_dir / "events.jsonl"

        def emit(name: str, values: dict[str, Any]) -> None:
            with events.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": name, **values}, default=str) + "\n")

        context = WorkflowContext(self.run_dir, emit, device)
        context.write_json("config.json", config)
        self._write_manifest(RunManifest(run_id, operation, RunStatus.RUNNING, self.run_dir, config))
        emit("run_started", {"run_id": run_id, "operation": operation})
        return context, run_id

    def finish(
        self,
        run_id: str,
        operation: str,
        config: dict[str, Any],
        *,
        status: RunStatus,
        artifacts: tuple[ArtifactRef, ...] = (),
        metrics: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> RunManifest:
        if self.run_dir is None:
            raise RuntimeError("run has not been started")
        manifest = RunManifest(run_id, operation, status, self.run_dir, config, (), artifacts, metrics or {}, error)
        self._write_manifest(manifest)
        return manifest

    def _write_manifest(self, manifest: RunManifest) -> None:
        payload = asdict(manifest)
        payload["status"] = manifest.status.value
        payload["run_dir"] = str(manifest.run_dir)
        payload["artifacts"] = [
            {"name": artifact.name, "path": str(artifact.path), "kind": artifact.kind, "sha256": artifact.sha256}
            for artifact in manifest.artifacts
        ]
        target = manifest.run_dir / "manifest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def artifact(path: Path, kind: str, name: str | None = None) -> ArtifactRef:
    path = path.expanduser().resolve()
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        checksum = digest.hexdigest()
    else:
        checksum = None
    return ArtifactRef(name or path.name, path, kind, checksum)
