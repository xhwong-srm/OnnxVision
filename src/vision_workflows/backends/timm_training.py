from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import random
from typing import Any, Mapping


def seed_everything(torch, seed: int, deterministic: bool) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = not deterministic
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = not deterministic


def capture_rng_state(torch, loader_generator) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "loader": loader_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return state


def restore_rng_state(torch, loader_generator, state: dict[str, Any]) -> None:
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    if state.get("loader") is not None:
        loader_generator.set_state(state["loader"])
    if state.get("numpy") is not None:
        try:
            import numpy as np
            np.random.set_state(state["numpy"])
        except ImportError:
            pass


def worker_seed(worker_id: int) -> None:
    import torch
    worker_seed_value = torch.initial_seed() % (2**32)
    random.seed(worker_seed_value)
    try:
        import numpy as np
        np.random.seed(worker_seed_value)
    except ImportError:
        pass


def _bool_option(options: Mapping[str, Any], name: str, default: bool) -> bool:
    value = options.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.casefold().strip()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean")


def _prefetch_factor(options: Mapping[str, Any]) -> int | None:
    value = options.get("prefetch_factor")
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("prefetch_factor must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("prefetch_factor must be a positive integer") from error
    if result <= 0 or str(value).strip() != str(result):
        raise ValueError("prefetch_factor must be a positive integer")
    return result


def _amp_dtype(options: Mapping[str, Any]) -> str | None:
    value = options.get("amp_dtype")
    if value is None:
        return None
    normalized = str(value).casefold().strip().removeprefix("torch.")
    aliases = {
        "float16": "float16",
        "fp16": "float16",
        "half": "float16",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError("amp_dtype must be one of: float16, bfloat16") from error


@dataclass(frozen=True)
class TimmTrainingOptions:
    prefetch_factor: int | None = None
    persistent_workers: bool = False
    pin_memory: bool = False
    amp: bool = False
    amp_dtype: str | None = None
    compile: bool = False

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any]) -> TimmTrainingOptions:
        return cls(
            prefetch_factor=_prefetch_factor(options),
            persistent_workers=_bool_option(options, "persistent_workers", False),
            pin_memory=_bool_option(options, "pin_memory", False),
            amp=_bool_option(options, "amp", False),
            amp_dtype=_amp_dtype(options),
            compile=_bool_option(options, "compile", False),
        )

    def data_loader_kwargs(self, workers: int) -> dict[str, Any]:
        if workers < 0:
            raise ValueError("workers must be non-negative")
        if workers == 0 and self.prefetch_factor is not None:
            raise ValueError("prefetch_factor requires workers > 0")
        if workers == 0 and self.persistent_workers:
            raise ValueError("persistent_workers requires workers > 0")

        kwargs: dict[str, Any] = {
            "num_workers": workers,
            "persistent_workers": self.persistent_workers,
            "pin_memory": self.pin_memory,
        }
        if self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def _torch_amp_dtype(self, torch, device):
        name = self.amp_dtype or ("float16" if device.type == "cuda" else "bfloat16")
        return getattr(torch, name)

    def autocast(self, torch, device):
        if not self.amp:
            return nullcontext()
        return torch.autocast(
            device_type=device.type,
            dtype=self._torch_amp_dtype(torch, device),
            enabled=True,
        )

    def grad_scaler(self, torch, device):
        if not self.amp or device.type not in {"cuda", "xpu", "hpu", "mtia", "maia"}:
            return None
        return torch.amp.GradScaler(device=device.type, enabled=True)

    @staticmethod
    def backward_step(loss, optimizer, scaler) -> None:
        if scaler is None:
            loss.backward()
            optimizer.step()
            return
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
