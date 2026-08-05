from __future__ import annotations

import pytest

from vision_workflows.backends.timm_training import TimmTrainingOptions


def test_timm_training_options_build_loader_kwargs() -> None:
    options = TimmTrainingOptions.from_mapping({
        "prefetch_factor": 3,
        "persistent_workers": True,
        "pin_memory": True,
        "amp": True,
        "amp_dtype": "bf16",
    })

    assert options.data_loader_kwargs(4) == {
        "num_workers": 4,
        "prefetch_factor": 3,
        "persistent_workers": True,
        "pin_memory": True,
    }
    assert options.amp is True
    assert options.amp_dtype == "bfloat16"


@pytest.mark.parametrize("name", ["prefetch_factor", "persistent_workers"])
def test_timm_worker_options_require_workers(name: str) -> None:
    value = 2 if name == "prefetch_factor" else True
    options = TimmTrainingOptions.from_mapping({name: value})

    with pytest.raises(ValueError, match="requires workers > 0"):
        options.data_loader_kwargs(0)


def test_timm_training_options_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="prefetch_factor"):
        TimmTrainingOptions.from_mapping({"prefetch_factor": 0})
    with pytest.raises(ValueError, match="pin_memory"):
        TimmTrainingOptions.from_mapping({"pin_memory": "sometimes"})
    with pytest.raises(ValueError, match="amp_dtype"):
        TimmTrainingOptions.from_mapping({"amp_dtype": "float8"})
