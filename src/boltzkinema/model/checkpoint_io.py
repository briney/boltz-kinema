"""Checkpoint loading utilities with suffix-based dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def find_model_weights_file(checkpoint_path: str | Path) -> Path | None:
    """Resolve model weight file from a checkpoint path.

    If ``checkpoint_path`` is a file, return it.
    If it is a directory, search for Accelerate-style weight files.
    """
    path = Path(checkpoint_path)
    if path.is_file():
        return path
    if not path.is_dir():
        return None

    candidates = sorted(path.glob("pytorch_model*.bin")) + sorted(
        path.glob("model*.safetensors")
    )
    if not candidates:
        return None
    return candidates[0]


def load_checkpoint_file(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    weights_only: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint file by extension.

    ``.safetensors`` files are loaded with ``safetensors.torch.load_file``.
    All other files are loaded with ``torch.load``.
    """
    path = Path(checkpoint_path)
    suffix = path.suffix.lower()

    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Checkpoint is a .safetensors file but `safetensors` is not installed. "
                "Install it with `pip install safetensors`."
            ) from exc

        device = str(map_location) if isinstance(map_location, torch.device) else map_location
        return load_file(str(path), device=device)

    state = torch.load(path, map_location=map_location, weights_only=weights_only)
    if not isinstance(state, dict):
        raise TypeError(
            f"Expected checkpoint to deserialize to a dict, got {type(state).__name__}"
        )
    return state


def load_model_state_dict(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load and normalize a model ``state_dict`` from checkpoint path."""
    state = load_checkpoint_file(checkpoint_path, map_location=map_location, weights_only=True)

    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise TypeError(
            f"Expected model state_dict to be a dict, got {type(state).__name__}"
        )
    return state
