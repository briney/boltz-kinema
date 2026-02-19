"""Tests for checkpoint loading dispatch helpers."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("boltz")

from safetensors.torch import save_file

from boltzkinema.model.checkpoint_io import (
    find_model_weights_file,
    load_checkpoint_file,
    load_model_state_dict,
)


def test_find_model_weights_file_from_directory(tmp_path) -> None:
    ckpt_dir = tmp_path / "step_1000"
    ckpt_dir.mkdir()
    pt_file = ckpt_dir / "pytorch_model.bin"
    torch.save({"weight": torch.tensor([1.0])}, pt_file)

    out = find_model_weights_file(ckpt_dir)
    assert out == pt_file


def test_load_checkpoint_file_torch_dispatch(tmp_path) -> None:
    path = tmp_path / "model.bin"
    expected = {"weight": torch.tensor([1.0, 2.0, 3.0])}
    torch.save(expected, path)

    loaded = load_checkpoint_file(path)
    assert torch.allclose(loaded["weight"], expected["weight"])


def test_load_checkpoint_file_safetensors_dispatch(tmp_path) -> None:
    path = tmp_path / "model.safetensors"
    expected = {"weight": torch.tensor([3.0, 2.0, 1.0])}
    save_file(expected, str(path))

    loaded = load_checkpoint_file(path)
    assert torch.allclose(loaded["weight"], expected["weight"])


def test_load_model_state_dict_unwraps_state_dict(tmp_path) -> None:
    path = tmp_path / "model.ckpt"
    inner = {"weight": torch.tensor([5.0])}
    torch.save({"state_dict": inner}, path)

    loaded = load_model_state_dict(path)
    assert torch.allclose(loaded["weight"], inner["weight"])
