"""Dataset behavior tests."""

from __future__ import annotations

import numpy as np
import torch

from boltzkinema.data.dataset import SystemInfo


def _make_system(tmp_path, *, include_observed_mask: bool) -> tuple[SystemInfo, str]:
    coords_path = tmp_path / "coords.npz"
    ref_path = tmp_path / "ref.npz"

    payload = {
        "coords": np.zeros((3, 4, 3), dtype=np.float32),
    }
    if include_observed_mask:
        payload["observed_atom_mask"] = np.array([True, False, True, True], dtype=np.bool_)
    np.savez(coords_path, **payload)

    np.savez(
        ref_path,
        mol_types=np.zeros(4, dtype=np.int64),
        residue_indices=np.arange(4, dtype=np.int64),
    )

    system = SystemInfo(
        system_id="sys",
        dataset="toy",
        n_frames=3,
        n_atoms=4,
        n_tokens=2,
        frame_dt_ns=0.1,
        split="train",
        coords_path=str(coords_path),
        trunk_cache_dir="unused",
        ref_path=str(ref_path),
    )
    return system, str(coords_path)


def test_observed_atom_mask_loaded_from_coords_npz(tmp_path) -> None:
    system, _ = _make_system(tmp_path, include_observed_mask=True)
    mask = system.observed_atom_mask
    expected = torch.tensor([True, False, True, True])
    assert torch.equal(mask, expected)


def test_observed_atom_mask_falls_back_to_all_ones_when_missing(tmp_path) -> None:
    system, _ = _make_system(tmp_path, include_observed_mask=False)
    mask = system.observed_atom_mask
    assert torch.equal(mask, torch.ones(4, dtype=torch.bool))


def test_coords_npz_metadata_is_cached(tmp_path, monkeypatch) -> None:
    system, coords_path = _make_system(tmp_path, include_observed_mask=True)

    original_load = np.load
    calls = 0

    def wrapped_load(path, *args, **kwargs):
        nonlocal calls
        if str(path) == coords_path:
            calls += 1
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", wrapped_load)

    _ = system.observed_atom_mask
    _ = system.observed_atom_mask
    _ = system.load_coords([0, 1])

    assert calls == 1
