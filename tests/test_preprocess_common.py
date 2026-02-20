"""Tests for shared preprocessing helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kinematic.data.preprocess_common import (
    collect_manifest_entries,
    finalize_processed_system,
    write_manifest_entries,
)


class _DummyTraj:
    def __init__(self) -> None:
        self.xyz = np.array(
            [
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[1.5, 0.0, 0.0], [2.5, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )
        self.timestep = 20.0  # ps
        self.n_frames = 2
        self.n_atoms = 2


def test_finalize_processed_system_builds_expected_manifest_entry(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    def fake_build_observation_mask(coords):
        calls["coords_shape"] = coords.shape
        return np.array([True, True], dtype=np.bool_)

    def fake_convert_trajectory(traj, system_id, output_dir, observed_mask):
        calls["convert"] = (traj.n_frames, system_id, Path(output_dir), observed_mask.tolist())
        return Path(output_dir) / f"{system_id}_coords.npz"

    def fake_extract_atom_metadata(_traj):
        return [
            {"residue_index": 0},
            {"residue_index": 1},
        ]

    def fake_save_reference_structure(atoms, ref_coords_A, output_path):
        calls["ref"] = (len(atoms), ref_coords_A.shape, Path(output_path))
        return Path(output_path)

    import kinematic.data.preprocess_common as pc_mod

    monkeypatch.setattr(pc_mod, "build_observation_mask", fake_build_observation_mask)
    monkeypatch.setattr(pc_mod, "convert_trajectory", fake_convert_trajectory)
    monkeypatch.setattr(pc_mod, "extract_atom_metadata", fake_extract_atom_metadata)
    monkeypatch.setattr(pc_mod, "save_reference_structure", fake_save_reference_structure)

    out = finalize_processed_system(
        system_id="sys1",
        dataset="toy",
        traj=_DummyTraj(),
        output_dir=tmp_path / "coords",
        ref_dir=tmp_path / "refs",
        frame_dt_ns=0.02,
    )

    assert calls["coords_shape"] == (2, 3)
    assert out["system_id"] == "sys1"
    assert out["dataset"] == "toy"
    assert out["n_frames"] == 2
    assert out["n_atoms"] == 2
    assert out["n_tokens"] == 2
    assert out["frame_dt_ns"] == 0.02
    assert out["split"] == "train"
    assert out["trunk_cache_dir"] == ""
    assert str(out["coords_path"]).endswith("sys1_coords.npz")
    assert str(out["ref_path"]).endswith("sys1_ref.npz")


def test_collect_manifest_entries_filters_failed_systems() -> None:
    systems = [{"id": 1}, {"id": 2}, {"id": 3}]

    def preprocess_fn(system):
        if system["id"] == 2:
            return None
        return {"id": system["id"]}

    out = collect_manifest_entries(systems, preprocess_fn)
    assert out == [{"id": 1}, {"id": 3}]


def test_write_manifest_entries_writes_json(tmp_path) -> None:
    entries = [{"system_id": "a"}, {"system_id": "b"}]
    path = tmp_path / "manifest.json"

    out_path = write_manifest_entries(entries, path)

    assert out_path == path
    with open(path) as f:
        loaded = json.load(f)
    assert loaded == entries
