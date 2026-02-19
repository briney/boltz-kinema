"""Tests for inference config schema handling."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
from omegaconf import OmegaConf


def _load_generate_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "generate.py"
    spec = importlib.util.spec_from_file_location("generate_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_build_config = _load_generate_module()._build_config


def test_build_config_accepts_valid_keys() -> None:
    cfg = OmegaConf.create(
        {
            "mode": "equilibrium",
            "fine_dt_ns": 0.2,
            "generation_window": 12,
        }
    )
    parsed = _build_config(cfg, strict=True)
    assert parsed.mode == "equilibrium"
    assert parsed.fine_dt_ns == pytest.approx(0.2)
    assert parsed.generation_window == 12


def test_build_config_rejects_unknown_keys_in_strict_mode() -> None:
    cfg = OmegaConf.create({"mode": "equilibrium", "bogus_key": 123})
    with pytest.raises(ValueError, match="Unknown inference config keys"):
        _build_config(cfg, strict=True)


def test_build_config_supports_legacy_aliases() -> None:
    cfg = OmegaConf.create(
        {
            "coarse_dt_ns": 10.0,
            "coarse_n_frames": 8,
            "fine_interp_factor": 4,
        }
    )
    parsed = _build_config(cfg, strict=True)
    assert parsed.generation_window == 8
    assert parsed.fine_dt_ns == pytest.approx(2.5)
