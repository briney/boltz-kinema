"""Training entry point using HuggingFace Accelerate.

Usage:
    # Self-contained config (includes all shared params):
    accelerate launch scripts/train.py --config configs/train_phase0.yaml

    # Base config + phase-specific overrides:
    accelerate launch scripts/train.py --base-config configs/base.yaml --config configs/train_phase0.yaml

    # Multi-GPU:
    accelerate launch --num_processes 4 scripts/train.py --config configs/train_phase0.yaml

    # CLI overrides (applied last, highest priority):
    accelerate launch scripts/train.py --config configs/train_phase0.yaml lr=2e-4 max_steps=50000
"""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BoltzKinema training with HuggingFace Accelerate"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., configs/train_phase0.yaml)",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default=None,
        help="Optional shared base config that --config overrides (e.g., configs/base.yaml)",
    )

    # Parse known args; remaining args are OmegaConf CLI overrides
    args, overrides = parser.parse_known_args()

    # Build config with inheritance: base → phase → CLI overrides
    if args.base_config:
        base_cfg = OmegaConf.load(args.base_config)
        phase_cfg = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(base_cfg, phase_cfg)
    else:
        cfg = OmegaConf.load(args.config)

    # Apply CLI overrides (e.g., lr=2e-4 max_steps=50000)
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # Import here to avoid slow imports before arg parsing
    from boltzkinema.training.trainer import train

    train(cfg)


if __name__ == "__main__":
    main()
