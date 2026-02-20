"""``kinematic precompute-trunk-embeddings`` subcommand."""

from __future__ import annotations

import logging
from pathlib import Path

import click


@click.command("precompute-trunk-embeddings")
@click.option(
    "--manifest",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to JSON manifest file.",
)
@click.option(
    "--checkpoint",
    type=click.Path(path_type=Path),
    default=Path("~/.boltz/boltz2_conf.ckpt"),
    show_default=True,
    help="Path to Boltz-2 checkpoint.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/processed/trunk_embeddings"),
    show_default=True,
    help="Directory for cached trunk .npz files.",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    show_default=True,
    help="Compute device.",
)
@click.option(
    "--recycling-steps",
    type=int,
    default=3,
    show_default=True,
    help="Number of recycling iterations.",
)
def precompute_trunk_embeddings(
    manifest: Path,
    checkpoint: Path,
    output_dir: Path,
    device: str,
    recycling_steps: int,
) -> None:
    """Precompute Boltz-2 trunk embeddings for all systems in a manifest."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from kinematic.data.precompute_trunk import precompute_all

    checkpoint = checkpoint.expanduser()
    precompute_all(
        manifest,
        checkpoint,
        output_dir,
        device=device,
        recycling_steps=recycling_steps,
    )
