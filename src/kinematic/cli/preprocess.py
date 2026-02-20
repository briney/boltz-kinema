"""``kinematic preprocess`` subcommand group.

Provides per-dataset preprocessing subcommands:

    kinematic preprocess atlas --input-dir data/raw/atlas
    kinematic preprocess megasim --input-dir data/raw/megasim --subsample-mutants
    kinematic preprocess misato --input-dir data/raw/misato --coords-unit auto
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import click


def common_preprocess_options(default_manifest: str):
    """Decorator factory for shared preprocessing options."""

    def decorator(func):
        @click.option(
            "--input-dir",
            required=True,
            type=click.Path(path_type=Path),
            help="Root directory of raw dataset files.",
        )
        @click.option(
            "--output-dir",
            type=click.Path(path_type=Path),
            default=Path("data/processed/coords"),
            show_default=True,
            help="Directory for processed coordinate .npz files.",
        )
        @click.option(
            "--ref-dir",
            type=click.Path(path_type=Path),
            default=Path("data/processed/refs"),
            show_default=True,
            help="Directory for reference structure .npz files.",
        )
        @click.option(
            "--manifest-out",
            type=click.Path(path_type=Path),
            default=Path(default_manifest),
            show_default=True,
            help="Output path for the JSON manifest.",
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _run_pipeline(
    *,
    dataset_name: str,
    find_fn,
    preprocess_fn,
    input_dir: Path,
    output_dir: Path,
    ref_dir: Path,
    manifest_out: Path,
    find_kwargs: dict | None = None,
    preprocess_kwargs: dict | None = None,
) -> None:
    """Common pipeline: find systems -> preprocess each -> write manifest."""
    from kinematic.data.preprocess_common import (
        collect_manifest_entries,
        write_manifest_entries,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    systems = find_fn(input_dir, **(find_kwargs or {}))
    logger.info("Found %d %s systems", len(systems), dataset_name)

    extra = preprocess_kwargs or {}
    manifest_entries = collect_manifest_entries(
        systems,
        lambda system: preprocess_fn(system, output_dir, ref_dir, **extra),
    )
    write_manifest_entries(manifest_entries, manifest_out)


@click.group()
def preprocess() -> None:
    """Preprocess raw trajectory datasets into training format."""


@preprocess.command()
@common_preprocess_options("data/processed/atlas_manifest.json")
def atlas(input_dir: Path, output_dir: Path, ref_dir: Path, manifest_out: Path) -> None:
    """Preprocess ATLAS dataset (GROMACS .gro/.xtc trajectories)."""
    from kinematic.data.datasets.atlas import find_atlas_systems, preprocess_one

    _run_pipeline(
        dataset_name="ATLAS",
        find_fn=find_atlas_systems,
        preprocess_fn=preprocess_one,
        input_dir=input_dir,
        output_dir=output_dir,
        ref_dir=ref_dir,
        manifest_out=manifest_out,
    )


@preprocess.command()
@common_preprocess_options("data/processed/cath2_manifest.json")
def cath(input_dir: Path, output_dir: Path, ref_dir: Path, manifest_out: Path) -> None:
    """Preprocess CATH2 dataset (CATH domain trajectories)."""
    from kinematic.data.datasets.cath import find_cath2_systems, preprocess_one

    _run_pipeline(
        dataset_name="CATH2",
        find_fn=find_cath2_systems,
        preprocess_fn=preprocess_one,
        input_dir=input_dir,
        output_dir=output_dir,
        ref_dir=ref_dir,
        manifest_out=manifest_out,
    )


@preprocess.command()
@common_preprocess_options("data/processed/dd13m_manifest.json")
def dd13m(input_dir: Path, output_dir: Path, ref_dir: Path, manifest_out: Path) -> None:
    """Preprocess DD-13M dataset (metadynamics dissociation trajectories)."""
    from kinematic.data.datasets.dd13m import find_dd13m_systems, preprocess_one

    _run_pipeline(
        dataset_name="DD-13M",
        find_fn=find_dd13m_systems,
        preprocess_fn=preprocess_one,
        input_dir=input_dir,
        output_dir=output_dir,
        ref_dir=ref_dir,
        manifest_out=manifest_out,
    )


@preprocess.command()
@common_preprocess_options("data/processed/mdposit_manifest.json")
def mdposit(input_dir: Path, output_dir: Path, ref_dir: Path, manifest_out: Path) -> None:
    """Preprocess MDposit/DynaRepo dataset (PDB + XTC trajectories)."""
    from kinematic.data.datasets.mdposit import find_mdposit_systems, preprocess_one

    _run_pipeline(
        dataset_name="MDposit",
        find_fn=find_mdposit_systems,
        preprocess_fn=preprocess_one,
        input_dir=input_dir,
        output_dir=output_dir,
        ref_dir=ref_dir,
        manifest_out=manifest_out,
    )


@preprocess.command()
@common_preprocess_options("data/processed/megasim_manifest.json")
@click.option(
    "--subsample-mutants",
    is_flag=True,
    default=False,
    help="Subsample mutants to ~4,500 (default for Phase 1 training).",
)
@click.option(
    "--all-mutants",
    is_flag=True,
    default=False,
    help="Keep all 21,458 mutants (for Phase 1.5 training).",
)
def megasim(
    input_dir: Path,
    output_dir: Path,
    ref_dir: Path,
    manifest_out: Path,
    subsample_mutants: bool,
    all_mutants: bool,
) -> None:
    """Preprocess MegaSim dataset (wildtype + mutant protein simulations)."""
    from kinematic.data.datasets.megasim import find_megasim_systems, preprocess_one

    subsample = subsample_mutants or not all_mutants
    _run_pipeline(
        dataset_name="MegaSim",
        find_fn=find_megasim_systems,
        preprocess_fn=preprocess_one,
        input_dir=input_dir,
        output_dir=output_dir,
        ref_dir=ref_dir,
        manifest_out=manifest_out,
        find_kwargs={"subsample_mutants": subsample},
    )


@preprocess.command()
@common_preprocess_options("data/processed/misato_manifest.json")
@click.option(
    "--coords-unit",
    type=click.Choice(["auto", "nm", "angstrom"]),
    default="auto",
    show_default=True,
    help="Coordinate unit for HDF5 coordinates. 'auto' uses metadata + geometric heuristics.",
)
def misato(
    input_dir: Path,
    output_dir: Path,
    ref_dir: Path,
    manifest_out: Path,
    coords_unit: str,
) -> None:
    """Preprocess MISATO dataset (HDF5 protein-ligand complexes)."""
    from kinematic.data.datasets.misato import find_misato_systems, preprocess_one

    _run_pipeline(
        dataset_name="MISATO",
        find_fn=find_misato_systems,
        preprocess_fn=preprocess_one,
        input_dir=input_dir,
        output_dir=output_dir,
        ref_dir=ref_dir,
        manifest_out=manifest_out,
        preprocess_kwargs={"coords_unit": coords_unit},
    )


@preprocess.command()
@common_preprocess_options("data/processed/octapeptides_manifest.json")
def octapeptides(
    input_dir: Path, output_dir: Path, ref_dir: Path, manifest_out: Path
) -> None:
    """Preprocess Octapeptides dataset (8-residue peptide trajectories)."""
    from kinematic.data.datasets.octapeptides import (
        find_octapeptide_systems,
        preprocess_one,
    )

    _run_pipeline(
        dataset_name="Octapeptides",
        find_fn=find_octapeptide_systems,
        preprocess_fn=preprocess_one,
        input_dir=input_dir,
        output_dir=output_dir,
        ref_dir=ref_dir,
        manifest_out=manifest_out,
    )
