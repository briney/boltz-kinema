"""ATLAS dataset preprocessing.

Converts GROMACS .xtc trajectories + .pdb topology to processed format.
~1,500 protein chains, 3x100ns trajectories each.

Pipeline:
  1. Solvent removal (MDAnalysis)
  2. Backbone alignment to frame 0 (mdtraj Kabsch)
  3. Build observation mask
  4. Unit conversion: nm -> A, ps -> ns
  5. Save coords .npz + reference structure .npz
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from kinematic.data.preprocessing import (
    align_trajectory,
    remove_solvent,
)
from kinematic.data.preprocess_common import finalize_processed_system

logger = logging.getLogger(__name__)


def find_atlas_systems(input_dir: Path) -> list[dict]:
    """Discover ATLAS systems from directory structure.

    Expected layout:
        input_dir/<chain_id>/<chain_id>.pdb
        input_dir/<chain_id>/<chain_id>_<replica>.xtc
    """
    systems = []
    for chain_dir in sorted(input_dir.iterdir()):
        if not chain_dir.is_dir():
            continue
        chain_id = chain_dir.name

        # Find topology file (.pdb preferred, .gro fallback)
        pdb_files = sorted(chain_dir.glob("*.pdb"))
        gro_files = sorted(chain_dir.glob("*.gro"))
        if pdb_files:
            topology = pdb_files[0]
        elif gro_files:
            topology = gro_files[0]
        else:
            logger.warning("No topology file (.pdb/.gro) found for %s, skipping", chain_id)
            continue

        # Find trajectory files (.xtc)
        xtc_files = sorted(chain_dir.glob("*.xtc"))
        for i, xtc in enumerate(xtc_files):
            systems.append({
                "system_id": f"atlas_{chain_id}_rep{i}",
                "chain_id": chain_id,
                "replica": i,
                "topology": topology,
                "trajectory": xtc,
            })

    return systems


def preprocess_one(
    system: dict,
    output_dir: Path,
    ref_dir: Path,
) -> dict | None:
    """Preprocess a single ATLAS system.

    Returns manifest entry dict, or None on failure.
    """
    system_id = system["system_id"]
    logger.info("Processing %s", system_id)

    try:
        # Step 1: Solvent removal
        with tempfile.TemporaryDirectory() as tmpdir:
            clean_traj = remove_solvent(
                system["topology"],
                system["trajectory"],
                Path(tmpdir) / "clean.xtc",
            )

            # Step 2: Frame alignment
            traj = align_trajectory(clean_traj, system["topology"])

        return finalize_processed_system(
            system_id=system_id,
            dataset="atlas",
            traj=traj,
            output_dir=output_dir,
            ref_dir=ref_dir,
        )

    except Exception:
        logger.exception("Failed to process %s", system_id)
        return None
