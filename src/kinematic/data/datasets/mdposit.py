"""MDposit/DynaRepo dataset preprocessing.

Converts PDB + XTC trajectories from DynaRepo to processed format.
~930 systems, ~700 unique proteins, 3 replicas x 500ns each.

Pipeline:
  1. Solvent removal (MDAnalysis)
  2. Backbone alignment to frame 0 (mdtraj Kabsch)
  3. Ligand valency check where applicable
  4. Build observation mask
  5. Unit conversion: nm -> A, ps -> ns
  6. Save coords .npz + reference structure .npz
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


def find_mdposit_systems(input_dir: Path) -> list[dict]:
    """Discover MDposit/DynaRepo systems.

    Expected layout:
        input_dir/<accession>/replica_<i>/structure.pdb
        input_dir/<accession>/replica_<i>/trajectory.xtc
        input_dir/<accession>/replica_<i>/topology.tpr
    """
    systems = []
    for accession_dir in sorted(input_dir.iterdir()):
        if not accession_dir.is_dir():
            continue
        accession = accession_dir.name

        for replica_dir in sorted(accession_dir.iterdir()):
            if not replica_dir.is_dir():
                continue

            pdb = replica_dir / "structure.pdb"
            xtc = replica_dir / "trajectory.xtc"
            tpr = replica_dir / "topology.tpr"

            if not xtc.exists():
                continue

            # Prefer TPR for topology (more complete), fall back to PDB
            topology = tpr if tpr.exists() else pdb

            replica_idx = replica_dir.name.replace("replica_", "")
            systems.append({
                "system_id": f"mdposit_{accession}_rep{replica_idx}",
                "accession": accession,
                "topology": topology,
                "trajectory": xtc,
                "pdb": pdb,
            })

    return systems


def preprocess_one(
    system: dict,
    output_dir: Path,
    ref_dir: Path,
) -> dict | None:
    """Preprocess a single MDposit system."""
    system_id = system["system_id"]
    logger.info("Processing %s", system_id)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Solvent removal
            clean_traj = remove_solvent(
                system["topology"],
                system["trajectory"],
                Path(tmpdir) / "clean.xtc",
            )

            # Use PDB as topology for mdtraj (it needs atom names)
            topo_for_mdtraj = system["pdb"] if system["pdb"].exists() else system["topology"]

            # Step 2: Frame alignment
            traj = align_trajectory(clean_traj, topo_for_mdtraj)

        return finalize_processed_system(
            system_id=system_id,
            dataset="mdposit",
            traj=traj,
            output_dir=output_dir,
            ref_dir=ref_dir,
        )

    except Exception:
        logger.exception("Failed to process %s", system_id)
        return None
