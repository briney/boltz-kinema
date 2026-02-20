"""Octapeptides dataset preprocessing.

~1,100 8-residue peptides, 5 x 1 us each, ~8 ms total.
Force field: AMBER ff99SB-ildn, 300K, explicit TIP3P, 0.1M NaCl.
Format: topology.pdb + trajs/run001_protein.cmprsd.xtc + dataset.json.
4 fs timestep with hydrogen mass repartitioning.

All 5 replicas per system are treated as independent trajectories.
System size is very small (8 residues, ~60-100 heavy atoms).
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


def find_octapeptide_systems(input_dir: Path) -> list[dict]:
    """Discover octapeptide systems.

    Expected layout:
        input_dir/<peptide_id>/topology.pdb
        input_dir/<peptide_id>/trajs/run*_protein.cmprsd.xtc (or *.xtc)
        input_dir/<peptide_id>/dataset.json
    """
    systems = []
    for peptide_dir in sorted(input_dir.iterdir()):
        if not peptide_dir.is_dir():
            continue

        topology = peptide_dir / "topology.pdb"
        if not topology.exists():
            pdb_files = sorted(peptide_dir.glob("*.pdb"))
            if not pdb_files:
                continue
            topology = pdb_files[0]

        # Find trajectory files (multiple replicas)
        trajs_dir = peptide_dir / "trajs"
        if trajs_dir.is_dir():
            traj_files = sorted(trajs_dir.glob("*.xtc"))
        else:
            traj_files = sorted(peptide_dir.glob("*.xtc"))

        if not traj_files:
            continue

        peptide_id = peptide_dir.name

        # Each replica is treated as an independent trajectory
        for i, traj_file in enumerate(traj_files):
            systems.append({
                "system_id": f"octa_{peptide_id}_rep{i}",
                "peptide_id": peptide_id,
                "topology": topology,
                "trajectory": traj_file,
            })

    return systems


def preprocess_one(
    system: dict,
    output_dir: Path,
    ref_dir: Path,
) -> dict | None:
    """Preprocess a single octapeptide trajectory."""
    system_id = system["system_id"]
    logger.info("Processing %s", system_id)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            clean_traj = remove_solvent(
                system["topology"],
                system["trajectory"],
                Path(tmpdir) / "clean.xtc",
            )
            traj = align_trajectory(clean_traj, system["topology"])

        return finalize_processed_system(
            system_id=system_id,
            dataset="octapeptides",
            traj=traj,
            output_dir=output_dir,
            ref_dir=ref_dir,
        )

    except Exception:
        logger.exception("Failed to process %s", system_id)
        return None
