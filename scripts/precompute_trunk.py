"""Precompute trunk embeddings (s_trunk, z_trunk) using frozen Boltz-2.

Runs the Boltz-2 trunk (input embedder + MSA module + Pairformer) on each
system's first frame and caches (s_trunk, z_trunk, s_inputs) to HDF5.
rel_pos_enc is NOT cached (recomputed at runtime per trunk cache rule).
"""

from __future__ import annotations


def main():
    raise NotImplementedError("Trunk precomputation not yet implemented.")


if __name__ == "__main__":
    main()
