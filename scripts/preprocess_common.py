"""Shared helpers for dataset preprocessing scripts.

.. deprecated::
    This module has moved to ``kinematic.data.preprocess_common``.
    Use ``from kinematic.data.preprocess_common import ...`` instead.
    This file re-exports for backwards compatibility and will be removed
    in a future release.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "scripts/preprocess_common.py is deprecated. "
    "Use 'from kinematic.data.preprocess_common import ...' or "
    "the 'kinematic preprocess' CLI instead.",
    DeprecationWarning,
    stacklevel=2,
)

from kinematic.data.preprocess_common import (  # noqa: E402, F401
    collect_manifest_entries,
    finalize_processed_system,
    write_manifest_entries,
)

# Keep _load_preprocessing_ops for any external code that referenced it.
def _load_preprocessing_ops():
    """Import preprocessing ops lazily (deprecated)."""
    from kinematic.data.preprocessing import (
        build_observation_mask,
        convert_trajectory,
        extract_atom_metadata,
        save_reference_structure,
    )
    return {
        "build_observation_mask": build_observation_mask,
        "convert_trajectory": convert_trajectory,
        "extract_atom_metadata": extract_atom_metadata,
        "save_reference_structure": save_reference_structure,
    }
