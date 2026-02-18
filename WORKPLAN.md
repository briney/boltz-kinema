# BoltzKinema: Detailed Implementation Workplan

## Context

BioKinema (Feng et al., 2026) is a diffusion-based generative model that predicts continuous-time, all-atom biomolecular trajectories by extending an AlphaFold 3-like architecture with a Spatial-Temporal Diffusion Module. This workplan describes a complete reimplementation (**BoltzKinema**) using **Boltz-2** as the structural backbone instead of Protenix. Boltz-2 is MIT-licensed, has pretrained weights, and is architecturally equivalent to AF3. The training infrastructure uses HuggingFace `accelerate` (not PyTorch Lightning).

## Normative Conventions (Must Follow)

### Coordinate and Time Units

| Quantity | Canonical Unit | Notes |
|----------|----------------|-------|
| Coordinates (`coords`, `x_noisy`, `x_denoised`) | Angstrom (A) | All model-facing tensors and cached trajectory arrays use Angstroms |
| Timestamps (`timestamps`) | nanoseconds (ns) | Stored and consumed in ns |
| Frame stride metadata (`dt`) | nanoseconds (ns) | Converted from source format during preprocessing |
| EDM noise `sigma` | Angstrom-equivalent coordinate scale | Same unit family as coordinates |

**Conversion rule from MDTraj:** `xyz` is loaded in nm and time in ps, so preprocessing must convert `nm -> A` and `ps -> ns` before writing processed files.

### Required Runtime Batch Contract

The following keys are required by the training forward pass and must be produced by the dataset/collator:
`coords`, `timestamps`, `sigma`, `conditioning_mask`, `s_trunk`, `z_trunk`, `s_inputs`, `atom_pad_mask`, `token_pad_mask`, `atom_to_token`, `mol_type_per_atom`, `observed_atom_mask`, `feats`.

Optional keys:
`bond_indices`, `bond_lengths`, `ligand_mask`, `split`.

### Trunk Cache Rule

`rel_pos_enc` is **not** cached in the trunk artifact (to reduce storage by ~2x for pair features). Relative positional features are recomputed by the runtime featurizer/conditioning path from structural metadata in `feats`.

---

## 1. Project Setup

### 1.1 Directory Structure

```
boltz-kinema/
├── pyproject.toml
├── README.md
├── configs/
│   ├── train_phase0.yaml             # Phase 0: monomer dynamics pretraining
│   ├── train_equilibrium.yaml        # Phase 1: full mixed training (updated)
│   ├── train_mutant_enrichment.yaml  # Phase 1.5: MegaSim mutant enrichment (optional)
│   ├── train_unbinding.yaml          # Phase 2: DD-13M fine-tuning config
│   └── inference.yaml                # Inference config
├── scripts/
│   ├── download_data.py            # Dataset acquisition
│   ├── preprocess_atlas.py         # ATLAS preprocessing
│   ├── preprocess_misato.py        # MISATO preprocessing
│   ├── preprocess_mdposit.py       # MDposit preprocessing
│   ├── preprocess_dd13m.py         # DD-13M preprocessing
│   ├── preprocess_cath.py          # CATH domains preprocessing
│   ├── preprocess_megasim.py       # MegaSim preprocessing (with mutant subsampling)
│   ├── preprocess_octapeptides.py  # Octapeptides preprocessing
│   ├── precompute_trunk.py         # Trunk embedding precomputation
│   ├── train.py                    # Training entry point (accelerate)
│   └── generate.py                 # Inference entry point
├── src/
│   └── boltzkinema/
│       ├── __init__.py
│       ├── model/
│       │   ├── __init__.py
│       │   ├── boltzkinema.py          # Top-level model (nn.Module)
│       │   ├── temporal_attention.py   # TemporalAttentionWithDecay
│       │   ├── spatial_temporal_transformer.py  # ST token transformer block
│       │   ├── spatial_temporal_atom.py # ST atom encoder/decoder blocks
│       │   ├── diffusion_module.py     # SpatialTemporalDiffusionModule
│       │   ├── edm.py                  # Per-frame EDM preconditioning
│       │   └── weight_loading.py       # Boltz-2 weight extraction utilities
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py              # BoltzKinemaDataset
│       │   ├── preprocessing.py        # Trajectory preprocessing utilities
│       │   ├── trunk_cache.py          # Trunk embedding I/O
│       │   ├── noise_masking.py        # Noise-as-masking sample construction
│       │   └── collator.py             # Batch collation with padding
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py              # Accelerate training loop
│       │   ├── losses.py               # All loss functions
│       │   └── scheduler.py            # LR scheduler
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── sampler.py              # EDM sampling loop
│       │   ├── hierarchical.py         # Coarse forecast + fine interpolation
│       │   └── unbinding.py            # Auto-regressive unbinding generation
│       └── evaluation/
│           ├── __init__.py
│           └── metrics.py              # RMSF, W2-distance, IMS, etc.
└── tests/
    ├── test_temporal_attention.py
    ├── test_single_frame_equivalence.py
    ├── test_noise_masking.py
    ├── test_losses.py
    └── test_shapes.py
```

### 1.2 Dependencies (`pyproject.toml`)

```toml
[project]
name = "boltzkinema"
requires-python = ">=3.10"
dependencies = [
    "boltz",                    # Boltz-2 (pip install boltz or from ../boltz)
    "torch>=2.1",
    "accelerate>=0.25",
    "wandb",
    "omegaconf",
    "mdanalysis",               # MD trajectory parsing
    "mdtraj",                   # Trajectory alignment
    "h5py",                     # HDF5 I/O for MISATO + trunk cache
    "numpy",
    "scipy",
    "biopython",
    "rdkit",                    # Ligand valency checks
    "einops",
]
```

### 1.3 Boltz-2 Key Facts (from codebase exploration)

These exact values are from the Boltz-2 source code at `../boltz`:

| Parameter | Value | Source File |
|-----------|-------|-------------|
| `token_s` | 384 | `src/boltz/main.py` |
| `token_z` | 128 | `src/boltz/main.py` |
| `atom_s` | 128 | `src/boltz/main.py` |
| `atom_z` | 16 | `src/boltz/main.py` |
| `atom_feature_dim` | 128 (Boltz-2) | `src/boltz/main.py` |
| Pairformer blocks | 64 (Boltz-2, not 48) | `PairformerArgsV2` in `main.py` |
| Pairformer heads | 16 | `PairformerArgsV2` |
| Token transformer depth | 24 | `structure.yaml` |
| Token transformer heads | 16 (not 8) | `structure.yaml` |
| Atom encoder/decoder depth | 3 | `structure.yaml` |
| Atom encoder/decoder heads | 4 | `structure.yaml` |
| `atoms_per_window_queries` | 32 | `main.py` |
| `atoms_per_window_keys` | 128 | `main.py` |
| `sigma_data` | 16.0 | `Boltz2DiffusionParams` |
| `sigma_min` | 0.0001 | `Boltz2DiffusionParams` |
| `sigma_max` | 160.0 | `Boltz2DiffusionParams` |
| `P_mean` | -1.2 | `Boltz2DiffusionParams` |
| `P_std` | 1.5 | `Boltz2DiffusionParams` |

**Boltz-2 weight key prefixes** (from checkpoint `state_dict`):
- Trunk: `input_embedder.*`, `msa_module.*`, `pairformer_module.*`, `s_init.*`, `z_init_1.*`, `z_init_2.*`, `rel_pos.*`, `token_bonds.*`, `s_norm.*`, `z_norm.*`, `s_recycle.*`, `z_recycle.*`
- Diffusion conditioning: `diffusion_conditioning.*`
- Score model: `structure_module.score_model.*`
  - Atom encoder: `structure_module.score_model.atom_attention_encoder.*`
  - Token transformer: `structure_module.score_model.token_transformer.*`
  - Atom decoder: `structure_module.score_model.atom_attention_decoder.*`
  - Single conditioner: `structure_module.score_model.single_conditioner.*`

---

## 2. Data Pipeline

### 2.1 Dataset Acquisition

**ATLAS** (~30 GB):
```bash
# Download from https://www.dsimb.inserm.fr/ATLAS/
# ~1,500 protein chains, 3x100ns trajectories each
# Format: GROMACS .xtc trajectories + .gro topology
wget -r https://www.dsimb.inserm.fr/ATLAS/database/
```

**MISATO** (~50 GB):
```bash
# Download from https://github.com/t7morgen/misato-dataset
# ~16,000 protein-ligand complexes, 8ns each (100 frames)
# Format: HDF5 with coordinates + topology
python -m misato.download --output-dir data/raw/misato/
```

**MDposit/DynaRepo** (~200 GB):
```python
# DynaRepo is part of the federated MDDB (Molecular Dynamics Data Bank).
# Multiple nodes share an identical REST API (unauthenticated, public).
# DynaRepo/Inria: ~930 systems, ~700 unique proteins, 3 replicas x 500ns each (~1,146 µs total)
# Other nodes: MMB/BSC (~4k systems), BioExcel COVID-19 (~1.5k), MDposit-dev (~10k)
# Reference: Tek et al., NAR 2025
# Format: PDB (structure) + XTC (trajectory) + TPR (topology)

import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# -- Configurable MDDB nodes (uncomment to include additional nodes) --
NODES = [
    ("dynarepo", "https://dynarepo.inria.fr"),
    # ("mmb", "https://mmb.mddbr.eu"),           # MoDEL dataset (~4k systems)
    # ("cv19", "https://bioexcel-cv19.mddbr.eu"), # COVID-19 sims (~1.5k systems)
]
OUTPUT_DIR = Path("data/raw/dynarepo")
FILES = ["structure.pdb", "trajectory.xtc", "topology.tpr"]
MAX_WORKERS = 4

def get_all_projects(base_url):
    """Paginate through all projects on a node."""
    projects = []
    page = 1
    while True:
        r = requests.get(f"{base_url}/api/rest/v1/projects", params={"page": page})
        r.raise_for_status()
        data = r.json()
        batch = data.get("projects", [])
        if not batch:
            break
        projects.extend(batch)
        if len(projects) >= data.get("filteredCount", 0):
            break
        page += 1
    return projects

def download_file(base_url, accession, filename, md_index, dest):
    """Download a single file for a specific replica."""
    if dest.exists():
        return  # skip existing (resume support)
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(
        f"{base_url}/api/rest/v1/projects/{accession}/files/{filename}",
        params={"md": md_index},
        stream=True,
    )
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

# Main download loop
for node_name, base_url in NODES:
    projects = get_all_projects(base_url)
    tasks = []
    for proj in projects:
        accession = proj["accession"]
        n_replicas = len(proj.get("mds", ["replica 1"]))
        for md_i in range(n_replicas):
            for fname in FILES:
                dest = OUTPUT_DIR / accession / f"replica_{md_i}" / fname
                tasks.append((base_url, accession, fname, md_i, dest))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_file, *t): t for t in tasks}
        for f in tqdm(as_completed(futures), total=len(futures), desc=node_name):
            f.result()  # raises on error
```

**DD-13M** (for unbinding, ~100 GB):
```bash
# Download from arXiv:2504.18367
# 565 complexes, 26,612 metadynamics dissociation trajectories
# Format: coordinate trajectories at 10 ps intervals
```

**CATH2 Domains** (~28 GB compressed):
```python
# Source: https://zenodo.org/records/15629740
# Download only MSR_cath2.zip (CATH1's adaptive sampling data is excluded)
# ~1,100 CATH domains, 50-200 amino acids, ~1 us each, ~41 ms total
# Force field: AMBER ff99SB-ildn, 300K, explicit TIP3P
# Format: topology.pdb + trajs/*.cmprsd.xtc + dataset.json per system
#
# NOTE: CATH1 (ONE_cath1.zip, 50 domains with adaptive sampling) is intentionally
# excluded — its non-equilibrium transition-state conformations conflict with the
# noise-as-masking paradigm.

import requests
from pathlib import Path
from tqdm import tqdm

ZENODO_RECORD = "15629740"
CATH2_FILENAME = "MSR_cath2.zip"
OUTPUT_DIR = Path("data/raw/cath2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Get file download URL from Zenodo REST API
r = requests.get(f"https://zenodo.org/api/records/{ZENODO_RECORD}")
r.raise_for_status()
files = r.json()["files"]
cath2_file = next(f for f in files if f["key"] == CATH2_FILENAME)
download_url = cath2_file["links"]["self"]
file_size = cath2_file["size"]

# Download with progress bar
dest = OUTPUT_DIR / CATH2_FILENAME
if not dest.exists():
    with requests.get(download_url, stream=True) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in tqdm(resp.iter_content(chunk_size=8192),
                              total=file_size // 8192, desc="CATH2"):
                f.write(chunk)
```

**MegaSim** (~10 GB compressed):
```python
# Source: https://zenodo.org/records/15641184
# Download both wildtype merge file (1.3 GB) and mutants all-atom file (8.4 GB)
# 271 wildtype proteins: AMBER ff14sb + ff99sb-disp, 295K, 1.5 us/seed (last 1 us retained)
# 21,458 point mutants: AMBER ff99sb-disp, 1 us each
# Format: topology.pdb + trajs/*.xtc + dataset.json per system
#
# Mutant subsampling logic: select all 271 wildtypes + ~4,500 mutants
# (top/bottom 10% by deltaG + diverse-position representatives)
# to keep precompute cost manageable.

import requests
from pathlib import Path
from tqdm import tqdm

ZENODO_RECORD = "15641184"
MEGASIM_FILES = [
    "megasim_wildtype_merge.zip",   # ~1.3 GB, all wildtypes
    "megasim_mutants_allatom.zip",  # ~8.4 GB, all mutants
]
OUTPUT_DIR = Path("data/raw/megasim")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

r = requests.get(f"https://zenodo.org/api/records/{ZENODO_RECORD}")
r.raise_for_status()
files = {f["key"]: f for f in r.json()["files"]}

for fname in MEGASIM_FILES:
    file_info = files[fname]
    download_url = file_info["links"]["self"]
    file_size = file_info["size"]
    dest = OUTPUT_DIR / fname
    if not dest.exists():
        with requests.get(download_url, stream=True) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in tqdm(resp.iter_content(chunk_size=8192),
                                  total=file_size // 8192, desc=fname):
                    f.write(chunk)
```

**Octapeptides** (~511 MB compressed, ~78 GB uncompressed):
```python
# Source: https://zenodo.org/records/15641199
# ~1,100 8-residue peptides, 5 x 1 us each, ~8 ms total
# Force field: AMBER ff99SB-ildn, 300K, explicit TIP3P, 0.1M NaCl
# Format: topology.pdb + trajs/run001_protein.cmprsd.xtc + dataset.json per system
# 4 fs timestep with hydrogen mass repartitioning

import requests
from pathlib import Path
from tqdm import tqdm

ZENODO_RECORD = "15641199"
OUTPUT_DIR = Path("data/raw/octapeptides")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

r = requests.get(f"https://zenodo.org/api/records/{ZENODO_RECORD}")
r.raise_for_status()
files = r.json()["files"]

for file_info in files:
    download_url = file_info["links"]["self"]
    file_size = file_info["size"]
    dest = OUTPUT_DIR / file_info["key"]
    if not dest.exists():
        with requests.get(download_url, stream=True) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in tqdm(resp.iter_content(chunk_size=8192),
                                  total=file_size // 8192, desc=file_info["key"]):
                    f.write(chunk)
```

### 2.2 Preprocessing Pipeline

Each preprocessing script (`scripts/preprocess_*.py`) performs these steps:

#### Step 1: Solvent Removal
```python
import MDAnalysis as mda

def remove_solvent(topology_path, trajectory_path):
    """Remove water, ions. Keep protein/ligand/DNA/RNA."""
    u = mda.Universe(topology_path, trajectory_path)
    non_solvent = u.select_atoms("not (resname HOH TIP3 WAT SOL NA CL K MG)")
    # Write cleaned trajectory
    with mda.Writer(out_path, non_solvent.n_atoms) as w:
        for ts in u.trajectory:
            w.write(non_solvent)
    return out_path
```

#### Step 2: Frame Alignment (remove rigid-body motion)
```python
import mdtraj

def align_trajectory(trajectory_path, topology_path):
    """Align all frames to frame 0 using Kabsch alignment on backbone atoms."""
    traj = mdtraj.load(trajectory_path, top=topology_path)
    # Select backbone atoms for alignment reference
    backbone = traj.topology.select("backbone")
    traj.superpose(traj, frame=0, atom_indices=backbone)
    return traj
```

#### Step 3: Missing-Atom/Residue Observation Mask
```python
def build_observation_mask(topology, reference_coords):
    """
    Mark atoms unresolved in the experimental/reference structure.

    Output:
      observed_atom_mask: (n_atoms,) bool
    This mask is propagated to training and used to suppress losses
    for unresolved atoms/residues.
    """
    observed = np.isfinite(reference_coords).all(axis=-1)
    return observed.astype(np.bool_)
```

#### Step 4: Ligand Valency Check (MISATO, MDposit with ligands)
```python
from rdkit import Chem
from rdkit.Chem import AllChem

def check_ligand_valency(mol_block):
    """Filter trajectories with chemically invalid ligand states."""
    mol = Chem.MolFromMolBlock(mol_block, sanitize=False)
    try:
        Chem.SanitizeMol(mol)
        return True
    except:
        return False
```

#### Step 5: Convert to Boltz-2 Compatible Input
```python
def convert_to_boltz_input(traj, system_id, output_dir):
    """
    Convert an MDAnalysis/mdtraj trajectory to Boltz-2 input format.

    For each system, produces:
    1. A structure file (.cif or .yaml) for Boltz-2 tokenization
    2. A coordinate array (.npz) with all frames: (n_frames, n_atoms, 3)
    3. Metadata: atom names, residue indices, chain IDs, molecule types
    """
    # Extract first frame as reference structure for tokenization
    ref_coords = traj.xyz[0]  # (n_atoms, 3)

    # Build atom-level metadata
    atoms = []
    for atom in traj.topology.atoms:
        atoms.append({
            'name': atom.name,
            'element': atom.element.symbol,
            'residue_name': atom.residue.name,
            'residue_index': atom.residue.index,
            'chain_id': atom.residue.chain.index,
            'mol_type': classify_mol_type(atom.residue),  # protein/dna/rna/ligand
        })

    # Convert MDTraj units: nm -> A, ps -> ns
    coords_A = (traj.xyz * 10.0).astype(np.float32)
    times_ns = (traj.time / 1000.0).astype(np.float32)
    dt_ns = float(traj.timestep) / 1000.0

    # Save coordinate trajectory (canonical units)
    np.savez_compressed(
        output_dir / f"{system_id}_coords.npz",
        coords=coords_A,            # (n_frames, n_atoms, 3) in Angstrom
        times=times_ns,             # (n_frames,) in ns
        dt=dt_ns,                   # ns per frame
        observed_atom_mask=build_observation_mask(traj.topology, coords_A[0]),
    )

    # Save reference structure for Boltz-2 tokenization
    save_reference_structure(atoms, ref_coords, output_dir / f"{system_id}_ref.npz")
```

#### Step 6: Tokenization (using Boltz-2's tokenizer)
```python
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer

def tokenize_system(ref_structure_path):
    """
    Run Boltz-2 tokenization on the reference structure.

    Tokenization rules (from Boltz-2 codebase):
    - Protein: 1 token per residue (center=CA, disto=CB)
    - DNA/RNA: 1 token per nucleotide
    - Ligands (NONPOLYMER): 1 token per atom

    Produces:
    - token_to_atom_map: (n_tokens, max_atoms_per_token) mapping
    - atom_to_token: (n_atoms, n_tokens) soft assignment matrix
    - res_type: (n_tokens,) residue type indices
    - mol_type: (n_tokens,) molecule type (0=protein, 1=DNA, 2=RNA, 3=ligand)
    """
    tokenizer = Boltz2Tokenizer()
    input_data = load_input(ref_structure_path)
    tokenized = tokenizer.tokenize(input_data)
    return tokenized
```

#### Step 7: Dataset-Specific Ranges

| Dataset | Trajectory Duration | Usable Δt Range | Frames per Traj |
|---------|-------------------|-----------------|-----------------|
| ATLAS | 3 × 100 ns | 0.1–10 ns | 3,000 (at 0.1 ns) |
| MISATO | 8 ns (after 2 ns equil) | 0.08–0.8 ns | 100 |
| MDposit | 2.47–5,350 ns | 0.1–100 ns | Varies |
| DD-13M | Variable | 10 ps steps | Varies |
| **CATH2** | **~1 µs each** | **0.1–50.0 ns** | **~10,000** |
| **MegaSim-WT** | **1 µs (after 0.5 µs burn-in)** | **0.1–50.0 ns** | **~10,000** |
| **MegaSim-mutants** | **1 µs** | **0.1–50.0 ns** | **~10,000** |
| **Octapeptides** | **5 × 1 µs** | **0.1–100.0 ns** | **~5,000/replica** |

**Dataset-specific preprocessing notes:**
- **MegaSim:** For mutant subsampling, load MEGAscale experimental deltaG values. Retain all wildtypes. For mutants, compute deltaG z-scores and select |z| > 1.28 (top/bottom ~10%) plus a stratified random sample across mutation positions for diversity. Mark folded/unfolded state classification per frame using FNC (fraction of native contacts) from the dataset metadata.
- **MegaSim force field note:** Wildtype data includes trajectories under two force fields (ff14sb and ff99sb-disp). Both are retained without distinction — the model learns an average over force fields.
- **Octapeptides:** All 5 replicas per system are treated as independent trajectories. System size is very small (8 residues, ~60-100 heavy atoms) — trunk precomputation is negligible.
- **CATH2:** Data uses compressed XTC format (`.cmprsd.xtc`). MDAnalysis/mdtraj handle this transparently. Domain sizes (50-200 residues) are comparable to ATLAS.

### 2.3 Trunk Embedding Precomputation

**Script:** `scripts/precompute_trunk.py`

This is the most critical preprocessing step. Since the Boltz-2 trunk is frozen, we precompute `s_trunk`, `z_trunk`, and `s_inputs` once per system.

```python
import torch
from boltz.main import BoltzModel  # or direct checkpoint load
from boltz.model.models.boltz2 import Boltz2
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.feature.featurizerv2 import Boltz2Featurizer

def precompute_trunk_embeddings(
    checkpoint_path: str,      # ~/.boltz/boltz2_conf.ckpt
    systems_dir: str,          # Directory with tokenized systems
    output_dir: str,           # Where to save embeddings
    device: str = "cuda",
    recycling_steps: int = 3,  # Boltz-2 default: 4 cycles
):
    """
    For each system, run the Boltz-2 trunk and cache:
    - s_inputs: (N_tokens, 384) - raw input embeddings
    - s_trunk:  (N_tokens, 384) - refined single representation
    - z_trunk:  (N_tokens, N_tokens, 128) - refined pair representation

    Storage estimate per system (200-residue protein):
    - s_inputs: 200 × 384 × 2 bytes = 150 KB
    - s_trunk:  200 × 384 × 2 bytes = 150 KB
    - z_trunk:  200 × 200 × 128 × 2 bytes = 10 MB
    - Total: ~10.3 MB per system
    - 20,000 systems: ~206 GB raw fp16 (typically reduced with chunked compression)
    """
    # Load Boltz-2 model
    model = Boltz2.load_from_checkpoint(
        checkpoint_path,
        strict=True,
        map_location=device,
        # ... (pass required args from Boltz2DiffusionParams, PairformerArgsV2, etc.)
    )
    model.eval()

    # We need to extract intermediate representations from Boltz2.forward().
    # The trunk computation is in Boltz2.forward() lines ~450-550.
    # We hook into the model to capture s, z, s_inputs after the pairformer.

    for system_path in tqdm(all_systems):
        feats = prepare_features(system_path)  # Using Boltz2Featurizer
        feats = {k: v.to(device).unsqueeze(0) for k, v in feats.items()}

        with torch.no_grad():
            # Run trunk (InputEmbedder -> MSA -> Pairformer with recycling)
            # Extract s, z, s_inputs
            s_inputs, s_trunk, z_trunk = run_trunk(model, feats, recycling_steps)

        # Save as compressed float16
        np.savez_compressed(
            output_dir / f"{system_id}_trunk.npz",
            s_inputs=s_inputs.cpu().half().numpy(),
            s_trunk=s_trunk.cpu().half().numpy(),
            z_trunk=z_trunk.cpu().half().numpy(),
        )


def run_trunk(model, feats, recycling_steps):
    """
    Execute only the trunk portion of Boltz2.forward() and return
    intermediate representations.

    This replicates the logic from Boltz2.forward() (boltz2.py ~line 450):
    1. input_embedder(feats) -> s_inputs
    2. s_init(s_inputs) -> s; z_init_1 + z_init_2 outer product -> z
    3. Add rel_pos, token_bonds, contact_conditioning to z
    4. For recycling_step in range(recycling_steps + 1):
       a. s = s_init + s_recycle(s_norm(s))
       b. z = z_init + z_recycle(z_norm(z))
       c. z += msa_module(z, s_inputs, feats)
       d. s, z = pairformer_module(s, z, mask, pair_mask)
    5. Return s, z, s_inputs
    """
    # --- InputEmbedder ---
    s_inputs = model.input_embedder(feats)  # (1, N, 384)

    # --- Initial projections ---
    s_init = model.s_init(s_inputs)  # (1, N, 384)
    z_init = (
        model.z_init_1(s_inputs).unsqueeze(2)
        + model.z_init_2(s_inputs).unsqueeze(1)
    )  # (1, N, N, 128)

    # --- Relative position encoding ---
    rel_pos_enc = model.rel_pos(feats)  # (1, N, N, 128)
    z_init = z_init + rel_pos_enc

    # --- Token bonds ---
    if hasattr(model, 'token_bonds'):
        z_init = z_init + model.token_bonds(feats["token_bonds"])

    # --- Contact conditioning ---
    if hasattr(model, 'contact_conditioning'):
        z_init = z_init + model.contact_conditioning(feats)

    # --- Recycling loop ---
    mask = feats["token_pad_mask"]
    pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)

    s = torch.zeros_like(s_init)
    z = torch.zeros_like(z_init)

    for cycle in range(recycling_steps + 1):
        s = s_init + model.s_recycle(model.s_norm(s))
        z = z_init + model.z_recycle(model.z_norm(z))
        z = z + model.msa_module(z, s_inputs, feats)
        s, z = model.pairformer_module(s, z, mask, pair_mask)

    return s_inputs.squeeze(0), s.squeeze(0), z.squeeze(0)
```

#### Storage and Compute Estimates (All Datasets)

| Dataset | Systems | Avg Tokens | z_trunk Storage | GPU-Hours (A100) |
|---------|---------|-----------|----------------|-----------------|
| ATLAS | 1,500 | ~150 | ~34 GB | ~12.5 |
| MISATO | 16,000 | ~200 | ~410 GB | ~133 |
| MDposit | 3,271 | ~250 | ~130 GB | ~27 |
| **CATH2** | **1,100** | **~130** | **~24 GB** | **~9** |
| **MegaSim-WT** | **271 x 2 FF** | **~120** | **~5 GB** | **~4.5** |
| **MegaSim-mutants (subset)** | **~4,500** | **~120** | **~39 GB** | **~37.5** |
| **Octapeptides** | **1,100** | **~8** | **~0.02 GB** | **~0.5** |
| **BioEmu datasets total** | **~7,000** | — | **~68 GB** | **~51.5** |

**Note:** Octapeptides have only ~8 tokens per system, so `z_trunk` is 8×8×128×2 = 16 KB per system — negligible storage and compute. MegaSim mutant subset is the largest new cost; if the optional Phase 1.5 uses all 21,458 mutants, precomputation increases to ~190 GB / ~180 GPU-hours.

### 2.4 Trunk Cache Scalability Strategy

To prevent the pair representation cache from dominating storage and I/O:

1. Store trunk caches in chunked format (`zarr` or chunked HDF5) with token-tile chunks for `z_trunk`.
2. Keep `z_trunk` as `float16`; keep token masks and index maps as compact integer types.
3. Do not cache `rel_pos_enc`; recompute at runtime from metadata.
4. Use memory-mapped reads and pinned host staging for `z_trunk`.
5. Use token-bucket batching to limit `(max_tokens)^2` padding waste in collator.

### 2.5 Dataset Class

**File:** `src/boltzkinema/data/dataset.py`

```python
class BoltzKinemaDataset(torch.utils.data.Dataset):
    """
    Trajectory dataset for BoltzKinema training.

    Each sample provides:
    - coords: (T, N_atoms, 3) trajectory segment coordinates in Angstrom
    - timestamps: (T,) continuous timestamps in nanoseconds
    - sigma: (T,) per-frame noise levels (0.0 for conditioning frames)
    - conditioning_mask: (T,) bool - True for conditioning (clean) frames
    - s_trunk: (N_tokens, 384) precomputed trunk single representation
    - z_trunk: (N_tokens, N_tokens, 128) precomputed trunk pair representation
    - s_inputs: (N_tokens, 384) precomputed raw input features
    - atom_pad_mask: (N_atoms,) valid atom mask
    - observed_atom_mask: (N_atoms,) observed-structure mask (False for unresolved atoms)
    - token_pad_mask: (N_tokens,) valid token mask
    - atom_to_token: (N_atoms, N_tokens) assignment matrix
    - mol_type_per_atom: (N_atoms,) molecule type per atom
    - split: str - dataset split label (train/val/test/ood)
    - task: str - 'forecasting' or 'interpolation'

    Constructor args:
        manifest_path: Path to JSON manifest listing all systems
        trunk_cache_dir: Directory with precomputed trunk embeddings
        coords_dir: Directory with coordinate trajectory files
        n_frames: int - number of frames per training sample (default: 32)
        dataset_weights: dict - sampling weights per dataset
            Phase 0 (monomer pretraining):
                {"cath2": 4.0, "atlas": 2.0, "octapeptides": 1.0}
            Phase 1 (full mixed):
                {"atlas": 1.0, "misato": 1.0, "mdposit": 1.0,
                 "cath2": 0.5, "megasim_wt": 0.5, "megasim_mut": 0.5}
        dt_ranges: dict - per-dataset Δt ranges in ns
            {"atlas": (0.1, 10.0), "misato": (0.08, 0.8), "mdposit": (0.1, 100.0),
             "cath2": (0.1, 50.0), "megasim_wt": (0.1, 50.0),
             "megasim_mut": (0.1, 50.0), "octapeptides": (0.1, 100.0)}
        noise_P_mean: float = -1.2 (EDM log-normal mean)
        noise_P_std: float = 1.5 (EDM log-normal std)
        sigma_data: float = 16.0
        forecast_prob: float = 0.5 (probability of forecasting vs interpolation task)
    """

    def __getitem__(self, idx):
        # 1. Sample a system (weighted by dataset ratios)
        system = self.sample_system()

        # 2. Sample inter-frame interval Δt
        dt_min, dt_max = self.dt_ranges[system.dataset]
        log_dt = random.uniform(math.log(dt_min), math.log(dt_max))
        dt_ns = math.exp(log_dt)  # Log-uniform sampling

        # 3. Convert Δt to frame indices
        dt_frames = max(1, round(dt_ns / system.frame_dt_ns))
        max_start = system.n_frames - self.n_frames * dt_frames
        if max_start <= 0:
            dt_frames = max(1, system.n_frames // self.n_frames)
            max_start = system.n_frames - self.n_frames * dt_frames
        start = random.randint(0, max(0, max_start))
        indices = [start + i * dt_frames for i in range(self.n_frames)]

        # 4. Load coordinates for selected frames (already in Angstrom)
        coords = system.load_coords(indices)  # (T, N_atoms, 3), Angstrom

        # 5. Random SE(3) augmentation
        R = random_rotation_matrix()
        t = torch.randn(3) * 10.0  # random translation (Å)
        coords = coords @ R.T + t

        # 6. Compute timestamps
        timestamps = torch.tensor([i * dt_ns for i in range(self.n_frames)],
                                   dtype=torch.float32)

        # 7. Noise-as-masking assignment
        sigma, conditioning_mask, task = self.assign_noise(self.n_frames)

        # 8. Load precomputed trunk embeddings
        trunk = system.load_trunk()  # s_trunk, z_trunk, s_inputs

        return {
            'coords': coords,
            'timestamps': timestamps,
            'sigma': sigma,
            'conditioning_mask': conditioning_mask,
            'task': task,
            's_trunk': trunk['s_trunk'],
            'z_trunk': trunk['z_trunk'],
            's_inputs': trunk['s_inputs'],
            'atom_pad_mask': system.atom_pad_mask,
            'observed_atom_mask': system.observed_atom_mask,
            'token_pad_mask': system.token_pad_mask,
            'atom_to_token': system.atom_to_token,
            'mol_type_per_atom': system.mol_type_per_atom,
            'split': system.split,
            'feats': system.feats,  # ref_pos, ref_charge, ref_element, etc.
        }

    def assign_noise(self, T):
        """Noise-as-masking: assign per-frame noise levels."""
        # Sample noise levels from log-normal distribution
        log_sigma = torch.randn(T) * self.noise_P_std + self.noise_P_mean
        sigma = self.sigma_data * torch.exp(log_sigma)  # (T,)

        # Choose task
        task = 'forecasting' if random.random() < self.forecast_prob else 'interpolation'

        if task == 'forecasting':
            sigma[0] = 0.0  # first frame is clean conditioning
        else:
            sigma[0] = 0.0   # first frame clean
            sigma[-1] = 0.0  # last frame clean

        conditioning_mask = (sigma == 0.0)
        return sigma, conditioning_mask, task
```

### 2.6 Collator

**File:** `src/boltzkinema/data/collator.py`

```python
class BoltzKinemaCollator:
    """
    Pads variable-size systems to batch max and constructs batch tensors.

    All systems in a batch are padded to:
    - max N_tokens across the batch
    - max N_atoms across the batch
    T (number of frames) is fixed per config.

    Padding uses zeros for coordinates, False for masks.
    """

    def __call__(self, samples: list[dict]) -> dict:
        B = len(samples)
        T = samples[0]['coords'].shape[0]
        max_atoms = max(s['coords'].shape[1] for s in samples)
        max_tokens = max(s['s_trunk'].shape[0] for s in samples)

        batch = {
            'coords': torch.zeros(B, T, max_atoms, 3),
            'timestamps': torch.stack([s['timestamps'] for s in samples]),
            'sigma': torch.stack([s['sigma'] for s in samples]),
            'conditioning_mask': torch.stack([s['conditioning_mask'] for s in samples]),
            's_trunk': torch.zeros(B, max_tokens, 384),
            'z_trunk': torch.zeros(B, max_tokens, max_tokens, 128),
            's_inputs': torch.zeros(B, max_tokens, 384),
            'atom_pad_mask': torch.zeros(B, max_atoms, dtype=torch.bool),
            'observed_atom_mask': torch.zeros(B, max_atoms, dtype=torch.bool),
            'token_pad_mask': torch.zeros(B, max_tokens, dtype=torch.bool),
            'atom_to_token': torch.zeros(B, max_atoms, max_tokens),
            'mol_type_per_atom': torch.zeros(B, max_atoms, dtype=torch.long),
            'split': [s['split'] for s in samples],
            'feats': self._collate_feats(samples, max_atoms, max_tokens),
        }

        for i, s in enumerate(samples):
            na, nt = s['coords'].shape[1], s['s_trunk'].shape[0]
            batch['coords'][i, :, :na] = s['coords']
            batch['s_trunk'][i, :nt] = s['s_trunk']
            batch['z_trunk'][i, :nt, :nt] = s['z_trunk']
            batch['s_inputs'][i, :nt] = s['s_inputs']
            batch['atom_pad_mask'][i, :na] = s['atom_pad_mask']
            batch['observed_atom_mask'][i, :na] = s['observed_atom_mask']
            batch['token_pad_mask'][i, :nt] = s['token_pad_mask']
            batch['atom_to_token'][i, :na, :nt] = s['atom_to_token']
            batch['mol_type_per_atom'][i, :na] = s['mol_type_per_atom']
            # Copy feats (ref_pos, ref_charge, etc.) with the same padded indexing

        return batch
```

`_collate_feats(...)` is a required helper that pads every featurizer field to `(max_atoms, max_tokens)`-compatible shapes while preserving the original masks.

### 2.7 Batch Contract Matrix (Normative)

| Key | Shape / Type | Required | Producer | Consumer |
|-----|--------------|----------|----------|----------|
| `coords` | `(B, T, M, 3)` float32/float16 (Angstrom) | Yes | Dataset + Collator | Model, losses |
| `timestamps` | `(B, T)` float32 (ns) | Yes | Dataset + Collator | Temporal attention, sampler |
| `sigma` | `(B, T)` float32 | Yes | Dataset + Collator | EDM scaling, losses |
| `conditioning_mask` | `(B, T)` bool | Yes | Dataset + Collator | Noise-as-masking, losses |
| `s_trunk` | `(B, N, 384)` fp16/fp32 | Yes | Trunk cache loader | Diffusion conditioning |
| `z_trunk` | `(B, N, N, 128)` fp16/fp32 | Yes | Trunk cache loader | Diffusion conditioning |
| `s_inputs` | `(B, N, 384)` fp16/fp32 | Yes | Trunk cache loader | Single conditioning |
| `atom_pad_mask` | `(B, M)` bool | Yes | Dataset + Collator | Attention/loss masking |
| `observed_atom_mask` | `(B, M)` bool | Yes | Preprocessing + Dataset | Loss masking |
| `token_pad_mask` | `(B, N)` bool | Yes | Dataset + Collator | Token attention |
| `atom_to_token` | `(B, M, N)` float32 | Yes | Tokenizer + Collator | Atom-token aggregation |
| `mol_type_per_atom` | `(B, M)` int64 | Yes | Dataset + Collator | Loss weighting |
| `feats` | dict | Yes | Featurizer | Diffusion conditioning |
| `bond_indices` | `(B, nb, 2)` int64 | Optional | Preprocessing | Bond loss |
| `bond_lengths` | `(B, nb)` float32 (Angstrom) | Optional | Preprocessing | Bond loss |

`rel_pos_enc` is intentionally excluded from the runtime batch contract.

### 2.8 Dataset Split and Leakage Policy

Every processed system must be assigned exactly one split in a manifest:
`train`, `val`, `test`, `atlas_ood`, `misato_ood`.

Rules:

1. Enforce sequence-identity filtering for OOD (<=40% identity vs training proteins).
2. Prevent complex-level leakage by deduplicating on `(protein sequence hash, ligand InChIKey, oligomeric state)`.
3. Keep all trajectories from the same molecular system in a single split.
4. Exclude unresolved atoms/residues from supervised terms via `observed_atom_mask`.
5. Report split sizes and leakage checks in preprocessing logs.
6. CATH2 splits should be stratified by CATH classification (class → architecture → topology → homologous superfamily) to ensure fold-class coverage in all splits.
7. MegaSim wildtype/mutant pairs must stay in the same split (never leak a wildtype into train and its mutant into test).
8. Octapeptides: all replicas of the same sequence must be in the same split.
9. Cross-dataset deduplication: check CATH2 domains against ATLAS chains for sequence overlap (>40% identity). Assign overlapping systems to the same split or exclude duplicates from one dataset.

---

## 3. Architecture Implementation

### 3.1 TemporalAttentionWithDecay

**File:** `src/boltzkinema/model/temporal_attention.py`

This is the core new module. It performs multi-head attention across the time dimension for each spatial position (atom or token), with a learnable exponential decay bias.

```python
class TemporalAttentionWithDecay(nn.Module):
    """
    Temporal attention across frames for each atom/token position.

    Bias: B_ij^(h) = -lambda_h * |t_i - t_j|
    After softmax: A_ij proportional to exp(QK^T/sqrt(d)) * exp(-lambda_h * |t_i - t_j|)

    Args:
        dim: int - input/output feature dimension
            Token-level: 2 * token_s = 768
            Atom-level: atom_s = 128
        n_heads: int - number of attention heads
            Token-level: 16
            Atom-level: 4
        causal: bool - if True, mask future frames (for metadynamics)

    Initialization:
        - Q, K, V projections: default Xavier/Kaiming init
        - Output projection: ZERO initialized (critical for training stability)
        - Lambda decay factors: geometric sequence from 0.004 to 0.7 (ALiBi-style)

    Input shapes:
        x: (B*N, T, dim) - features for each position across frames
            N = N_atoms for atom-level, N_tokens for token-level
        timestamps: (B, T) - continuous float timestamps in nanoseconds

    Output shape: (B*N, T, dim)
    """

    def __init__(self, dim: int, n_heads: int, causal: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.causal = causal
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(dim, elementwise_affine=True)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # Output projection — ZERO INITIALIZED for residual safety
        self.to_out = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.to_out.weight)

        # Learnable decay factors: log(lambda) as parameter
        # Geometric sequence from 0.004 to 0.7
        log_lambdas = torch.linspace(math.log(0.004), math.log(0.7), n_heads)
        self.log_lambda = nn.Parameter(log_lambdas)

    def forward(self, x, timestamps, n_spatial):
        """
        Args:
            x: (B*N, T, C) - already reshaped from (B, T, N, C) by caller
            timestamps: (B, T) - in nanoseconds
            n_spatial: int - N (atoms or tokens), needed to reconstruct B
        Returns: (B*N, T, C)
        """
        BN, T, C = x.shape
        B = BN // n_spatial

        x_normed = self.norm(x)
        Q = self.to_q(x_normed).reshape(BN, T, self.n_heads, self.head_dim)
        K = self.to_k(x_normed).reshape(BN, T, self.n_heads, self.head_dim)
        V = self.to_v(x_normed).reshape(BN, T, self.n_heads, self.head_dim)

        # Attention scores: (BN, H, T, T)
        scores = torch.einsum('bthd,bshd->bhts', Q, K) * self.scale

        # Exponential decay bias
        lambda_h = torch.exp(self.log_lambda)  # (H,) positive
        # |t_i - t_j|: (B, T, T)
        dt = torch.abs(timestamps.unsqueeze(-1) - timestamps.unsqueeze(-2))
        bias = -lambda_h.view(1, -1, 1, 1) * dt.unsqueeze(1)  # (B, H, T, T)
        # Expand to (BN, H, T, T): same bias for all atoms/tokens within a batch
        bias = bias.repeat_interleave(n_spatial, dim=0)

        scores = scores + bias

        # Causal mask (for metadynamics/unbinding)
        if self.causal:
            # timestamps are ordered; mask where t_j > t_i
            causal_mask = timestamps.unsqueeze(-1) < timestamps.unsqueeze(-2)  # (B, T, T)
            causal_mask = causal_mask.unsqueeze(1).repeat_interleave(n_spatial, dim=0)
            scores.masked_fill_(causal_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('bhts,bshd->bthd', attn, V)
        out = out.reshape(BN, T, C)

        return self.to_out(out)
```

### 3.2 SpatialTemporalTokenTransformerBlock

**File:** `src/boltzkinema/model/spatial_temporal_transformer.py`

Each block performs: spatial attention → temporal attention → conditioned transition.

```python
class SpatialTemporalTokenTransformerBlock(nn.Module):
    """
    One block of the spatial-temporal token transformer.

    Sequence: spatial attention (pair-biased) → temporal attention (decay) → FFN.

    The spatial components are initialized from Boltz-2 pretrained weights.
    The temporal components are initialized from scratch (zero output proj).

    Args:
        dim: int = 768 (2 * token_s)
        dim_single_cond: int = 768
        n_spatial_heads: int = 16 (from Boltz-2 token_transformer_heads)
        n_temporal_heads: int = 16 (from BioKinema paper)
        causal: bool = False
    """

    def __init__(self, dim, dim_single_cond, n_spatial_heads, n_temporal_heads, causal=False):
        super().__init__()
        # === Spatial attention (from Boltz-2's DiffusionTransformerLayer) ===
        # These are imported and initialized from pretrained weights
        from boltz.model.modules.transformersv2 import AdaLN, ConditionedTransitionBlock
        from boltz.model.layers.attentionv2 import AttentionPairBias

        self.adaln = AdaLN(dim, dim_single_cond)
        self.spatial_attn = AttentionPairBias(
            c_s=dim, num_heads=n_spatial_heads, compute_pair_bias=False
        )
        # Output gating (matches Boltz-2's DiffusionTransformerLayer)
        self.spatial_output_projection = nn.Sequential(
            nn.Linear(dim_single_cond, dim),  # init: zeros weight, -2.0 bias
            nn.Sigmoid()
        )

        # === Temporal attention (NEW) ===
        self.temporal_attn = TemporalAttentionWithDecay(
            dim=dim, n_heads=n_temporal_heads, causal=causal
        )

        # === Conditioned transition (from Boltz-2) ===
        self.transition = ConditionedTransitionBlock(dim, dim_single_cond)
        self.post_lnorm = nn.Identity()  # unless post_layer_norm=True

    def forward(self, a, s, bias, timestamps, mask, n_tokens, to_keys=None, multiplicity=1):
        """
        Args:
            a: (B, T, N, dim) token activations
            s: (B*mult, N, dim) single conditioning (sigma-aware, shared across frames)
            bias: (B, N, N, n_heads) or (B, N, M, n_heads) pair bias for this layer
            timestamps: (B, T)
            mask: (B, N) token padding mask
            n_tokens: int
        Returns: (B, T, N, dim)
        """
        B, T, N, C = a.shape

        # 1. Spatial attention (per-frame, treating T frames as batch)
        a_flat = a.reshape(B * T, N, C)
        # Expand s for T frames: s is (B, N, C), need (B*T, N, C)
        s_expanded = s.unsqueeze(1).expand(B, T, N, C).reshape(B * T, N, C)

        b = self.adaln(a_flat, s_expanded)
        k_in = b
        if to_keys is not None:
            k_in = to_keys(b)
        b = self.spatial_attn(s=b, z=bias, mask=mask, k_in=k_in, multiplicity=1)
        b = self.spatial_output_projection(s_expanded) * b
        a_flat = a_flat + b
        a = a_flat.reshape(B, T, N, C)

        # 2. Temporal attention (per-token across frames)
        a_transposed = a.permute(0, 2, 1, 3).reshape(B * N, T, C)
        a_transposed = a_transposed + self.temporal_attn(
            a_transposed, timestamps, n_spatial=N
        )
        a = a_transposed.reshape(B, N, T, C).permute(0, 2, 1, 3)

        # 3. Conditioned transition (per-frame)
        a_flat = a.reshape(B * T, N, C)
        a_flat = a_flat + self.transition(a_flat, s_expanded)
        a_flat = self.post_lnorm(a_flat)
        a = a_flat.reshape(B, T, N, C)

        return a
```

### 3.3 SpatialTemporalAtomBlock

**File:** `src/boltzkinema/model/spatial_temporal_atom.py`

Atom-level blocks use windowed (sequence-local) spatial attention + temporal attention.

```python
class SpatialTemporalAtomEncoderBlock(nn.Module):
    """
    One block of atom-level spatial-temporal processing in the encoder.

    Spatial: windowed atom attention (W=32 queries, H=128 keys)
    Temporal: full attention across T frames per atom

    Args:
        atom_s: int = 128
        n_spatial_heads: int = 4 (from Boltz-2)
        n_temporal_heads: int = 4 (from BioKinema paper)
        atoms_per_window_queries: int = 32
        atoms_per_window_keys: int = 128
        causal: bool = False
    """

    def __init__(self, atom_s, n_spatial_heads, n_temporal_heads,
                 atoms_per_window_queries, atoms_per_window_keys, causal=False):
        super().__init__()
        # Spatial: reuse Boltz-2's AtomTransformer layer internals
        from boltz.model.modules.transformersv2 import DiffusionTransformerLayer

        self.spatial_layer = DiffusionTransformerLayer(
            heads=n_spatial_heads, dim=atom_s, dim_single_cond=atom_s
        )
        self.W = atoms_per_window_queries
        self.H = atoms_per_window_keys

        # Temporal
        self.temporal_attn = TemporalAttentionWithDecay(
            dim=atom_s, n_heads=n_temporal_heads, causal=causal
        )

    def forward(self, q, c, bias, to_keys, mask, timestamps, n_atoms):
        """
        Args:
            q: (B, T, M, atom_s) atom queries
            c: (B, T, M, atom_s) atom conditioning (expanded for T frames)
            bias: (B, K, W, H, n_heads) windowed pair bias for this layer
            to_keys: callable for windowed key selection
            mask: (B, M) atom padding mask
            timestamps: (B, T)
            n_atoms: int = M
        Returns: (B, T, M, atom_s)
        """
        B, T, M, D = q.shape

        # 1. Spatial attention (per-frame, windowed)
        q_flat = q.reshape(B * T, M, D)
        c_flat = c.reshape(B * T, M, D)

        # Window reshape: (B*T, M, D) -> (B*T*NW, W, D) where NW = M/W
        NW = M // self.W
        q_win = q_flat.reshape(B * T * NW, self.W, D)
        c_win = c_flat.reshape(B * T * NW, self.W, D)
        mask_win = mask.unsqueeze(1).expand(B, T, M).reshape(B * T, M)
        mask_win = mask_win.reshape(B * T * NW, self.W)

        # Apply spatial DiffusionTransformerLayer within windows
        # (need to handle bias and to_keys windowing)
        q_win = self.spatial_layer(q_win, s=c_win, bias=bias, mask=mask_win,
                                    to_keys=to_keys, multiplicity=1)
        q_flat = q_win.reshape(B * T, M, D)
        q = q_flat.reshape(B, T, M, D)

        # 2. Temporal attention (per-atom across frames)
        q_transposed = q.permute(0, 2, 1, 3).reshape(B * M, T, D)
        q_transposed = q_transposed + self.temporal_attn(
            q_transposed, timestamps, n_spatial=M
        )
        q = q_transposed.reshape(B, M, T, D).permute(0, 2, 1, 3)

        return q
```

### 3.4 SpatialTemporalDiffusionModule

**File:** `src/boltzkinema/model/diffusion_module.py`

This assembles the full spatial-temporal score network.

```python
class SpatialTemporalDiffusionModule(nn.Module):
    """
    The complete spatial-temporal diffusion score model.

    Replaces Boltz-2's DiffusionModule with multi-frame processing.

    Architecture:
        1. SingleConditioning: inject sigma into token representations
        2. SpatialTemporalAtomEncoder (3 blocks): atom attention + temporal → token aggregation
        3. SpatialTemporalTokenTransformer (24 blocks): token attention + temporal + FFN
        4. SpatialTemporalAtomDecoder (3 blocks): broadcast + atom attention + temporal → coords

    Args:
        token_s: int = 384
        atom_s: int = 128
        atoms_per_window_queries: int = 32
        atoms_per_window_keys: int = 128
        sigma_data: float = 16.0
        dim_fourier: int = 256
        atom_encoder_depth: int = 3
        atom_encoder_heads: int = 4
        atom_temporal_heads: int = 4
        token_transformer_depth: int = 24
        token_transformer_heads: int = 16
        token_temporal_heads: int = 16
        atom_decoder_depth: int = 3
        atom_decoder_heads: int = 4
        conditioning_transition_layers: int = 2
        causal: bool = False
        activation_checkpointing: bool = False
    """

    def __init__(self, ...):
        super().__init__()

        # --- SingleConditioning (reuse from Boltz-2) ---
        # This embeds the per-frame sigma into the token conditioning signal
        from boltz.model.modules.encodersv2 import SingleConditioning
        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data, token_s=token_s, dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers
        )

        # --- Atom Attention Encoder (modified for spatial-temporal) ---
        # Reuse Boltz-2's AtomAttentionEncoder but wrap each layer with temporal
        from boltz.model.modules.encodersv2 import AtomAttentionEncoder
        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s, token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=True,
        )

        # Add temporal attention after each encoder layer
        self.encoder_temporal_layers = nn.ModuleList([
            TemporalAttentionWithDecay(dim=atom_s, n_heads=atom_temporal_heads, causal=causal)
            for _ in range(atom_encoder_depth)
        ])

        # --- s_to_a projection (from Boltz-2) ---
        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(2 * token_s),
            nn.Linear(2 * token_s, 2 * token_s, bias=False)  # init zeros
        )
        nn.init.zeros_(self.s_to_a_linear[1].weight)

        # --- Token Transformer (24 spatial-temporal blocks) ---
        self.token_transformer_blocks = nn.ModuleList([
            SpatialTemporalTokenTransformerBlock(
                dim=2 * token_s,
                dim_single_cond=2 * token_s,
                n_spatial_heads=token_transformer_heads,
                n_temporal_heads=token_temporal_heads,
                causal=causal,
            )
            for _ in range(token_transformer_depth)
        ])

        self.a_norm = nn.LayerNorm(2 * token_s)

        # --- Atom Attention Decoder (modified for spatial-temporal) ---
        from boltz.model.modules.encodersv2 import AtomAttentionDecoder
        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_s=atom_s, token_s=token_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
        )

        # Add temporal attention after each decoder layer
        self.decoder_temporal_layers = nn.ModuleList([
            TemporalAttentionWithDecay(dim=atom_s, n_heads=atom_temporal_heads, causal=causal)
            for _ in range(atom_decoder_depth)
        ])

        self.activation_checkpointing = activation_checkpointing

    def forward(self, s_inputs, s_trunk, r_noisy, sigma, timestamps, feats,
                diffusion_conditioning):
        """
        Args:
            s_inputs: (B, N, token_s) raw input features
            s_trunk: (B, N, token_s) refined single representation
            r_noisy: (B, T, M, 3) noisy atom coordinates (already c_in-scaled)
            sigma: (B, T) per-frame noise levels
            timestamps: (B, T) continuous timestamps in ns
            feats: dict with atom-level features (ref_pos, masks, etc.)
            diffusion_conditioning: dict with q, c, to_keys, biases
        Returns:
            r_update: (B, T, M, 3) coordinate updates
        """
        B, T, M, _ = r_noisy.shape
        N = s_trunk.shape[1]

        # --- Step 1: Per-frame sigma conditioning ---
        # SingleConditioning expects (B, ) times and (B, N, token_s) trunk features
        # We process each frame's sigma independently
        # Reshape: treat B*T as batch for SingleConditioning
        sigma_flat = sigma.reshape(B * T)  # (B*T,)
        c_noise = torch.log(sigma_flat / self.sigma_data) * 0.25  # EDM c_noise
        # Handle sigma=0 (conditioning frames): use a small value to avoid log(0)
        c_noise = torch.where(sigma_flat > 0, c_noise, torch.zeros_like(c_noise))

        s_trunk_expanded = s_trunk.unsqueeze(1).expand(B, T, N, -1).reshape(B * T, N, -1)
        s_inputs_expanded = s_inputs.unsqueeze(1).expand(B, T, N, -1).reshape(B * T, N, -1)

        s_cond, _ = self.single_conditioner(c_noise, s_trunk_expanded, s_inputs_expanded)
        # s_cond: (B*T, N, 2*token_s=768)
        s_cond = s_cond.reshape(B, T, N, -1)

        # --- Step 2: Atom Attention Encoder ---
        # Process each frame through atom encoder, then apply temporal attention
        r_flat = r_noisy.reshape(B * T, M, 3)
        q = diffusion_conditioning["q"]  # (B, M, atom_s)
        c = diffusion_conditioning["c"]  # (B, M, atom_s)

        # Run atom encoder per-frame (spatial)
        # Then interleave temporal attention after each spatial layer
        # (This requires modifying the AtomAttentionEncoder to expose per-layer hooks,
        #  or re-implementing the encoder loop here with temporal insertions)
        a_tokens, q_skip, c_skip, to_keys = self._atom_encode_with_temporal(
            r_flat, q, c, diffusion_conditioning, feats, timestamps, B, T, M
        )
        # a_tokens: (B, T, N, 2*token_s)

        # --- Step 3: Add sigma conditioning ---
        a_tokens = a_tokens + self.s_to_a_linear(s_cond)

        # --- Step 4: Token Transformer (24 blocks with temporal) ---
        # Split the stacked token_trans_bias into per-layer biases
        token_bias = diffusion_conditioning["token_trans_bias"]
        # token_bias: (B, N, N, 24*16=384) -> split to (B, N, N, 16) per layer
        token_bias_per_layer = token_bias.chunk(len(self.token_transformer_blocks), dim=-1)

        mask = feats["token_pad_mask"]  # (B, N)

        for i, block in enumerate(self.token_transformer_blocks):
            if self.activation_checkpointing:
                a_tokens = torch.utils.checkpoint.checkpoint(
                    block, a_tokens, s_cond, token_bias_per_layer[i],
                    timestamps, mask, N, use_reentrant=False
                )
            else:
                a_tokens = block(
                    a_tokens, s_cond.reshape(B*T, N, -1),
                    token_bias_per_layer[i], timestamps, mask, N
                )

        a_tokens_normed = self.a_norm(a_tokens.reshape(B*T, N, -1)).reshape(B, T, N, -1)

        # --- Step 5: Atom Attention Decoder ---
        r_update = self._atom_decode_with_temporal(
            a_tokens_normed, q_skip, c_skip, diffusion_conditioning,
            feats, timestamps, to_keys, B, T, M
        )
        # r_update: (B, T, M, 3)

        return r_update
```

### 3.5 Per-Frame EDM Preconditioning

**File:** `src/boltzkinema/model/edm.py`

```python
class PerFrameEDM(nn.Module):
    """
    EDM preconditioning applied independently per frame.

    Each frame has its own sigma. Conditioning frames have sigma=0,
    which means c_skip=1, c_out=0 (output is just the input = clean coords).

    Args:
        sigma_data: float = 16.0

    EDM equations:
        c_in(sigma) = 1 / sqrt(sigma^2 + sigma_data^2)
        c_skip(sigma) = sigma_data^2 / (sigma^2 + sigma_data^2)
        c_out(sigma) = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
        c_noise(sigma) = log(sigma / sigma_data) * 0.25

        D_theta(x; sigma) = c_skip(sigma) * x + c_out(sigma) * F_theta(c_in(sigma) * x; c_noise(sigma))
    """

    def __init__(self, sigma_data: float = 16.0):
        super().__init__()
        self.sigma_data = sigma_data

    def c_in(self, sigma):
        """Input scaling. sigma: (B, T)"""
        return 1.0 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_skip(self, sigma):
        """Skip connection weight. sigma: (B, T)"""
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        """Output scaling. sigma: (B, T)"""
        return sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)

    def loss_weight(self, sigma):
        """EDM loss weighting. sigma: (B, T)"""
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data + 1e-8)**2

    def scale_input(self, x, sigma):
        """
        x: (B, T, M, 3)
        sigma: (B, T) -> broadcast to (B, T, 1, 1)
        """
        c = self.c_in(sigma).unsqueeze(-1).unsqueeze(-1)
        return x * c

    def combine_output(self, x_noisy, f_out, sigma):
        """
        x_noisy: (B, T, M, 3) - the noisy input coordinates
        f_out: (B, T, M, 3) - raw network output (r_update)
        sigma: (B, T)
        Returns: (B, T, M, 3) denoised coordinates
        """
        cs = self.c_skip(sigma).unsqueeze(-1).unsqueeze(-1)
        co = self.c_out(sigma).unsqueeze(-1).unsqueeze(-1)
        return cs * x_noisy + co * f_out

    def sample_noise(self, shape, P_mean=-1.2, P_std=1.5):
        """Sample noise levels from log-normal distribution."""
        log_sigma = torch.randn(shape) * P_std + P_mean
        return self.sigma_data * torch.exp(log_sigma)
```

### 3.6 Top-Level BoltzKinema Model

**File:** `src/boltzkinema/model/boltzkinema.py`

```python
class BoltzKinema(nn.Module):
    """
    Top-level BoltzKinema model.

    Wraps:
    - Boltz-2 DiffusionConditioning (frozen or fine-tuned)
    - SpatialTemporalDiffusionModule (trainable)
    - PerFrameEDM (preconditioning)

    The Boltz-2 trunk is NOT part of this module (embeddings are precomputed).

    Weight initialization strategy:
    - Spatial attention layers (in encoder, transformer, decoder):
        Load from Boltz-2 pretrained structure_module.score_model.*
    - Temporal attention layers:
        Random init Q/K/V, ZERO init output projection
    - SingleConditioning:
        Load from Boltz-2 pretrained structure_module.score_model.single_conditioner.*
    - DiffusionConditioning:
        Load from Boltz-2 pretrained diffusion_conditioning.*
    """

    def __init__(self, config):
        super().__init__()

        self.edm = PerFrameEDM(sigma_data=config.sigma_data)

        # DiffusionConditioning from Boltz-2 (computes q, c, biases from trunk features)
        from boltz.model.modules.diffusion_conditioning import DiffusionConditioning
        self.diffusion_conditioning = DiffusionConditioning(
            token_s=config.token_s,
            token_z=config.token_z,
            atom_s=config.atom_s,
            atom_z=config.atom_z,
            atoms_per_window_queries=config.atoms_per_window_queries,
            atoms_per_window_keys=config.atoms_per_window_keys,
            atom_encoder_depth=config.atom_encoder_depth,
            atom_encoder_heads=config.atom_encoder_heads,
            token_transformer_depth=config.token_transformer_depth,
            token_transformer_heads=config.token_transformer_heads,
            atom_decoder_depth=config.atom_decoder_depth,
            atom_decoder_heads=config.atom_decoder_heads,
            atom_feature_dim=config.atom_feature_dim,
            conditioning_transition_layers=config.conditioning_transition_layers,
        )

        # Spatial-Temporal Diffusion Module (the trainable score model)
        self.score_model = SpatialTemporalDiffusionModule(
            token_s=config.token_s,
            atom_s=config.atom_s,
            atoms_per_window_queries=config.atoms_per_window_queries,
            atoms_per_window_keys=config.atoms_per_window_keys,
            sigma_data=config.sigma_data,
            dim_fourier=config.dim_fourier,
            atom_encoder_depth=config.atom_encoder_depth,
            atom_encoder_heads=config.atom_encoder_heads,
            atom_temporal_heads=config.atom_temporal_heads,
            token_transformer_depth=config.token_transformer_depth,
            token_transformer_heads=config.token_transformer_heads,
            token_temporal_heads=config.token_temporal_heads,
            atom_decoder_depth=config.atom_decoder_depth,
            atom_decoder_heads=config.atom_decoder_heads,
            conditioning_transition_layers=config.conditioning_transition_layers,
            causal=config.causal,
            activation_checkpointing=config.activation_checkpointing,
        )

    def forward(self, batch):
        """
        Full forward pass for training.

        1. Compute diffusion conditioning from precomputed trunk features
        2. Construct noisy trajectory
        3. Apply EDM preconditioning per-frame
        4. Run spatial-temporal score model
        5. Apply EDM output combination per-frame

        Returns: dict with denoised coords, sigma, conditioning_mask
        """
        coords = batch['coords']              # (B, T, M, 3)
        timestamps = batch['timestamps']       # (B, T)
        sigma = batch['sigma']                 # (B, T)
        cond_mask = batch['conditioning_mask']  # (B, T)
        s_trunk = batch['s_trunk']             # (B, N, 384)
        z_trunk = batch['z_trunk']             # (B, N, N, 128)
        s_inputs = batch['s_inputs']           # (B, N, 384)

        # 1. Diffusion conditioning (from trunk, shared across frames)
        diff_cond = self.diffusion_conditioning(
            s_trunk=s_trunk, z_trunk=z_trunk,
            feats=batch['feats'],
        )
        # diff_cond: dict with q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias

        # 2. Construct noisy trajectory
        eps = torch.randn_like(coords)
        sigma_expanded = sigma.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
        x_noisy = coords + sigma_expanded * eps
        # Clean conditioning frames
        cond_expanded = cond_mask.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
        x_noisy = torch.where(cond_expanded, coords, x_noisy)

        # 3. EDM input scaling
        r_noisy = self.edm.scale_input(x_noisy, sigma)  # (B, T, M, 3)

        # 4. Run score model
        r_update = self.score_model(
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            r_noisy=r_noisy,
            sigma=sigma,
            timestamps=timestamps,
            feats=batch['feats'],
            diffusion_conditioning=diff_cond,
        )

        # 5. EDM output combination
        x_denoised = self.edm.combine_output(x_noisy, r_update, sigma)

        return {
            'x_denoised': x_denoised,     # (B, T, M, 3)
            'sigma': sigma,
            'conditioning_mask': cond_mask,
        }
```

### 3.7 Weight Loading from Boltz-2

**File:** `src/boltzkinema/model/weight_loading.py`

```python
def load_boltz2_weights(model: BoltzKinema, checkpoint_path: str):
    """
    Load pretrained Boltz-2 weights into BoltzKinema.

    Mapping:
    - diffusion_conditioning.* -> model.diffusion_conditioning.*
    - structure_module.score_model.single_conditioner.* -> model.score_model.single_conditioner.*
    - structure_module.score_model.atom_attention_encoder.* ->
        model.score_model.atom_attention_encoder.* (spatial layers only)
    - structure_module.score_model.token_transformer.layers.{i}.* ->
        model.score_model.token_transformer_blocks.{i}.{spatial components}.*
    - structure_module.score_model.atom_attention_decoder.* ->
        model.score_model.atom_attention_decoder.* (spatial layers only)
    - structure_module.score_model.s_to_a_linear.* -> model.score_model.s_to_a_linear.*
    - structure_module.score_model.a_norm.* -> model.score_model.a_norm.*

    Temporal attention layers are NOT loaded (they start from zero-init output).
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['state_dict']

    # Build mapping
    weight_map = {}

    # DiffusionConditioning: direct copy
    for key in state_dict:
        if key.startswith('diffusion_conditioning.'):
            weight_map[key] = key  # same prefix in our model

    # SingleConditioning
    for key in state_dict:
        if key.startswith('structure_module.score_model.single_conditioner.'):
            new_key = key.replace('structure_module.score_model.', 'score_model.')
            weight_map[key] = new_key

    # Atom encoder spatial components
    for key in state_dict:
        if key.startswith('structure_module.score_model.atom_attention_encoder.'):
            new_key = key.replace('structure_module.score_model.', 'score_model.')
            weight_map[key] = new_key

    # Token transformer: map Boltz-2 DiffusionTransformerLayer -> our SpatialTemporalBlock
    for key in state_dict:
        if key.startswith('structure_module.score_model.token_transformer.layers.'):
            # Extract layer index and sub-key
            # e.g. "structure_module.score_model.token_transformer.layers.5.adaln.a_norm.weight"
            parts = key.split('.')
            layer_idx = parts[4]
            sub_key = '.'.join(parts[5:])

            # Map adaln -> adaln, pair_bias_attn -> spatial_attn,
            # output_projection -> spatial_output_projection, transition -> transition
            if sub_key.startswith('adaln.'):
                new_key = f'score_model.token_transformer_blocks.{layer_idx}.adaln.{sub_key[6:]}'
            elif sub_key.startswith('pair_bias_attn.'):
                new_key = f'score_model.token_transformer_blocks.{layer_idx}.spatial_attn.{sub_key[14:]}'
            elif sub_key.startswith('output_projection.'):
                new_key = f'score_model.token_transformer_blocks.{layer_idx}.spatial_output_projection.{sub_key[18:]}'
            elif sub_key.startswith('transition.'):
                new_key = f'score_model.token_transformer_blocks.{layer_idx}.transition.{sub_key[11:]}'
            else:
                continue
            weight_map[key] = new_key

    # Atom decoder spatial components
    for key in state_dict:
        if key.startswith('structure_module.score_model.atom_attention_decoder.'):
            new_key = key.replace('structure_module.score_model.', 'score_model.')
            weight_map[key] = new_key

    # s_to_a and a_norm
    for key in state_dict:
        if key.startswith('structure_module.score_model.s_to_a_linear.'):
            new_key = key.replace('structure_module.score_model.', 'score_model.')
            weight_map[key] = new_key
        if key.startswith('structure_module.score_model.a_norm.'):
            new_key = key.replace('structure_module.score_model.', 'score_model.')
            weight_map[key] = new_key

    # Load mapped weights
    new_state = model.state_dict()
    loaded, skipped = 0, 0
    for old_key, new_key in weight_map.items():
        if new_key in new_state and state_dict[old_key].shape == new_state[new_key].shape:
            new_state[new_key] = state_dict[old_key]
            loaded += 1
        else:
            skipped += 1
            print(f"Skipped: {old_key} -> {new_key}")

    model.load_state_dict(new_state, strict=False)
    print(f"Loaded {loaded} weights, skipped {skipped}")

    # Verify temporal layers are zero-initialized
    for name, param in model.named_parameters():
        if 'temporal_attn.to_out.weight' in name:
            assert (param == 0).all(), f"Temporal output not zero: {name}"
```

---

## 4. Training

### 4.1 Loss Functions

**File:** `src/boltzkinema/training/losses.py`

```python
class BoltzKinemaLoss(nn.Module):
    """
    Combined loss for BoltzKinema training.

    Components:
    1. L_struct: EDM-weighted structure reconstruction (MSE + bond + smooth lDDT)
    2. L_flex: Flexibility loss (RMSF + pairwise distance std + local)
    3. L_center: Ligand geometric center loss (unbinding only)

    Loss masking policy:
    - Conditioning frames (`sigma=0`) do not contribute to supervised terms.
    - Unobserved atoms/residues do not contribute to supervised terms.

    MD training:     L = L_struct + beta_flex * L_flex
    Unbinding:       L = L_struct + beta_center * L_center

    Args:
        sigma_data: float = 16.0
        alpha_bond: float = 1.0
        beta_flex: float = 1.0
        beta_abs: float = 1.0 (absolute RMSF weight within flex)
        beta_rel_g: float = 4.0 (global relative weight)
        beta_rel_l: float = 4.0 (local relative weight)
        beta_center: float = 1.0 (ligand center weight, unbinding only)
        mol_weights: dict = {'protein': 1.0, 'dna': 5.0, 'rna': 5.0, 'ligand': 10.0}
    """

    def structure_loss(self, x_pred, x_gt, sigma, cond_mask, atom_mask, observed_mask, mol_weights):
        """
        EDM-weighted MSE on non-conditioning frames.

        x_pred, x_gt: (B, T, M, 3)
        sigma: (B, T)
        cond_mask: (B, T) bool
        atom_mask: (B, M) bool
        observed_mask: (B, M) bool
        mol_weights: (B, M) per-atom weights

        L = sum over non-cond frames of:
            [(sigma^2 + sigma_data^2) / (sigma * sigma_data)^2] *
            mean_atoms[w_atom * ||x_pred - x_gt||^2]
        """
        target_mask = ~cond_mask  # (B, T)

        valid_atom_mask = atom_mask & observed_mask

        # Per-atom squared error
        sq_err = ((x_pred - x_gt) ** 2).sum(dim=-1)  # (B, T, M)
        # Apply atom mask and molecule-type weights
        weighted_err = sq_err * mol_weights.unsqueeze(1) * valid_atom_mask.unsqueeze(1).float()
        # Mean over atoms
        per_frame_loss = weighted_err.sum(dim=-1) / (valid_atom_mask.float().sum(dim=-1, keepdim=True) + 1e-8).squeeze(-1)

        # EDM loss weight (only for non-conditioning frames)
        edm_weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data + 1e-8)**2
        edm_weight = edm_weight * target_mask.float()

        l_struct = (edm_weight * per_frame_loss).sum() / (target_mask.float().sum() + 1e-8)
        return l_struct

    def bond_loss(self, x_pred, batch, target_frame_mask):
        """
        Covalent bond length loss for ligand-protein connections.
        Penalizes deviations from reference bond lengths.

        Uses batch['bond_indices'] (pairs of bonded atoms) and
        batch['bond_lengths'] (reference lengths in Angstroms).
        """
        if 'bond_indices' not in batch:
            return torch.tensor(0.0, device=x_pred.device)

        idx_i, idx_j = batch['bond_indices'].unbind(-1)  # (B, n_bonds)
        ref_lengths = batch['bond_lengths']  # (B, n_bonds)

        # Compute predicted bond lengths across all frames, then mask to target frames
        B, T, M, _ = x_pred.shape
        x_flat = x_pred.reshape(B, T, M, 3)
        d = x_flat[:, :, idx_i] - x_flat[:, :, idx_j]  # (B, T, n_bonds, 3)
        pred_lengths = d.norm(dim=-1)  # (B, T, n_bonds)

        frame_mask = target_frame_mask.unsqueeze(-1).float()  # (B, T, 1)
        sq = (pred_lengths - ref_lengths.unsqueeze(1)) ** 2
        return (sq * frame_mask).sum() / (frame_mask.sum() * sq.shape[-1] + 1e-8)

    def smooth_lddt_loss(self, x_pred, x_gt, atom_mask, observed_mask, target_frame_mask):
        """
        Differentiable lDDT approximation (AF3 Algorithm 27).

        For each atom pair within 15A (30A for nucleotides):
        score = mean over thresholds [0.5, 1, 2, 4] of:
            sigmoid(threshold - |d_pred - d_gt|)

        Loss = 1 - mean(score)
        """
        B, T, M, _ = x_pred.shape

        # Compute pairwise distances
        d_pred = torch.cdist(x_pred.reshape(B*T, M, 3), x_pred.reshape(B*T, M, 3))
        d_gt = torch.cdist(x_gt.reshape(B*T, M, 3), x_gt.reshape(B*T, M, 3))

        # Distance mask: only valid atoms, target frames, and pairs within 15A
        valid_atom_mask = atom_mask & observed_mask
        valid_atom_mask_bt = valid_atom_mask.unsqueeze(1).expand(B, T, M).reshape(B * T, M)
        frame_mask = target_frame_mask.reshape(B * T, 1, 1)
        dist_mask = (d_gt < 15.0)
        dist_mask = dist_mask & valid_atom_mask_bt.unsqueeze(-1) & valid_atom_mask_bt.unsqueeze(-2)
        dist_mask = dist_mask & frame_mask
        dist_mask = dist_mask & ~torch.eye(M, device=x_pred.device).bool()

        # 4-threshold sigmoid scoring
        diff = torch.abs(d_pred - d_gt)
        thresholds = torch.tensor([0.5, 1.0, 2.0, 4.0], device=x_pred.device)
        scores = torch.sigmoid(thresholds.view(1, 1, 1, -1) - diff.unsqueeze(-1))
        score = scores.mean(dim=-1)  # (B*T, M, M)

        # Masked mean
        lddt = (score * dist_mask.float()).sum() / (dist_mask.float().sum() + 1e-8)
        return 1.0 - lddt

    def flexibility_loss(self, x_pred, x_gt, atom_mask, observed_mask, mol_type_per_atom, target_frame_mask):
        """
        Supervise ensemble distributional properties.

        Three components:
        1. Absolute RMSF: per-atom root-mean-square fluctuation
        2. Global relative: pairwise distance standard deviation
        3. Local relative: intra-residue distance std

        x_pred, x_gt: (B, T, M, 3) - uses target frames only; need >=2 target frames

        Distance-dependent weights for global relative:
            <= 5A: 4x, (5,10]A: 2x, > 10A: 1x
        Molecule-type interaction weights:
            ligand-ligand: 10x, protein-ligand: 5x, protein-protein: 1x

        Combined: L_flex = 1*L_abs + 4*L_rel_global + 4*L_rel_local
        """
        B, T, M, _ = x_pred.shape
        if target_frame_mask.float().sum() < 2:
            return torch.tensor(0.0, device=x_pred.device)
        valid_atom_mask = atom_mask & observed_mask
        frame_w = target_frame_mask.float()
        frame_w = frame_w / (frame_w.sum(dim=1, keepdim=True) + 1e-8)

        # --- Absolute RMSF ---
        pred_mean = (x_pred * frame_w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
        gt_mean = (x_gt * frame_w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)

        rmsf_pred = (((x_pred - pred_mean)**2) * frame_w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1).sum(dim=-1).sqrt()
        rmsf_gt = (((x_gt - gt_mean)**2) * frame_w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1).sum(dim=-1).sqrt()

        l_abs = ((rmsf_pred - rmsf_gt)**2 * valid_atom_mask.float()).mean()

        # --- Global relative (pairwise distance std) ---
        # Use representative atoms (CA for protein, heavy atoms for ligand)
        # For simplicity, compute on all atoms but subsample for memory
        d_pred = torch.cdist(x_pred.reshape(B*T, M, 3),
                              x_pred.reshape(B*T, M, 3)).reshape(B, T, M, M)
        d_gt = torch.cdist(x_gt.reshape(B*T, M, 3),
                            x_gt.reshape(B*T, M, 3)).reshape(B, T, M, M)

        mean_pred = (d_pred * frame_w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
        mean_gt = (d_gt * frame_w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
        std_pred = torch.sqrt((((d_pred - mean_pred) ** 2) * frame_w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1) + 1e-8)
        std_gt = torch.sqrt((((d_gt - mean_gt) ** 2) * frame_w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1) + 1e-8)

        # Distance-dependent weights
        mean_dist = mean_gt.squeeze(1)  # (B, M, M)
        gamma = torch.where(mean_dist <= 5.0, torch.tensor(4.0),
                torch.where(mean_dist <= 10.0, torch.tensor(2.0), torch.tensor(1.0)))

        pair_mask = valid_atom_mask.unsqueeze(-1) & valid_atom_mask.unsqueeze(-2)
        l_rel_global = (gamma * (std_pred - std_gt)**2 * pair_mask.float()).mean()

        # --- Local relative (intra-residue) ---
        # Computed per-residue for side chain atoms (simplified)
        l_rel_local = self._compute_local_flex(x_pred, x_gt, valid_atom_mask)

        return self.beta_abs * l_abs + self.beta_rel_g * l_rel_global + self.beta_rel_l * l_rel_local

    def ligand_center_loss(self, x_pred, x_gt, ligand_mask, target_frame_mask):
        """
        For unbinding training: supervise ligand geometric center trajectory.

        L_center = mean_t ||C(x_pred_t) - C(x_gt_t)||^2
        where C(x) = mean of ligand atom positions
        """
        B, T, M, _ = x_pred.shape

        lig_mask = ligand_mask.unsqueeze(1).unsqueeze(-1).float()  # (B, 1, M, 1)
        n_lig = ligand_mask.float().sum(dim=-1, keepdim=True).unsqueeze(1) + 1e-8

        center_pred = (x_pred * lig_mask).sum(dim=2) / n_lig.squeeze(-1)  # (B, T, 3)
        center_gt = (x_gt * lig_mask).sum(dim=2) / n_lig.squeeze(-1)

        sq = ((center_pred - center_gt) ** 2).sum(dim=-1)
        return (sq * target_frame_mask.float()).sum() / (target_frame_mask.float().sum() + 1e-8)

    def forward(self, output, batch, mode='equilibrium'):
        """
        Compute total loss.

        mode: 'equilibrium' or 'unbinding'
        """
        x_pred = output['x_denoised']
        x_gt = batch['coords']
        sigma = output['sigma']
        cond_mask = output['conditioning_mask']
        atom_mask = batch['atom_pad_mask']
        observed_mask = batch['observed_atom_mask']
        target_frame_mask = ~cond_mask
        mol_weights = self._get_mol_weights(batch['mol_type_per_atom'])

        l_struct = self.structure_loss(
            x_pred, x_gt, sigma, cond_mask, atom_mask, observed_mask, mol_weights
        )
        l_bond = self.bond_loss(x_pred, batch, target_frame_mask)
        l_lddt = self.smooth_lddt_loss(
            x_pred, x_gt, atom_mask, observed_mask, target_frame_mask
        )

        total = l_struct + self.alpha_bond * l_bond + l_lddt

        if mode == 'equilibrium':
            l_flex = self.flexibility_loss(
                x_pred, x_gt, atom_mask, observed_mask,
                batch['mol_type_per_atom'], target_frame_mask
            )
            total = total + self.beta_flex * l_flex
        elif mode == 'unbinding':
            ligand_mask = (batch['mol_type_per_atom'] == 3)  # 3 = ligand
            l_center = self.ligand_center_loss(x_pred, x_gt, ligand_mask, target_frame_mask)
            total = total + self.beta_center * l_center

        return total, {
            'loss': total.item(),
            'l_struct': l_struct.item(),
            'l_bond': l_bond.item(),
            'l_lddt': l_lddt.item(),
        }
```

### 4.2 Training Loop with Accelerate

**File:** `scripts/train.py`

```python
from accelerate import Accelerator
from accelerate.utils import set_seed

def train(config):
    """
    Main training loop using HuggingFace Accelerate.

    Handles:
    - Multi-GPU distributed training (DDP)
    - Mixed precision (bf16)
    - Gradient accumulation
    - Gradient clipping
    - Checkpoint saving/resuming
    - WandB logging
    """
    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with='wandb',
    )

    set_seed(config.seed)

    # Initialize WandB
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name='boltzkinema',
            config=vars(config),
        )

    # Build model
    model = BoltzKinema(config)
    load_boltz2_weights(model, config.boltz2_checkpoint)

    # Freeze DiffusionConditioning if configured
    if config.freeze_diffusion_conditioning:
        for param in model.diffusion_conditioning.parameters():
            param.requires_grad = False

    # Count trainable parameters
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    accelerator.print(f"Trainable: {n_trainable/1e6:.1f}M / Total: {n_total/1e6:.1f}M")

    # Build dataset and dataloader
    dataset = BoltzKinemaDataset(
        manifest_path=config.manifest_path,
        trunk_cache_dir=config.trunk_cache_dir,
        coords_dir=config.coords_dir,
        n_frames=config.n_frames,
        dataset_weights=config.dataset_weights,
        dt_ranges=config.dt_ranges,
        noise_P_mean=config.P_mean,
        noise_P_std=config.P_std,
        sigma_data=config.sigma_data,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size_per_gpu,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=BoltzKinemaCollator(),
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer: Adam with fixed LR + linear warmup
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # LR scheduler: linear warmup then constant
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss
    criterion = BoltzKinemaLoss(
        sigma_data=config.sigma_data,
        alpha_bond=config.alpha_bond,
        beta_flex=config.beta_flex,
    )

    # Prepare with accelerator
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # Resume from checkpoint
    global_step = 0
    if config.resume_from:
        accelerator.load_state(config.resume_from)
        global_step = int(config.resume_from.split('step_')[-1])

    # Training loop
    model.train()
    for epoch in range(config.max_epochs):
        for batch in dataloader:
            with accelerator.accumulate(model):
                output = model(batch)
                loss, loss_dict = criterion(output, batch, mode=config.training_mode)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1

                # Logging
                if global_step % config.log_every == 0:
                    accelerator.log(loss_dict, step=global_step)
                    accelerator.log({'lr': scheduler.get_last_lr()[0]}, step=global_step)

                # Checkpointing
                if global_step % config.save_every == 0:
                    accelerator.save_state(f"{config.output_dir}/step_{global_step}")

                if global_step >= config.max_steps:
                    break

        if global_step >= config.max_steps:
            break

    accelerator.end_training()
```

### Multi-Phase Training Workflow

The staged curriculum requires running `train.py` multiple times with different configs:

1. Phase 0: `accelerate launch scripts/train.py --config configs/train_phase0.yaml`
2. Phase 1: `accelerate launch scripts/train.py --config configs/train_equilibrium.yaml`
   - `resume_from` points to Phase 0 final checkpoint
   - Optimizer state is NOT resumed (new LR schedule)
3. Phase 1.5 (optional): `accelerate launch scripts/train.py --config configs/train_mutant_enrichment.yaml`
4. Phase 2: `accelerate launch scripts/train.py --config configs/train_unbinding.yaml`

`train.py` must support two resume modes:
- `resume_from` with `resume_optimizer: true` — full resume (same phase, interrupted training)
- `resume_from` with `resume_optimizer: false` — model-only resume (new phase, fresh optimizer)

### 4.3 Multi-Phase Training Curriculum

The training uses a staged curriculum to let the temporal attention module learn general protein dynamics before encountering multi-molecular complexity. The Boltz-2 trunk remains frozen throughout all phases.

**Rationale:** The temporal attention module starts from zero-initialized outputs (contributing nothing initially). Monomeric data provides the cleanest, most unambiguous learning signal for calibrating the learned exponential decay rates (`log_lambda`). Introducing complex multi-component systems too early risks learning incorrect temporal correlations before basic protein physics is established.

**Shared model configuration** (applies to all phases unless overridden):

```yaml
# === Model ===
token_s: 384
token_z: 128
atom_s: 128
atom_z: 16
atom_feature_dim: 128
atoms_per_window_queries: 32
atoms_per_window_keys: 128
sigma_data: 16.0
dim_fourier: 256
atom_encoder_depth: 3
atom_encoder_heads: 4
atom_temporal_heads: 4         # temporal heads at atom level
token_transformer_depth: 24
token_transformer_heads: 16
token_temporal_heads: 16       # temporal heads at token level
atom_decoder_depth: 3
atom_decoder_heads: 4
conditioning_transition_layers: 2
activation_checkpointing: true
freeze_diffusion_conditioning: false

# === Shared training params ===
max_epochs: 1000
batch_size_per_gpu: 1
gradient_accumulation_steps: 4   # effective batch_size = n_gpus * 1 * 4
grad_clip: 10.0
seed: 42
n_frames: 32
P_mean: -1.2
P_std: 1.5
forecast_prob: 0.5
num_workers: 4

# === Loss ===
alpha_bond: 1.0
beta_flex: 1.0
beta_abs: 1.0
beta_rel_g: 4.0
beta_rel_l: 4.0
mol_weights:
  protein: 1.0
  dna: 5.0
  rna: 5.0
  ligand: 10.0

# === Paths ===
boltz2_checkpoint: ~/.boltz/boltz2_conf.ckpt
manifest_path: data/processed/manifest.json
trunk_cache_dir: data/processed/trunk_embeddings/
coords_dir: data/processed/coords/
log_every: 50
save_every: 5000
```

#### Phase 0: Monomer Dynamics Pretraining

**File:** `configs/train_phase0.yaml`

```yaml
# Phase 0: Temporal attention warmup on monomeric proteins
# Initialize from: Boltz-2 pretrained weights (load_boltz2_weights)
# Duration: 100K-150K steps

training_mode: equilibrium
causal: false
lr: 1e-4
warmup_steps: 200
max_steps: 150000

dataset_weights:
  cath2: 4.0           # Broad fold coverage (primary)
  atlas: 2.0           # Established equilibrium dynamics
  octapeptides: 1.0    # Backbone flexibility (downweighted)
dt_ranges:
  cath2: [0.1, 50.0]
  atlas: [0.1, 10.0]
  octapeptides: [0.1, 100.0]

output_dir: checkpoints/phase0/
```

**Key details:**
- CATH2 dominates (weight 4.0) for maximum fold-class diversity
- ATLAS provides established equilibrium dynamics benchmarks
- Octapeptides downweighted (1.0) — lack structural context, risk biasing decay rates toward fast timescales
- Uses same model architecture, loss functions, and hyperparameters as Phase 1

#### Phase 1: Full Mixed Training

**File:** `configs/train_equilibrium.yaml` (updated)

```yaml
# Phase 1: Full mixed training with BioKinema + BioEmu datasets
# Initialize from: Phase 0 checkpoint
# Duration: 350K-400K steps

training_mode: equilibrium
causal: false
resume_from: checkpoints/phase0/step_150000
resume_optimizer: false            # Fresh optimizer for new phase
lr: 5e-5                          # Reduced from Phase 0
warmup_steps: 100
max_steps: 400000

dataset_weights:
  atlas: 1.0                      # BioKinema original (1:1:1 ratio preserved)
  misato: 1.0
  mdposit: 1.0
  cath2: 0.5                      # BioEmu data at reduced weight
  megasim_wt: 0.5                 # Wildtype (both force fields)
  megasim_mut: 0.5                # Curated mutant subset (~4,500 systems)
dt_ranges:
  atlas: [0.1, 10.0]
  misato: [0.08, 0.8]
  mdposit: [0.1, 100.0]
  cath2: [0.1, 50.0]
  megasim_wt: [0.1, 50.0]
  megasim_mut: [0.1, 50.0]

output_dir: checkpoints/phase1/
```

**Key details:**
- BioKinema original datasets maintain 1:1:1 ratio — this preserves the inter-molecular signal from MISATO (protein-ligand) and MDposit (multimers) that is critical for Ab-antigen applications
- BioEmu data enters at 0.5x weight to prevent monomeric data (~79 ms) from overwhelming multi-component data (~2 ms)
- LR reduced to 5e-5 (spatial weights are already well-adapted from Phase 0)
- Octapeptides dropped from Phase 1 (their value was in Phase 0 warmup)

#### Phase 1.5: MegaSim Mutant Enrichment (Optional)

**File:** `configs/train_mutant_enrichment.yaml`

```yaml
# Phase 1.5: Focus on mutation-dynamics relationship
# Initialize from: Phase 1 checkpoint
# Trigger: Only if Phase 1 eval shows weak mutation-effect sensitivity
# Duration: 50K-100K steps

training_mode: equilibrium
causal: false
resume_from: checkpoints/phase1/step_XXXXX
resume_optimizer: false
lr: 2e-5
warmup_steps: 50
max_steps: 100000

# Increased flexibility loss weight to emphasize distributional properties
beta_flex: 1.5

dataset_weights:
  megasim_mut_full: 2.0   # All 21,458 mutants
  atlas: 1.0
  misato: 1.0
dt_ranges:
  megasim_mut_full: [0.1, 50.0]
  atlas: [0.1, 10.0]
  misato: [0.08, 0.8]

output_dir: checkpoints/phase1.5/
```

**Key details:**
- Uses ALL 21,458 MegaSim mutants (requires full trunk precomputation: ~190 GB / ~180 GPU-hours)
- Increased `beta_flex` (1.5 vs 1.0) to emphasize ensemble distributional properties
- Only run if Phase 1 evaluation shows the model doesn't differentiate mutant vs. wildtype dynamics
- Directly relevant to the affinity optimization use case: teaches how mutations affect conformational dynamics

#### Phase 2: Unbinding Fine-Tuning

**File:** `configs/train_unbinding.yaml`

```yaml
training_mode: unbinding
causal: true                    # causal temporal attention for metadynamics
resume_from: checkpoints/phase1/step_XXXXX  # or phase1.5 if used
resume_optimizer: false
lr: 5e-5
warmup_steps: 100
max_steps: 200000
beta_center: 1.0
dt_ranges:
  dd13m: [0.01, 0.01]          # 10 ps fixed steps
dataset_weights:
  dd13m: 1.0

output_dir: checkpoints/phase2/
```

### 4.4 Memory Estimates

Per-sample memory at bf16 (approximate):

| Component | T=20, N=200, M=1500 | T=32, N=200, M=1500 | T=50, N=200, M=1500 |
|-----------|---------------------|---------------------|---------------------|
| Coordinates | 0.4 MB | 0.6 MB | 0.9 MB |
| Atom features | 7.5 MB | 12 MB | 19 MB |
| Token features | 6.0 MB | 9.6 MB | 15 MB |
| Temporal attention (T^2) | 0.3 MB | 0.8 MB | 1.9 MB |
| Gradients (2x params) | ~200 MB | ~200 MB | ~200 MB |
| Activations (with ckpt) | ~2 GB | ~3.5 GB | ~5.5 GB |
| **Total per sample** | **~3 GB** | **~4.5 GB** | **~7 GB** |

Recommended batch sizes:
- 48 GB GPU (A6000): batch_size=1, grad_accum=4, T=32
- 80 GB GPU (A100): batch_size=2, grad_accum=2, T=32 or batch_size=1, T=50

### 4.5 Launch Commands

```bash
# Phase 0: Monomer pretraining (100K-150K steps)
accelerate launch --num_processes=8 --mixed_precision=bf16 \
    scripts/train.py --config configs/train_phase0.yaml

# Phase 1: Full mixed training (350K-400K steps)
accelerate launch --num_processes=8 --mixed_precision=bf16 \
    scripts/train.py --config configs/train_equilibrium.yaml

# Phase 1.5 (optional): Mutant enrichment
accelerate launch --num_processes=8 --mixed_precision=bf16 \
    scripts/train.py --config configs/train_mutant_enrichment.yaml

# Phase 2: Unbinding fine-tuning
accelerate launch --num_processes=8 --mixed_precision=bf16 \
    scripts/train.py --config configs/train_unbinding.yaml
```

---

## 5. Inference Pipeline

### 5.1 EDM Sampler

**File:** `src/boltzkinema/inference/sampler.py`

```python
class EDMSampler:
    """
    EDM stochastic sampler for multi-frame denoising.

    Uses Karras et al. (2022) schedule with configurable parameters.

    Args:
        model: BoltzKinema model
        sigma_min: float = 0.0001
        sigma_max: float = 160.0
        sigma_data: float = 16.0
        rho: float = 7
        n_steps: int = 20
        noise_scale: float = 1.75
        step_scale: float = 1.5
    """

    def get_schedule(self, n_steps):
        """
        Karras et al. EDM noise schedule.

        `sigma_min`/`sigma_max` are absolute sigma values in coordinate units
        (do not multiply by `sigma_data` again).
        """
        inv_rho = 1.0 / self.rho
        steps = torch.arange(n_steps)
        sigmas = (
            self.sigma_max ** inv_rho
            + steps / (n_steps - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)
        ) ** self.rho
        sigmas = F.pad(sigmas, (0, 1), value=0.0)
        # Sanity checks for schedule endpoints
        assert torch.isclose(sigmas[0], torch.tensor(self.sigma_max), rtol=1e-4, atol=1e-6)
        assert torch.isclose(sigmas[-2], torch.tensor(self.sigma_min), rtol=1e-4, atol=1e-6)
        return sigmas

    @torch.no_grad()
    def sample(self, x_cond, t_cond, x_init, t_target, s_trunk, z_trunk,
               s_inputs, feats, mode='forecast'):
        """
        Denoise target frames given conditioning frames.

        Args:
            x_cond: (n_cond, M, 3) clean conditioning frame coordinates
            t_cond: (n_cond,) conditioning timestamps in ns
            x_init: (n_target, M, 3) initial noisy target coordinates
            t_target: (n_target,) target timestamps in ns
            s_trunk, z_trunk, s_inputs: precomputed trunk features
            feats: dict of atom-level features
            mode: 'forecast' or 'interpolate'
        Returns:
            x_denoised: (n_target, M, 3) denoised target coordinates
        """
        sigmas = self.get_schedule(self.n_steps)
        x_target = x_init

        for k in range(self.n_steps - 1):
            sigma_k = sigmas[k]
            sigma_next = sigmas[k + 1]

            # Assemble full sequence: conditioning (sigma=0) + targets (sigma=sigma_k)
            if mode == 'forecast':
                x_full = torch.cat([x_cond, x_target], dim=0).unsqueeze(0)
                sigma_full = torch.cat([
                    torch.zeros(len(x_cond)),
                    torch.full((len(x_target),), sigma_k.item())
                ]).unsqueeze(0)
                t_full = torch.cat([t_cond, t_target]).unsqueeze(0)
                cond_mask = torch.cat([
                    torch.ones(len(x_cond), dtype=torch.bool),
                    torch.zeros(len(x_target), dtype=torch.bool)
                ]).unsqueeze(0)
            else:  # interpolate
                x_full = torch.cat([x_cond[0:1], x_target, x_cond[-1:]]).unsqueeze(0)
                sigma_full = torch.cat([
                    torch.zeros(1),
                    torch.full((len(x_target),), sigma_k.item()),
                    torch.zeros(1),
                ]).unsqueeze(0)
                t_full = torch.cat([t_cond[0:1], t_target, t_cond[-1:]]).unsqueeze(0)
                cond_mask = torch.cat([
                    torch.ones(1, dtype=torch.bool),
                    torch.zeros(len(x_target), dtype=torch.bool),
                    torch.ones(1, dtype=torch.bool),
                ]).unsqueeze(0)

            batch = {
                'coords': x_full,  # not used directly (model uses x_noisy)
                'timestamps': t_full,
                'sigma': sigma_full,
                'conditioning_mask': cond_mask,
                's_trunk': s_trunk.unsqueeze(0),
                'z_trunk': z_trunk.unsqueeze(0),
                's_inputs': s_inputs.unsqueeze(0),
                'feats': feats,
            }

            output = self.model(batch)
            x_denoised_full = output['x_denoised'].squeeze(0)

            # Extract target predictions
            if mode == 'forecast':
                x_pred = x_denoised_full[len(x_cond):]
            else:
                x_pred = x_denoised_full[1:-1]

            # EDM update step
            d = (x_target - x_pred) / sigma_k
            x_target = x_pred + sigma_next * d

        return x_target
```

### 5.2 Hierarchical Trajectory Generation

**File:** `src/boltzkinema/inference/hierarchical.py`

```python
class HierarchicalGenerator:
    """
    Generate long trajectories using coarse forecasting + fine interpolation.

    Stage 1: Coarse auto-regressive forecasting at large Δt
    Stage 2: Fine interpolation between coarse anchors

    Args:
        model: BoltzKinema model
        sampler: EDMSampler
        coarse_dt_ns: float = 5.0 (coarse time step)
        fine_dt_ns: float = 0.1 (fine time step)
        generation_window: int = 40 (frames per AR block)
        history_window: int = 10 (conditioning frames from history)
    """

    @torch.no_grad()
    def generate(self, initial_structure, total_time_ns, s_trunk, z_trunk,
                 s_inputs, feats):
        """
        Generate trajectory from initial structure.

        Args:
            initial_structure: (M, 3) starting atom coordinates
            total_time_ns: float, total simulation time
            s_trunk, z_trunk, s_inputs: precomputed trunk embeddings
            feats: atom-level features
        Returns:
            trajectory: (N_frames, M, 3) full trajectory at fine resolution
            timestamps: (N_frames,) timestamps in ns
        """
        # === Stage 1: Coarse forecasting ===
        coarse_times = torch.arange(0, total_time_ns + self.coarse_dt_ns, self.coarse_dt_ns)
        coarse_traj = [initial_structure]

        idx = 0
        while idx < len(coarse_times) - 1:
            n_gen = min(self.generation_window, len(coarse_times) - 1 - idx)
            target_times = coarse_times[idx + 1: idx + 1 + n_gen]

            # History context
            hist_start = max(0, len(coarse_traj) - self.history_window)
            history = torch.stack(coarse_traj[hist_start:])
            history_times = coarse_times[hist_start: idx + 1]

            # Generate
            x_init = torch.randn(n_gen, *initial_structure.shape) * self.sampler.sigma_max
            x_gen = self.sampler.sample(
                x_cond=history, t_cond=history_times,
                x_init=x_init, t_target=target_times,
                s_trunk=s_trunk, z_trunk=z_trunk, s_inputs=s_inputs,
                feats=feats, mode='forecast',
            )

            for i in range(n_gen):
                coarse_traj.append(x_gen[i])
            idx += n_gen

        coarse_traj = torch.stack(coarse_traj)  # (N_coarse, M, 3)

        # === Stage 2: Fine interpolation ===
        n_interp = int(self.coarse_dt_ns / self.fine_dt_ns) - 1
        fine_traj = [coarse_traj[0]]
        fine_times = [coarse_times[0]]

        for i in range(len(coarse_traj) - 1):
            t0, t1 = coarse_times[i], coarse_times[i + 1]
            t_interp = torch.linspace(t0, t1, n_interp + 2)[1:-1]

            anchors = torch.stack([coarse_traj[i], coarse_traj[i + 1]])
            anchor_times = torch.tensor([t0, t1])

            x_init = torch.randn(n_interp, *initial_structure.shape) * self.sampler.sigma_max
            x_interp = self.sampler.sample(
                x_cond=anchors, t_cond=anchor_times,
                x_init=x_init, t_target=t_interp,
                s_trunk=s_trunk, z_trunk=z_trunk, s_inputs=s_inputs,
                feats=feats, mode='interpolate',
            )

            for j in range(n_interp):
                fine_traj.append(x_interp[j])
                fine_times.append(t_interp[j])
            fine_traj.append(coarse_traj[i + 1])
            fine_times.append(t1)

        return torch.stack(fine_traj), torch.tensor(fine_times)
```

### 5.3 Unbinding Generation

**File:** `src/boltzkinema/inference/unbinding.py`

```python
class UnbindingGenerator:
    """
    Auto-regressive unbinding trajectory generation.
    Uses the causal variant of the model (trained on DD-13M).

    Args:
        model: BoltzKinema model (causal=True variant)
        sampler: EDMSampler
        dt_ps: float = 10.0 (time step in picoseconds)
    """

    @torch.no_grad()
    def generate(self, complex_structure, total_ps, n_trajectories,
                 s_trunk, z_trunk, s_inputs, feats):
        """
        Generate multiple unbinding trajectories.

        Returns:
            trajectories: (n_traj, n_frames, M, 3)
        """
        n_steps = int(total_ps / self.dt_ps)
        all_trajs = []

        for traj_idx in range(n_trajectories):
            trajectory = [complex_structure]

            for step in range(n_steps):
                context = torch.stack(trajectory)
                context_times = torch.arange(len(context)) * self.dt_ps / 1000.0  # convert to ns
                target_time = torch.tensor([len(context) * self.dt_ps / 1000.0])

                x_init = torch.randn(1, *complex_structure.shape) * self.sampler.sigma_max

                x_next = self.sampler.sample(
                    x_cond=context, t_cond=context_times,
                    x_init=x_init, t_target=target_time,
                    s_trunk=s_trunk, z_trunk=z_trunk, s_inputs=s_inputs,
                    feats=feats, mode='forecast',
                )
                trajectory.append(x_next.squeeze(0))

            all_trajs.append(torch.stack(trajectory))

        return torch.stack(all_trajs)
```

---

## 6. Evaluation

### 6.1 Metrics

**File:** `src/boltzkinema/evaluation/metrics.py`

| Metric | Description | Used For |
|--------|-------------|----------|
| Pairwise RMSD | RMSD between all pairs of generated frames | ATLAS test |
| Ca RMSF correlation | Pearson correlation of per-residue RMSF (pred vs GT) | ATLAS test |
| W2-distance | Wasserstein-2 distance between distributions of pairwise distances | ATLAS test |
| Interaction Map Similarity (IMS) | Correlation of protein-ligand contact maps | MISATO test |
| Physical stability (post-minimization RMSD) | RMSD of generated structures after energy minimization | Long trajectory QC |
| MolProbity / Ramachandran | Structural quality metrics | Long trajectory QC |
| Unbinding precision/recall | Pathway similarity vs experimental unbinding paths | DD-13M test |

---

## 7. Implementation Order

1. **Finalize interfaces and manifests** (batch schema, split schema, unit conventions).
2. **Download and validate BioEmu datasets** (CATH2, MegaSim, Octapeptides from Zenodo).
3. **Implement preprocessing scripts** (solvent removal, alignment, missing-atom masks, unit conversion) for existing datasets.
4. **Implement preprocessing for BioEmu datasets** (`preprocess_cath.py`, `preprocess_megasim.py` with mutant subsampling, `preprocess_octapeptides.py`).
5. **Implement data-quality checks** (ligand valency, leakage guards, split validation).
6. **Cross-dataset deduplication** (CATH2 vs ATLAS sequence overlap check).
7. **Implement trunk precomputation and cache I/O** (chunked format, fp16 pair cache, I/O benchmark).
8. **Run trunk precomputation for BioEmu datasets** (~51 GPU-hours for recommended subset).
9. **Implement dataset + collator** with schema validation against the normative contract (with updated dataset_weights/dt_ranges for multi-phase).
10. **Implement `TemporalAttentionWithDecay` + `PerFrameEDM`** and unit tests.
11. **Implement spatial-temporal encoder/transformer/decoder** and top-level `BoltzKinema`.
12. **Implement Boltz-2 weight loading** and run single-frame equivalence validation.
13. **Implement masked loss suite** (`L_struct`, `L_bond`, `L_smooth_lddt`, `L_flex`, `L_center`).
14. **Implement accelerate training loop** with checkpoint/resume/logging (with phase-aware checkpoint resume).
15. **Run smoke training** (Phase 0 on small ATLAS+CATH2 subset) and memory profiling.
16. **Implement inference stack** (EDM sampler, hierarchical generation, unbinding AR).
17. **Implement evaluation metrics + OOD benchmarks**.
18. **Run Phase 0 → Phase 1 → evaluate.**
19. **Decide on Phase 1.5** based on mutation-effect sensitivity evaluation.

## 8. Verification Plan

1. **Unit and contract tests**
   - `TemporalAttentionWithDecay`: output shape `(B*N, T, C)`, zero-init output, causal masking correctness.
   - `PerFrameEDM`: `c_skip(0)=1`, `c_out(0)=0`, finite `loss_weight` for masked frames.
   - Batch schema validator: all required keys present; explicit failure on missing required keys.
   - Trunk-cache loader: handles required fields exactly (`s_inputs`, `s_trunk`, `z_trunk`) and rejects stale schemas.

2. **Data integrity tests**
   - Unit roundtrip: preprocessing outputs coordinates in Angstrom and times in ns.
   - Split integrity: no system-level leakage across train/val/test/OOD.
   - OOD filter check: OOD proteins satisfy sequence identity threshold.
   - Missing-residue mask check: unresolved atoms are excluded from supervised losses.

3. **Model equivalence and objective tests**
   - Single-frame equivalence: with `T=1` and temporal outputs zero-initialized, match Boltz-2 diffusion output.
   - Loss masking: conditioning frames and unobserved atoms contribute zero gradient for supervised terms.
   - EDM schedule sanity: sampled schedule starts at `sigma_max`, ends at `sigma_min`, no accidental rescaling.

4. **Training and inference smoke tests**
   - Train on 10 ATLAS systems for 1,000 steps; verify stable optimization and non-zero temporal gradients.
   - Generate a 100 ns trajectory for a small protein; verify bond sanity/no severe clashes.
   - Verify hierarchical generation returns continuous timestamps and no duplicated anchor frames.

5. **Data integrity tests (BioEmu datasets)**
   - CATH2 preprocessing: verify domain count (~1,100), residue length range (50-200), trajectory duration (~1 us).
   - MegaSim mutant subsampling: verify ~4,500 systems selected, deltaG distribution covers extremes.
   - Octapeptides preprocessing: verify all 5 replicas per system, sequence length = 8.
   - Cross-dataset deduplication: log and resolve CATH2/ATLAS overlaps.

6. **Training validation checkpoints (BioEmu integration)**
   - After Phase 0: verify temporal attention decay rates (`log_lambda`) have diverged from initialization (different heads at different timescales). Log mean and std of `exp(log_lambda)` across heads.
   - After Phase 0: evaluate on held-out ATLAS proteins — RMSF Pearson correlation should exceed 0.7.
   - After Phase 1: **critical** — verify model does NOT generate partially unfolded conformations for well-folded test proteins. If observed, MegaSim folding/unfolding data may need filtering to retain only folded-state frames.
   - After Phase 1: compare predicted dynamics changes between MegaSim wildtype and mutant systems against known deltaG shifts.

## 9. Decision Log (Compared to OVERVIEW.md)

| Area | Decision in WORKPLAN | Reason |
|------|----------------------|--------|
| Training framework | HuggingFace `accelerate` | Explicit project constraint; avoids PyTorch Lightning dependency |
| Backbone constants | Use Boltz-2 values from local codebase (`token_s=384`, pairformer blocks=64, etc.) | Prevent mismatch with AF3/Protenix defaults |
| Unit conventions | Canonical runtime units are Angstrom + ns | Avoid mixed nm/ps vs Angstrom/ns errors in augmentation/loss |
| EDM schedule semantics | `sigma_min`/`sigma_max` treated as absolute sigma values | Prevent accidental `sigma_data` double-scaling |
| Trunk artifact schema | Cache `s_inputs`, `s_trunk`, `z_trunk`; do not cache `rel_pos_enc` | Reduce storage and simplify cache contract |
| Loss masking | All supervised terms use target-frame mask; unresolved atoms masked out | Align with noise-as-masking training semantics |
| CATH1 exclusion | Exclude CATH1 adaptive sampling data | Non-equilibrium transition-state conformations conflict with noise-as-masking paradigm; only 50 systems |
| MegaSim mutant subsampling | Use ~4,500 of 21,458 mutants initially | Reduces trunk precomputation from ~180 to ~38 GPU-hours while retaining most informative mutants (extreme deltaG + positional diversity) |
| Force field heterogeneity | Ignore (no FF conditioning token) | BioKinema paper's approach of mixing without FF conditioning works; diversity may help generalization; implementation complexity not justified initially |
| Training curriculum | 4-phase staged curriculum | Temporal attention trains from scratch; monomeric pretraining provides clean signal for decay rate calibration before multi-molecular complexity |
| BioEmu data weight in Phase 1 | 0.5x relative to BioKinema datasets | Prevents ~79 ms monomeric data from drowning out ~2 ms inter-molecular data critical for Ab-antigen use case |
| Octapeptides Phase 1 exclusion | Drop from Phase 1, only use in Phase 0 | Value is in temporal attention warmup; free peptide dynamics not useful once model has learned from full proteins and complexes |

## 10. Workplan Acceptance Checklist

- [ ] Unit conventions are explicit and consistent across preprocessing/training/inference.
- [ ] Required runtime batch schema is fully specified with producer/consumer ownership.
- [ ] Trunk-cache artifact schema is versioned and storage-feasible at target scale.
- [ ] Dataset split and OOD leakage checks are specified and testable.
- [ ] Loss masking rules are explicit for conditioning frames and unresolved atoms.
- [ ] EDM schedule equations and parameter semantics are unambiguous.
- [ ] Implementation order is dependency-correct (data before training execution).
- [ ] Verification plan includes unit, integration, data-integrity, and smoke tests.
