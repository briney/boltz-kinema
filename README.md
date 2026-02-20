# Kinematic

Kinematic is a diffusion-based generative model for predicting continuous-time, all-atom biomolecular trajectories. It extends [Boltz-2](https://github.com/jwohlwend/boltz) — an AlphaFold 3-like structure prediction model — with a Spatial-Temporal Diffusion Module that jointly models spatial relationships within molecular frames and temporal dependencies across frames.

The frozen Boltz-2 trunk provides structural representations (single and pair embeddings), while Kinematic's trainable temporal diffusion module learns to generate physically plausible dynamics: equilibrium fluctuations, conformational changes, and ligand unbinding pathways.

**Paper:** Feng et al., 2026 — [bioRxiv 10.64898/2026.02.15.705956](https://doi.org/10.64898/2026.02.15.705956)

## Key Features

- **Temporal attention with exponential decay** — physically grounded attention bias derived from Langevin dynamics, supporting continuous timestamps from sub-nanosecond to microsecond scales
- **Noise-as-masking training** — clean frames (sigma=0) condition the model while noisy frames are denoised, unifying forecasting and interpolation in a single architecture
- **Hierarchical inference** — coarse-to-fine generation enables long trajectories (1+ microseconds) without unbounded error accumulation
- **Zero-initialization** — temporal output projections start as zeros, so the model begins as single-frame Boltz-2 and smoothly learns dynamics during training

## Installation

Kinematic requires Python 3.10+ and PyTorch 2.1+.

### 1. Install Boltz-2

Boltz-2 must be installed separately from source:

```bash
git clone https://github.com/jwohlwend/boltz.git
cd boltz
pip install -e .
```

### 2. Install Kinematic

```bash
git clone https://github.com/briney/kinematic.git
cd kinematic
pip install -e ".[dev]"
```

The `[dev]` extra includes `pytest` and `ruff` for development.

## Quick Start

### Generate a trajectory

```bash
python scripts/generate.py \
    --config configs/inference.yaml \
    --input structure.npz \
    --checkpoint checkpoints/phase1/best
```

Override any config parameter from the command line:

```bash
python scripts/generate.py \
    --config configs/inference.yaml \
    --input structure.npz \
    mode=unbinding \
    total_frames=200 \
    dt_ns=0.5
```

### Hierarchical generation (for long trajectories)

Two-stage coarse-to-fine sampling generates long trajectories efficiently — coarse forecasting at large time steps followed by fine interpolation:

```bash
python scripts/generate.py \
    --config configs/inference.yaml \
    --input structure.npz \
    hierarchical=true \
    coarse_dt_ns=10.0 \
    fine_dt_ns=2.5
```

## Training

Training proceeds in three phases with increasing data complexity. All phases use the [Accelerate](https://huggingface.co/docs/accelerate) launcher for single- or multi-GPU training.

### Download training data

```bash
# Download all datasets
kinematic download-training-data --datasets all --output-dir data/raw

# Or select specific datasets
kinematic download-training-data --datasets atlas,cath2,octapeptides --output-dir data/raw
```

Available datasets:

| Dataset | Systems | Size | Description |
|---------|---------|------|-------------|
| ATLAS | ~1,500 chains | ~30 GB | Protein monomers, 3x100 ns trajectories |
| CATH2 | ~1,100 domains | ~28 GB | CATH protein domains, ~1 microsecond each |
| Octapeptides | ~1,100 peptides | ~511 MB | 8-residue peptides, 5x1 microsecond each |
| MISATO | ~16,972 complexes | ~190 GB | Protein-ligand complexes, 10 ns each |
| DynaRepo | ~930 systems | ~200 GB | Diverse MD (proteins, multimers, nucleic acids) |
| MegaSim | 271 WT + 21,458 mutants | ~10 GB | Wildtype and mutant protein simulations |
| DD-13M | 565 complexes, 26,612 trajectories | ~100 GB | Metadynamics ligand dissociation |

### Precompute trunk embeddings

Before training, precompute the frozen Boltz-2 trunk representations (run once):

```bash
python scripts/precompute_trunk.py \
    --manifest data/processed/manifest.json \
    --checkpoint ~/.boltz/boltz2_conf.ckpt \
    --output-dir data/processed/trunk_embeddings \
    --device cuda
```

### Run training

```bash
# Phase 0: Temporal warmup on monomeric proteins (loads Boltz-2 weights)
accelerate launch -m kinematic train --base-config base --config train_phase0

# Phase 1: Full mixed equilibrium training (resumes from Phase 0)
accelerate launch -m kinematic train --config train_equilibrium

# Phase 2: Unbinding fine-tune with causal temporal attention (resumes from Phase 1)
accelerate launch -m kinematic train --config train_unbinding
```

Config values can be overridden from the command line using OmegaConf syntax:

```bash
accelerate launch -m kinematic train --config train_phase0 lr=2e-4 max_steps=50000
```

For multi-GPU training, pass an Accelerate config:

```bash
accelerate launch --config_file configs/accelerate_multi_gpu.yaml \
    -m kinematic train --config train_phase0
```

## Architecture

```
Kinematic
├── Boltz-2 Trunk (frozen, precomputed)
│   ├── Input Embedder        sequence + MSA → single (s) and pair (z) representations
│   ├── MSA Module            multiple sequence alignment processing
│   └── Pairformer            triangle updates + attention (4 recycling cycles)
│
└── Spatial-Temporal Diffusion Module (trainable)
    ├── Atom Encoder           3 blocks: spatial atom attention + temporal attention
    ├── Token Transformer      24 blocks: spatial token attention + temporal attention
    └── Atom Decoder           3 blocks: spatial atom attention + temporal attention
```

Each block in the diffusion module applies:
1. **Spatial attention** within each frame (from Boltz-2)
2. **Temporal attention** across all frames for each atom/token (new)
3. **Feed-forward transition**

### Generation Modes

- **Equilibrium** — bidirectional temporal attention for conformational sampling (forecasting and interpolation tasks)
- **Unbinding** — causal temporal attention for ligand dissociation pathway generation
- **Hierarchical** — two-stage coarse-to-fine generation for long trajectories

## Project Structure

```
src/kinematic/
├── cli/            CLI commands (train, download-training-data)
├── model/          Model modules (temporal attention, EDM, spatial-temporal blocks)
├── data/           Dataset, collator, trunk cache, preprocessing
├── training/       Trainer, losses, learning rate scheduler
├── inference/      EDM sampler, hierarchical and unbinding generators
└── evaluation/     Trajectory evaluation metrics

configs/            YAML configuration files for training and inference
scripts/            Preprocessing, trunk precomputation, and inference scripts
tests/              Test suite
```

## Development

### Run tests

```bash
python -m pytest                                       # all tests
python -m pytest tests/test_temporal_attention.py       # single file
python -m pytest tests/test_shapes.py -k "test_name"   # single test
```

Some tests require the `boltz` package and will skip automatically if it is not installed.

### Lint

```bash
ruff check src scripts tests
ruff check --fix src scripts tests   # auto-fix
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@article{feng2026kinematic,
    title={Kinematic: Boltz-2 Based All-Atom Biomolecular Dynamics},
    author={Feng, ...},
    journal={bioRxiv},
    doi={10.64898/2026.02.15.705956},
    year={2026}
}
```
