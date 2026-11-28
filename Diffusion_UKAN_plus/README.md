# Diffusion U-KAN++

Standalone diffusion models of the U-KAN++ codebase.

This directory contains the training and sampling code for the standalone diffusion models used in the U-KAN / U-KAN++ line of work, the `images_64` preparation scripts, and the Inception Score code. The generation path of the joint U-KAN++ model is in `Seg_UKAN_plus/`. Local datasets, checkpoints, and generated outputs are intentionally excluded from the repository.

## Setup

```bash
conda create -n ukanpp python=3.10
conda activate ukanpp
pip install -r requirements.txt
```

## Directory Layout

- `Diffusion/`: model definitions, diffusion process, and training utilities
- `training_scripts/`: lightweight dataset-specific launch scripts
- `tools/`: simple preprocessing helpers
- `Main.py`: training entry point
- `Main_Test.py`: sampling / test entry point
- `run_table11.sh`: launch script for standalone BUSI / GlaS / CVC generation runs

## Data

By default, the code expects local data folders under:

```text
Diffusion_UKAN_plus/data/
  busi/
  glas/
  cvc/
```

These folders are ignored by Git and should be prepared locally.

## Quick Start

Train:

```bash
python Main.py --model UKan_PP_Hybrid --dataset busi --epoch 5000
```

Sample:

```bash
python Main_Test.py
```

Portable launcher example:

```bash
PYTHON=python ./run_table11.sh
```

## Notes

- The code keeps the original model naming used by the generation branch, such as `UKan_PP_Hybrid`.
- For larger-scale runs, set dataset paths, save roots, and CUDA devices explicitly in your shell environment.
- Generated samples and released checkpoints are intentionally not versioned in this cleaned repository.
