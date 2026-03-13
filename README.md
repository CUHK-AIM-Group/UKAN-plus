# U-KAN++

Official PyTorch codebase for **U-KAN++**, a KAN-based medical image analysis framework that unifies:

- frequency-routed multi-scale representation learning
- spline-driven interpretable gating
- joint segmentation and image generation

This repository is organized as a compact research code release. It contains the core training, evaluation, and generation code, while excluding paper sources, local datasets, checkpoints, and experiment logs.

## Highlights

- `FR-KAN`: multi-scale frequency-routed KAN blocks for stronger hierarchical feature modeling
- `STAG`: Spline Transformation Attention Gate for decoder modulation and mechanism-level interpretability
- `USG`: unified segmentation-generation training with a shared backbone
- support for both 2D and 3D medical image segmentation experiments
- diffusion-based image generation branch for the generative component of U-KAN++

## Repository Structure

- `Seg_UKAN_plus/`
  Core segmentation code, including 2D/3D models, STAG, USG, evaluation, and experiment launch scripts.
- `Diffusion_UKAN_plus/`
  Diffusion-based generation code and supporting utilities for the generative branch.
- `LICENSE`
  Upstream license file retained with the released code.

## Environment Setup

Segmentation branch:

```bash
conda create -n ukanpp python=3.10
conda activate ukanpp
pip install -r Seg_UKAN_plus/requirements.txt
```

Alternative segmentation environment:

```bash
conda env create -f Seg_UKAN_plus/environment.yml
conda activate ukan
```

Generation branch:

```bash
pip install -r Diffusion_UKAN_plus/requirements.txt
```

## Data Preparation

This repository does not bundle datasets. Please prepare the datasets locally and place them under your own working directories before training or evaluation.

Expected data/output folders are intentionally excluded from version control, including:

- segmentation inputs, logs, and outputs
- diffusion training data
- generated samples and released checkpoints

## Quick Start

Train 2D segmentation:

```bash
cd Seg_UKAN_plus
python train.py --arch UKAN --dataset busi --input_w 256 --input_h 256 --name busi_ukanpp --data_dir ./inputs
```

Train 3D segmentation:

```bash
cd Seg_UKAN_plus
python train_3d.py
```

Run unified segmentation-generation training:

```bash
cd Seg_UKAN_plus
python usg_train.py
```

Run SAM / STAG faithfulness evaluation:

```bash
cd Seg_UKAN_plus
python sam_eval.py
```

Run diffusion training:

```bash
cd Diffusion_UKAN_plus
python Main.py
```

Run diffusion sampling / test:

```bash
cd Diffusion_UKAN_plus
python Main_Test.py
```

## Included Experiment Scripts

The repository keeps the lightweight launch scripts used for the main study settings, including:

- `run_table1.sh`, `run_table2_synapse.sh`, `run_table3_acdc.sh`, `run_table4_brats.sh`
- `run_table6_ablation.sh`, `run_table7_usg.sh`, `run_table8_progressive.sh`
- `run_sam_eval.sh`
- `Diffusion_UKAN_plus/run_table11.sh`

## Notes

- This is a cleaned private code snapshot for U-KAN++ rather than a full public artifact release.
- Dataset paths, checkpoints, and generated outputs should be configured locally.
- The code is kept close to the active research workspace, with paper assets removed for a cleaner repository.

## Citation

If you use this codebase in research, please cite the corresponding U-KAN / U-KAN++ paper version used in your work. Citation details can be updated here when the final public manuscript metadata is fixed.
