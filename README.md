# U-KAN++

Official PyTorch codebase for **U-KAN++**, the journal extension of U-KAN (AAAI 2025). U-KAN++ is a KAN-based medical image analysis framework that unifies:

- multigrid spline routing within each KAN block
- spline-driven interpretable decoder gating
- joint segmentation and image generation

This repository contains the core training, evaluation, and generation code, while excluding paper sources, local datasets, checkpoints, and experiment logs.

## Highlights

- `FR-KAN`: Frequency-Routed KAN blocks with parallel B-spline branches at multiple grid resolutions, fused by an attention router
- `STAG`: Spline Transformation Attention Gate for decoder modulation and mechanism-level interpretability
- `USG`: unified segmentation-generation training of a shared FR-KAN backbone with segmentation and diffusion branches
- support for both 2D and 3D medical image segmentation experiments

## Repository Structure

- `Seg_UKAN_plus/`
  Core segmentation code, including 2D/3D models, STAG, USG, evaluation, and experiment launch scripts.
- `Diffusion_UKAN_plus/`
  Standalone diffusion models and generation-evaluation utilities.
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

ISIC 2018 uses the official train/validation/test split:

```bash
cd Seg_UKAN_plus
python prepare_isic2018.py --isic_root /path/to/ISIC2018 --out_dir /path/to/data_root/isic2018
```

## Quick Start

Train 2D segmentation:

```bash
cd Seg_UKAN_plus
python train.py --arch UKANPP --dataset busi --input_w 256 --input_h 256 --name busi_frkan --data_dir /path/to/data_root
```

Train and evaluate 3D segmentation:

```bash
cd Seg_UKAN_plus
python train_3d.py --dataset synapse --data_dir /path/to/synapse --use_stag --iters_per_epoch 250
python eval_3d.py --exp_dir outputs_3d/synapse_ukanpp3d_seed2981
```

Run unified segmentation-generation training and sample from the generation path:

```bash
cd Seg_UKAN_plus
python usg_train.py --name usg_joint_dynamic_busi_seed2981 --dataset busi --data_dir /path/to/data_root --mode joint --use_stag
python usg_sample.py --exp_dir outputs/usg_joint_dynamic_busi_seed2981
```

Run STAG faithfulness evaluation:

```bash
cd Seg_UKAN_plus
python sam_eval.py --checkpoint outputs/usg_joint_dynamic_busi_seed2981 --output_dir results/stag_busi
```

Train and sample the standalone diffusion model:

```bash
cd Diffusion_UKAN_plus
python Main.py --model UKan_PP_Hybrid --exp_nme UKan_PP_busi --dataset busi --epoch 5000
python Main_Test.py --model UKan_PP_Hybrid --dataset busi --save_root ./Output/ --exp_nme UKan_PP_busi --test_load_weight ckpt_5000_.pt
```

The experiment launch scripts (`run_*.sh`) are in `Seg_UKAN_plus/`, for example:

```bash
PYTHON=python DATA_ROOT=/path/to/data_root ./Seg_UKAN_plus/run_table1.sh
```

## Notes

- Dataset paths, checkpoints, and generated outputs should be configured locally.
- Third-party baselines (TransUNet, U-NeXt, Rolling-UNet, U-Mamba, UNETR++) are not included; `Seg_UKAN_plus/baselines.py` loads them from local clones of their official repositories.

## Citation

If you use this codebase in research, please cite the corresponding U-KAN / U-KAN++ paper version used in your work. Citation details can be updated here when the final public manuscript metadata is fixed.
