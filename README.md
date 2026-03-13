# U-KAN++

Minimal cleaned codebase for the U-KAN++ project.

This repository keeps the model and experiment code only. It intentionally omits:

- paper sources
- figure assets
- local datasets
- training outputs, logs, and checkpoints

## Repository Layout

- `Seg_UKAN/`: segmentation, 3D extension, STAG, USG, and evaluation scripts
- `Diffusion_UKAN/`: diffusion-based generation code and training entry points
- `LICENSE`: upstream MIT license file carried from the segmentation codebase

## Main Entry Points

Segmentation:

- `python Seg_UKAN/train.py`
- `python Seg_UKAN/train_3d.py`
- `python Seg_UKAN/usg_train.py`
- `python Seg_UKAN/sam_eval.py`

Generation:

- `python Diffusion_UKAN/Main.py`
- `python Diffusion_UKAN/Main_Test.py`

## Environment

Two environment specs are retained from the working codebase:

- `Seg_UKAN/requirements.txt`
- `Seg_UKAN/environment.yml`
- `Diffusion_UKAN/requirements.txt`

## Notes

- This is a compact research code snapshot for U-KAN++, not a full artifact release.
- Dataset folders and checkpoints should be created locally and are ignored by `.gitignore`.
- The `run_table*.sh` scripts are kept as lightweight experiment launchers for the main reported settings.
