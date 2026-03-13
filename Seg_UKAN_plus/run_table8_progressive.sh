#!/bin/bash
# Table VIII: Progressive Ablation on BUSI
# Row 1: U-KAN baseline (grid_sizes=5, original) — from AAAI paper, or re-run
# Row 2: + MS-KAN (grid_sizes=3,5,10)
# Row 3: + MS-KAN + SAM (post-hoc, no training needed)
# Row 4: + MS-KAN + USG (needs usg_train.py, separate script)

PYTHON="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/.venv/bin/python"
DATA_DIR="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/U-KAN/data"
OUTPUT_DIR="outputs"

SEEDS=(2981 6142 1187)
GPU=0

# Row 1: U-KAN baseline (original arch, single grid_size=5)
for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
    --name "progressive_ukan_baseline_seed${seed}" \
    --dataset busi \
    --data_dir "$DATA_DIR" \
    --arch UKAN \
    --epochs 400 \
    --batch_size 8 \
    --input_h 256 --input_w 256 \
    --dataseed $seed \
    --output_dir $OUTPUT_DIR &
  GPU=$(( (GPU + 1) % 8 ))
done

# Row 2: + MS-KAN (U-KAN++ with multi-scale)
for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
    --name "progressive_ukan_mskan_seed${seed}" \
    --dataset busi \
    --data_dir "$DATA_DIR" \
    --arch UKANPP \
    --epochs 400 \
    --batch_size 8 \
    --input_h 256 --input_w 256 \
    --dataseed $seed \
    --output_dir $OUTPUT_DIR \
    --grid_sizes 3,5,10 \
    --fusion_type softmax &
  GPU=$(( (GPU + 1) % 8 ))
done

wait
echo "=== Table VIII progressive ablation (rows 1-2) complete ==="
echo "Row 3 (SAM) is post-hoc analysis, no training needed."
echo "Row 4 (USG) requires usg_train.py — run separately."
