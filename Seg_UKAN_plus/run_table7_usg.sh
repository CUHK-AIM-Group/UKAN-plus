#!/bin/bash
# Table VII: USG Ablation on BUSI
# 3 configs × 3 seeds = 9 runs (seg_only reuses Table I BUSI results)

PYTHON="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/.venv/bin/python"
DATA_DIR="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/U-KAN/data"
OUTPUT_DIR="outputs"

SEEDS=(2981 6142 1187)
GPU=0

# Gen-only
for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON usg_train.py \
    --name "usg_genonly_seed${seed}" \
    --dataset busi \
    --data_dir "$DATA_DIR" \
    --mode gen_only \
    --grid_sizes 3,5,10 \
    --dataseed $seed \
    --epochs 400 \
    --batch_size 8 \
    --output_dir $OUTPUT_DIR &
  GPU=$(( (GPU + 1) % 8 ))
done

# Joint (fixed lambda=0.5)
for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON usg_train.py \
    --name "usg_joint_fixed_seed${seed}" \
    --dataset busi \
    --data_dir "$DATA_DIR" \
    --mode joint \
    --lambda_type fixed --lambda_val 0.5 \
    --grid_sizes 3,5,10 \
    --dataseed $seed \
    --epochs 400 \
    --batch_size 8 \
    --output_dir $OUTPUT_DIR &
  GPU=$(( (GPU + 1) % 8 ))
done

# Joint (dynamic lambda) — paper default
for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON usg_train.py \
    --name "usg_joint_dynamic_seed${seed}" \
    --dataset busi \
    --data_dir "$DATA_DIR" \
    --mode joint \
    --lambda_type dynamic --lambda_0 1.0 --gamma 0.5 \
    --grid_sizes 3,5,10 \
    --dataseed $seed \
    --epochs 400 \
    --batch_size 8 \
    --output_dir $OUTPUT_DIR &
  GPU=$(( (GPU + 1) % 8 ))
done

wait
echo "=== Table VII USG ablation complete ==="
