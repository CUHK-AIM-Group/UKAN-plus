#!/bin/bash
# Baseline U-KAN on ACDC (single-scale grid_size=5)
# For comparison with U-KAN++ in Table III

PYTHON="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/.venv/bin/python"
DATA_DIR="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/U-KAN/data/acdc"
OUTPUT_DIR="outputs"

SEEDS=(2981 6142 1187)
GPU=5

for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON train_multiclass.py \
    --name "acdc_ukan_seed${seed}" \
    --dataset acdc \
    --data_dir "$DATA_DIR" \
    --num_classes 4 \
    --img_size 224 \
    --epochs 150 \
    --batch_size 24 \
    --dataseed $seed \
    --arch UKAN \
    --eval_interval 10 \
    --output_dir $OUTPUT_DIR &
  GPU=$(( (GPU + 1) % 8 ))
done

wait
echo "=== Baseline U-KAN ACDC complete ==="
