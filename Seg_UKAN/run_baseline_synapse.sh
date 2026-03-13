#!/bin/bash
# Baseline U-KAN on Synapse (single-scale grid_size=5)
# For comparison with U-KAN++ in Table II

PYTHON="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/.venv/bin/python"
DATA_DIR="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/U-KAN/data/synapse"
OUTPUT_DIR="outputs"

SEEDS=(2981 6142 1187)
GPU=3

for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON train_multiclass.py \
    --name "synapse_ukan_seed${seed}" \
    --dataset synapse \
    --data_dir "$DATA_DIR" \
    --num_classes 9 \
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
echo "=== Baseline U-KAN Synapse complete ==="
