#!/bin/bash
# Baseline U-KAN on BraTS 2021 (single-scale grid_size=5)
# Using UKANPP3D with grid_sizes=5 to simulate single-scale U-KAN
# For comparison with U-KAN++ in Table IV

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../data}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}/brats2021}"
OUTPUT_DIR="outputs_3d"
cd "$SCRIPT_DIR"

SEEDS=(2981 6142 1187)
GPU=5

for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON train_3d.py \
    --name "brats_ukan_seed${seed}" \
    --dataset brats2021 \
    --data_dir "$DATA_DIR" \
    --output_dir $OUTPUT_DIR \
    --epochs 300 \
    --batch_size 2 \
    --dataseed $seed \
    --embed_dims 64,128,256 \
    --grid_sizes 5 \
    --fusion_type softmax \
    --crop_size 96,96,96 \
    --lr 1e-4 \
    --kan_lr 1e-2 \
    --num_workers 8 &
  GPU=$(( (GPU + 1) % 8 ))
done

wait
echo "=== Baseline U-KAN BraTS complete ==="
