#!/bin/bash
# Table IV: U-KAN++ 3D on BraTS 2021
# 3 seeds, 300 epochs each

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../data}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}/brats2021}"
OUTPUT_DIR="outputs_3d"
cd "$SCRIPT_DIR"

SEEDS=(2981 6142 1187)
GPU=0

for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON train_3d.py \
    --name "brats_ukanpp3d_seed${seed}" \
    --dataset brats2021 \
    --data_dir "$DATA_DIR" \
    --output_dir $OUTPUT_DIR \
    --epochs 300 \
    --batch_size 2 \
    --dataseed $seed \
    --embed_dims 64,128,256 \
    --grid_sizes 3,5,10 \
    --fusion_type softmax \
    --crop_size 96,96,96 \
    --lr 1e-4 \
    --kan_lr 1e-2 \
    --num_workers 8 &
  GPU=$(( (GPU + 1) % 8 ))
done

wait
echo "=== Table IV BraTS experiments complete ==="
