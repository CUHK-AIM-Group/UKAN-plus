#!/bin/bash
# Table III: U-KAN++ on ACDC
# 3 seeds, 150 epochs each

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../data}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}/acdc}"
OUTPUT_DIR="outputs"
cd "$SCRIPT_DIR"

SEEDS=(2981 6142 1187)
GPU=0

for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON train_multiclass.py \
    --name "acdc_ukanpp_seed${seed}" \
    --dataset acdc \
    --data_dir "$DATA_DIR" \
    --num_classes 4 \
    --img_size 224 \
    --epochs 150 \
    --batch_size 24 \
    --dataseed $seed \
    --arch UKANPP \
    --grid_sizes 3,5,10 \
    --fusion_type softmax \
    --eval_interval 10 \
    --output_dir $OUTPUT_DIR &
  GPU=$(( (GPU + 1) % 8 ))
done

wait
echo "=== Table III ACDC experiments complete ==="
