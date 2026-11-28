#!/bin/bash
# FR-KAN only (segmentation-only, no STAG) on BUSI/GlaS/CVC/ISIC2018
# 4 datasets × 3 seeds = 12 runs, distributed across 8 GPUs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../data}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}}"
OUTPUT_DIR="outputs"
cd "$SCRIPT_DIR"

SEEDS=(2981 6142 1187)
GPU=0

# --- BUSI (256×256) ---
for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
    --name "busi_frkan_seed${seed}" \
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

# --- GlaS (512×512) ---
for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
    --name "glas_frkan_seed${seed}" \
    --dataset glas \
    --data_dir "$DATA_DIR" \
    --arch UKANPP \
    --epochs 400 \
    --batch_size 8 \
    --input_h 512 --input_w 512 \
    --input_list 128,160,256 \
    --dataseed $seed \
    --output_dir $OUTPUT_DIR \
    --grid_sizes 3,5,10 \
    --fusion_type softmax &
  GPU=$(( (GPU + 1) % 8 ))
done

# --- CVC (256×256) ---
for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
    --name "cvc_frkan_seed${seed}" \
    --dataset cvc \
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

# Wait for first batch (9 runs on 8 GPUs, one GPU runs 2 sequentially)
wait

# --- ISIC 2018 (256×256) ---
GPU=0
for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
    --name "isic2018_frkan_seed${seed}" \
    --dataset isic2018 \
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
echo "=== FR-KAN-only experiments complete ==="
