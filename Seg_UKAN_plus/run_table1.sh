#!/bin/bash
# Table I: U-KAN++ 2D Segmentation on BUSI/GlaS/CVC/ISIC2018
# 4 datasets × 3 seeds = 12 runs, distributed across 8 GPUs

PYTHON="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/.venv/bin/python"
DATA_DIR="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/U-KAN/data"
OUTPUT_DIR="outputs"

SEEDS=(2981 6142 1187)
GPU=0

# --- BUSI (256×256) ---
for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
    --name "busi_ukanpp_seed${seed}" \
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
    --name "glas_ukanpp_seed${seed}" \
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
    --name "cvc_ukanpp_seed${seed}" \
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
    --name "isic2018_ukanpp_seed${seed}" \
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
echo "=== Table I experiments complete ==="
