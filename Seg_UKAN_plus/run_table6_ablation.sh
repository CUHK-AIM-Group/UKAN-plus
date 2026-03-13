#!/bin/bash
# Table VI: MS-KAN Ablation on BUSI
# 7 configs × 3 seeds = 21 runs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../data}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}}"
OUTPUT_DIR="outputs"
cd "$SCRIPT_DIR"

SEEDS=(2981 6142 1187)

# Ablation configs: grid_sizes|fusion_type
# Single-scale baselines
CONFIGS=(
  "3|softmax"             # G=3 only (single branch, fusion irrelevant)
  "5|softmax"             # G=5 only (original U-KAN equivalent)
  "10|softmax"            # G=10 only
  "3,5|softmax"           # G={3,5} softmax
  "3,5,10|concat"         # G={3,5,10} concat
  "3,5,10|sum"            # G={3,5,10} mean
  "3,5,10|softmax"        # G={3,5,10} softmax (default U-KAN++)
)

# Batch 1: first 8 runs
GPU=0
BATCH=0
for config in "${CONFIGS[@]}"; do
  IFS='|' read -r grids fusion <<< "$config"
  grid_tag="${grids//,/_}"
  for seed in "${SEEDS[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
      --name "ablation_g${grid_tag}_${fusion}_seed${seed}" \
      --dataset busi \
      --data_dir "$DATA_DIR" \
      --arch UKANPP \
      --epochs 400 \
      --batch_size 8 \
      --input_h 256 --input_w 256 \
      --dataseed $seed \
      --output_dir $OUTPUT_DIR \
      --grid_sizes "$grids" \
      --fusion_type "$fusion" &

    BATCH=$(( BATCH + 1 ))
    GPU=$(( (GPU + 1) % 8 ))

    # Wait every 8 runs to avoid overloading GPUs
    if [ $((BATCH % 8)) -eq 0 ]; then
      wait
    fi
  done
done

wait
echo "=== Table VI ablation experiments complete ==="
