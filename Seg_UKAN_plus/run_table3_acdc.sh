#!/bin/bash
# Table III: U-KAN++ 3D on ACDC
# 3 seeds, 1000 epochs each, SGD (lr 1e-2, momentum 0.99) with polynomial decay

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../data}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}/acdc}"
OUTPUT_DIR="outputs_3d"
cd "$SCRIPT_DIR"

SEEDS=(2981 6142 1187)
GPU=0

for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON train_3d.py \
    --name "acdc_ukanpp3d_seed${seed}" \
    --dataset acdc \
    --data_dir "$DATA_DIR" \
    --output_dir $OUTPUT_DIR \
    --epochs 1000 \
    --batch_size 2 \
    --dataseed $seed \
    --embed_dims 64,128,256 \
    --grid_sizes 3,5,10 \
    --fusion_type softmax \
    --use_stag --stag_stages dec --stag_alpha 0.5 \
    --crop_size 16,96,96 \
    --optimizer SGD \
    --momentum 0.99 \
    --lr 1e-2 \
    --kan_lr 1e-2 \
    --scheduler poly \
    --num_workers 8 &
  GPU=$(( (GPU + 1) % 8 ))
done

wait
for seed in "${SEEDS[@]}"; do
  $PYTHON eval_3d.py --exp_dir "$OUTPUT_DIR/acdc_ukanpp3d_seed${seed}"
done
echo "=== Table III ACDC experiments complete ==="
