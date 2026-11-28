#!/bin/bash
# Baseline U-KAN on ACDC (single-scale grid_size=5)
# Using UKANPP3D with grid_sizes=5 to simulate single-scale U-KAN (3D record for Table VIII)
# Same 3D schedule as Table III: 3 seeds, 1000 epochs each,
# SGD (lr 1e-2, momentum 0.99) with polynomial decay, 96x96x16 crops

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../data}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}/acdc}"
OUTPUT_DIR="outputs_3d"
cd "$SCRIPT_DIR"

SEEDS=(2981 6142 1187)
GPU=5

for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON train_3d.py \
    --name "acdc_ukan_seed${seed}" \
    --dataset acdc \
    --data_dir "$DATA_DIR" \
    --output_dir $OUTPUT_DIR \
    --epochs 1000 \
    --batch_size 2 \
    --dataseed $seed \
    --embed_dims 64,128,256 \
    --grid_sizes 5 \
    --fusion_type softmax \
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
  $PYTHON eval_3d.py --exp_dir "$OUTPUT_DIR/acdc_ukan_seed${seed}"
done
echo "=== Baseline U-KAN ACDC complete ==="
