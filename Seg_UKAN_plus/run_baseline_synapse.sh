#!/bin/bash
# Baseline U-KAN on Synapse (single-scale grid_size=5)
# Using UKANPP3D with grid_sizes=5 to simulate single-scale U-KAN (3D record for Table VIII)
# Same 3D schedule as Table II: 3 seeds, 1000 epochs of 250 iterations,
# SGD (lr 1e-2, momentum 0.99) with polynomial decay, 96^3 crops

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../data}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}/synapse}"
OUTPUT_DIR="outputs_3d"
cd "$SCRIPT_DIR"

SEEDS=(2981 6142 1187)
GPU=3

for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON train_3d.py \
    --name "synapse_ukan_seed${seed}" \
    --dataset synapse \
    --data_dir "$DATA_DIR" \
    --output_dir $OUTPUT_DIR \
    --epochs 1000 \
    --batch_size 2 \
    --dataseed $seed \
    --embed_dims 64,128,256 \
    --grid_sizes 5 \
    --fusion_type softmax \
    --crop_size 96,96,96 \
    --iters_per_epoch 250 \
    --val_interval 10 \
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
  $PYTHON eval_3d.py --exp_dir "$OUTPUT_DIR/synapse_ukan_seed${seed}" \
    ${SPACING_JSON:+--spacing_json "$SPACING_JSON"}
done
echo "=== Baseline U-KAN Synapse complete ==="
