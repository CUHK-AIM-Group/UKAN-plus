#!/bin/bash
# Table II: U-KAN++ 3D on Synapse
# 3 seeds, 1000 epochs each, SGD (lr 1e-2, momentum 0.99) with polynomial decay
# The 18 training cases are stacked from the TransUNet slices into volumes, so an
# epoch is fixed at 250 iterations (nnU-Net convention) instead of one pass.
# Evaluation: sliding-window inference on whole test volumes (eval_3d.py);
# HD95 is in voxels unless SPACING_JSON gives per-case spacing in mm.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../data}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}/synapse}"
OUTPUT_DIR="outputs_3d"
cd "$SCRIPT_DIR"

SEEDS=(2981 6142 1187)
GPU=0

for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON train_3d.py \
    --name "synapse_ukanpp3d_seed${seed}" \
    --dataset synapse \
    --data_dir "$DATA_DIR" \
    --output_dir $OUTPUT_DIR \
    --epochs 1000 \
    --batch_size 2 \
    --dataseed $seed \
    --embed_dims 64,128,256 \
    --grid_sizes 3,5,10 \
    --fusion_type softmax \
    --use_stag --stag_stages dec --stag_alpha 0.5 \
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
  $PYTHON eval_3d.py --exp_dir "$OUTPUT_DIR/synapse_ukanpp3d_seed${seed}" \
    ${SPACING_JSON:+--spacing_json "$SPACING_JSON"}
done
echo "=== Table II Synapse experiments complete ==="
