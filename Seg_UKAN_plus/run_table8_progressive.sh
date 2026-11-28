#!/bin/bash
# Fig. 7(a) and Table V: Progressive Ablation on BUSI
# Row 1: U-KAN baseline (grid_sizes=5, original) — from AAAI paper, or re-run
# Row 2: + MS-KAN (grid_sizes=3,5,10)
# Row 3: + MS-KAN + STAG (trained; see run_stag_400epoch.sh)
# Row 4: + MS-KAN + STAG + USG (joint; see run_table7_usg.sh)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../data}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}}"
OUTPUT_DIR="outputs"
cd "$SCRIPT_DIR"

SEEDS=(2981 6142 1187)
GPU=0

# Row 1: U-KAN baseline (original arch, single grid_size=5)
for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
    --name "progressive_ukan_baseline_seed${seed}" \
    --dataset busi \
    --data_dir "$DATA_DIR" \
    --arch UKAN \
    --epochs 400 \
    --batch_size 8 \
    --input_h 256 --input_w 256 \
    --dataseed $seed \
    --output_dir $OUTPUT_DIR &
  GPU=$(( (GPU + 1) % 8 ))
done

# Row 2: + MS-KAN (U-KAN++ with multi-scale)
for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
    --name "progressive_ukan_mskan_seed${seed}" \
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

wait
echo "=== Progressive ablation (rows 1-2) complete ==="
echo "Row 3 (STAG): run_stag_400epoch.sh"
echo "Row 4 (STAG + USG): run_table7_usg.sh"
