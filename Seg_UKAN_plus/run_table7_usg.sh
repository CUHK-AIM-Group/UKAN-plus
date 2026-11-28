#!/bin/bash
# USG runs on BUSI, GlaS and CVC
# Gen-only: BUSI, GlaS, CVC x 3 seeds = 9 runs
# Joint (full U-KAN++: FR-KAN + STAG + USG, dynamic lambda): BUSI x 3 seeds = 3 runs
# Joint runs on GlaS, CVC and ISIC 2018 are in run_batch3_usg_3d.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../data}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}}"
OUTPUT_DIR="outputs"
cd "$SCRIPT_DIR"

SEEDS=(2981 6142 1187)
GPU=0

# Gen-only
for ds in busi glas cvc; do
  for seed in "${SEEDS[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON usg_train.py \
      --name "usg_genonly_${ds}_seed${seed}" \
      --dataset $ds \
      --data_dir "$DATA_DIR" \
      --mode gen_only \
      --grid_sizes 3,5,10 \
      --dataseed $seed \
      --epochs 400 \
      --batch_size 8 \
      --output_dir $OUTPUT_DIR &
    GPU=$(( (GPU + 1) % 8 ))
  done
done
wait

# Joint (dynamic lambda) with STAG — full U-KAN++
GPU=0
for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON usg_train.py \
    --name "usg_joint_dynamic_busi_seed${seed}" \
    --dataset busi \
    --data_dir "$DATA_DIR" \
    --mode joint \
    --lambda_type dynamic --lambda_0 1.0 --gamma 0.5 \
    --grid_sizes 3,5,10 \
    --use_stag --stag_stages dec --stag_alpha 0.5 \
    --dataseed $seed \
    --epochs 400 \
    --batch_size 8 \
    --output_dir $OUTPUT_DIR &
  GPU=$(( (GPU + 1) % 8 ))
done

wait
echo "=== USG runs complete ==="
