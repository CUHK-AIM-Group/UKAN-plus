#!/bin/bash
# =====================================================================
# USG joint training (full U-KAN++: FR-KAN + STAG + USG) for GlaS, CVC,
# ISIC2018 — 3 seeds each
# Total: 9 jobs, round-robin on 8 GPUs
# =====================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../data}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}}"
OUTPUT_DIR="outputs"
cd "$SCRIPT_DIR"

SEEDS=(2981 6142 1187)
GPU=0

# =====================================================================
# Part 1: 2D USG joint training (glas 512x512; cvc, isic2018 256x256)
# =====================================================================
echo "=== Part 1: 2D USG joint training ==="
for ds in glas cvc isic2018; do
  if [ "$ds" = "glas" ]; then SIZE=512; else SIZE=256; fi
  for seed in "${SEEDS[@]}"; do
    echo "  Launching USG $ds seed=$seed on GPU=$GPU"
    CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON usg_train.py \
      --name "usg_joint_dynamic_${ds}_seed${seed}" \
      --dataset $ds \
      --data_dir "$DATA_DIR" \
      --epochs 400 \
      --batch_size 8 \
      --input_h $SIZE --input_w $SIZE \
      --dataseed $seed \
      --output_dir $OUTPUT_DIR \
      --grid_sizes 3,5,10 \
      --fusion_type softmax \
      --use_stag --stag_stages dec --stag_alpha 0.5 \
      --mode joint \
      --lambda_type dynamic \
      --lambda_0 1.0 \
      --gamma 0.5 \
      > nohup_usg_joint_dynamic_${ds}_seed${seed}.log 2>&1 &
    GPU=$(( (GPU + 1) % 8 ))
  done
done

echo ""
echo "=== All jobs launched ==="
echo "  Part 1: 9 jobs — USG {glas, cvc, isic2018} x 3 seeds"
echo "  Total: 9 jobs round-robin on 8 GPUs"
echo ""
echo "Monitor: tail -f outputs/usg_joint_dynamic_*/log.csv"
