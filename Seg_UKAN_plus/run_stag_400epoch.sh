#!/bin/bash
# =====================================================================
# FR-KAN + STAG, segmentation-only, 400 epochs, 3 seeds
# Part 1: BUSI, default STAG (dec1+dec2, alpha=0.5) and placement
#         runs (dec1 only, dec2 only)
# Part 2: GlaS, CVC-ClinicDB, ISIC 2018 with the default STAG
# Part 3: inference-time alpha sweep on the Part 1 default models
#         (weights fixed; alpha=0 disables the gate only at inference)
# =====================================================================
# GPU assignment: 8 GPUs available (0-7), round-robin

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../data}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}}"
OUTPUT_DIR="outputs"
TAG="stag400ep_$(date +%Y%m%d_%H%M%S)"
cd "$SCRIPT_DIR"

SEEDS=(2981 6142 1187)
GPU=0

# =====================================================================
# Part 1: BUSI. stag_stages "dec" applies STAG at dec1 and dec2.
# =====================================================================
echo "=== Part 1: BUSI FR-KAN + STAG (dec, dec1, dec2; alpha=0.5) ==="
for stage in dec dec1 dec2; do
  for seed in "${SEEDS[@]}"; do
    echo "  Launching stag_stages=$stage seed=$seed on GPU=$GPU"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
      --name "${TAG}_busi_frkan_stag_${stage}_a05_s${seed}" \
      --dataset busi \
      --data_dir "$DATA_DIR" \
      --arch UKANPP \
      --epochs 400 \
      --batch_size 8 \
      --input_h 256 --input_w 256 \
      --dataseed $seed \
      --output_dir $OUTPUT_DIR \
      --grid_sizes 3,5,10 \
      --fusion_type softmax \
      --ms_stages all \
      --use_stag \
      --stag_stages $stage \
      --stag_alpha 0.5 \
      --stag_norm minmax &
    GPU=$(( (GPU + 1) % 8 ))
  done
done
wait

# =====================================================================
# Part 2: GlaS (512x512), CVC-ClinicDB and ISIC 2018 (256x256)
# =====================================================================
echo "=== Part 2: FR-KAN + STAG on GlaS, CVC, ISIC 2018 (dec, alpha=0.5) ==="
GPU=0
for ds in glas cvc isic2018; do
  if [ "$ds" = "glas" ]; then SIZE=512; else SIZE=256; fi
  for seed in "${SEEDS[@]}"; do
    echo "  Launching $ds seed=$seed on GPU=$GPU"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
      --name "${TAG}_${ds}_frkan_stag_dec_a05_s${seed}" \
      --dataset $ds \
      --data_dir "$DATA_DIR" \
      --arch UKANPP \
      --epochs 400 \
      --batch_size 8 \
      --input_h $SIZE --input_w $SIZE \
      --dataseed $seed \
      --output_dir $OUTPUT_DIR \
      --grid_sizes 3,5,10 \
      --fusion_type softmax \
      --ms_stages all \
      --use_stag \
      --stag_stages dec \
      --stag_alpha 0.5 \
      --stag_norm minmax &
    GPU=$(( (GPU + 1) % 8 ))
  done
done
wait

# =====================================================================
# Part 3: inference-time alpha sweep (BUSI, default models from Part 1)
# =====================================================================
echo "=== Part 3: inference-time alpha sweep on BUSI ==="
for seed in "${SEEDS[@]}"; do
  RUN="${TAG}_busi_frkan_stag_dec_a05_s${seed}"
  for alpha in 0 0.01 0.05 0.1 0.5; do
    echo "  Evaluating $RUN with alpha=$alpha"
    CUDA_VISIBLE_DEVICES=0 $PYTHON val.py \
      --name "$RUN" \
      --output_dir $OUTPUT_DIR \
      --stag_alpha $alpha | tee "${OUTPUT_DIR}/${RUN}/val_stag_alpha_${alpha}.txt"
  done
done

echo ""
echo "=== All experiments finished ==="
echo "  Part 1: 9 runs  — BUSI, stag_stages {dec, dec1, dec2}, alpha=0.5, seeds {2981,6142,1187}"
echo "  Part 2: 9 runs  — {glas, cvc, isic2018}, stag_stages dec, alpha=0.5"
echo "  Part 3: alpha sweep {0, 0.01, 0.05, 0.1, 0.5} -> outputs/${TAG}_busi_frkan_stag_dec_a05_s*/val_stag_alpha_*.txt"
