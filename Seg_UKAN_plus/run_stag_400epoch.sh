#!/bin/bash
# =====================================================================
# B1(b): FR-KAN + STAG (dec2, alpha=0.05), BUSI, 400 epoch, 3 seeds
# B2:    STAG hyperparameter ablation (alpha & stage placement)
# =====================================================================
# GPU assignment: 8 GPUs available (0-7), one job per GPU

PYTHON="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/.venv/bin/python"
DATA_DIR="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/U-KAN/data"
OUTPUT_DIR="outputs"
TAG="stag400ep_$(date +%Y%m%d_%H%M%S)"

SEEDS=(2981 6142 1187)
GPU=0

# =====================================================================
# B1(b): Main STAG config — FR-KAN + STAG (dec2, alpha=0.05), 3 seeds
# This fills Progressive Ablation Table Row 3
# =====================================================================
echo "=== B1(b): FR-KAN + STAG (dec2, alpha=0.05), 400 epochs, 3 seeds ==="
for seed in "${SEEDS[@]}"; do
  echo "  Launching seed=$seed on GPU=$GPU"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
    --name "${TAG}_frkan_stag_dec2_a005_s${seed}" \
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
    --stag_stages dec2 \
    --stag_alpha 0.05 \
    --stag_norm minmax &
  GPU=$(( (GPU + 1) % 8 ))
done

# =====================================================================
# B2: STAG alpha ablation — dec2 stage, vary alpha
# alpha = 0.01, 0.1, 0.5  (0.05 already covered above)
# Use seed=2981 (single seed for ablation efficiency)
# =====================================================================
echo "=== B2a: Alpha ablation (dec2, seed=2981) ==="
for alpha in 0.01 0.1 0.5; do
  alpha_tag=${alpha//.}
  echo "  Launching alpha=$alpha on GPU=$GPU"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
    --name "${TAG}_ablation_alpha_dec2_a${alpha_tag}_s2981" \
    --dataset busi \
    --data_dir "$DATA_DIR" \
    --arch UKANPP \
    --epochs 400 \
    --batch_size 8 \
    --input_h 256 --input_w 256 \
    --dataseed 2981 \
    --output_dir $OUTPUT_DIR \
    --grid_sizes 3,5,10 \
    --fusion_type softmax \
    --ms_stages all \
    --use_stag \
    --stag_stages dec2 \
    --stag_alpha $alpha \
    --stag_norm minmax &
  GPU=$(( (GPU + 1) % 8 ))
done

# =====================================================================
# B2: STAG stage ablation — alpha=0.05, vary stage placement
# dec1, dec12  (dec2 already covered in B1(b))
# Use seed=2981
# =====================================================================
echo "=== B2b: Stage ablation (alpha=0.05, seed=2981) ==="
for stage in dec1 dec12; do
  echo "  Launching stag_stages=$stage on GPU=$GPU"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
    --name "${TAG}_ablation_stage_${stage}_a005_s2981" \
    --dataset busi \
    --data_dir "$DATA_DIR" \
    --arch UKANPP \
    --epochs 400 \
    --batch_size 8 \
    --input_h 256 --input_w 256 \
    --dataseed 2981 \
    --output_dir $OUTPUT_DIR \
    --grid_sizes 3,5,10 \
    --fusion_type softmax \
    --ms_stages all \
    --use_stag \
    --stag_stages $stage \
    --stag_alpha 0.05 \
    --stag_norm minmax &
  GPU=$(( (GPU + 1) % 8 ))
done

echo ""
echo "=== All 8 jobs launched (GPUs 0-7) ==="
echo "  B1(b): 3 jobs  — FR-KAN+STAG dec2 a=0.05, seeds {2981,6142,1187}"
echo "  B2a:   3 jobs  — alpha ablation {0.01, 0.1, 0.5}, dec2, seed=2981"
echo "  B2b:   2 jobs  — stage ablation {dec1, dec12}, a=0.05, seed=2981"
echo ""
echo "Monitor with: tail -f outputs/${TAG}_*/log.csv"

wait
echo "=== All experiments finished ==="
