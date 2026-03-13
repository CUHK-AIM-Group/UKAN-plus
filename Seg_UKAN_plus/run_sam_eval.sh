#!/bin/bash
# Tables IX, X + Figures 1, 2: SAM Evaluation
# Requires trained U-KAN++ checkpoints from Table I

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
OUTPUT_DIR="outputs"
cd "$SCRIPT_DIR"

# Run SAM evaluation on BUSI (seed 2981)
CUDA_VISIBLE_DEVICES=0 $PYTHON sam_eval.py \
  --checkpoint ${OUTPUT_DIR}/busi_ukanpp_seed2981 \
  --output_dir results/sam_busi \
  --num_samples 50 --num_vis 8 &

# Run SAM evaluation on ISIC2018 (seed 2981)
CUDA_VISIBLE_DEVICES=1 $PYTHON sam_eval.py \
  --checkpoint ${OUTPUT_DIR}/isic2018_ukanpp_seed2981 \
  --output_dir results/sam_isic \
  --num_samples 50 --num_vis 8 &

wait
echo "=== SAM evaluation complete ==="
