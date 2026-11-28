#!/bin/bash
# Figs. 10 and 11: STAG vs Grad-CAM, Grad-CAM++ and Score-CAM
# Deletion/insertion faithfulness (Dice) and localization (IoU@0.5, Pointing Game)
# for the complete Joint U-KAN++ model (seed 2981)
# Fig. 10: GlaS, CVC, ISIC 2018; Fig. 11: BUSI curves
# Requires the joint checkpoints from run_table7_usg.sh (BUSI) and run_batch3_usg_3d.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
OUTPUT_DIR="outputs"
cd "$SCRIPT_DIR"

GPU=0
for ds in busi glas cvc isic2018; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON sam_eval.py \
    --checkpoint ${OUTPUT_DIR}/usg_joint_dynamic_${ds}_seed2981 \
    --output_dir results/stag_${ds} \
    --num_samples 50 --num_vis 8 &
  GPU=$(( (GPU + 1) % 8 ))
done

wait
echo "=== STAG evaluation complete ==="
echo "Fig. 10: results/stag_{glas,cvc,isic2018}/summary.json"
echo "Fig. 11: results/stag_busi/perturbation_curves.pdf"
