#!/bin/bash
# =====================================================================
# USG joint training for GlaS, CVC, ISIC2018 — 3 seeds each
# + Synapse/ACDC longer training — 3 seeds each
# Total: 9 (2D USG) + 6 (3D) = 15 jobs, round-robin on 8 GPUs
# =====================================================================

PYTHON="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/.venv/bin/python"
DATA_DIR="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/U-KAN/data"
OUTPUT_DIR="outputs"
TAG="batch3_$(date +%Y%m%d_%H%M%S)"

SEEDS=(2981 6142 1187)
GPU=0

# =====================================================================
# Part 1: 2D USG joint training (glas, cvc, isic2018)
# =====================================================================
echo "=== Part 1: 2D USG joint training ==="
for ds in glas cvc isic2018; do
  for seed in "${SEEDS[@]}"; do
    echo "  Launching USG $ds seed=$seed on GPU=$GPU"
    CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON usg_train.py \
      --name "${TAG}_usg_${ds}_s${seed}" \
      --dataset $ds \
      --data_dir "$DATA_DIR" \
      --epochs 400 \
      --batch_size 8 \
      --input_h 256 --input_w 256 \
      --dataseed $seed \
      --output_dir $OUTPUT_DIR \
      --grid_sizes 3,5,10 \
      --fusion_type softmax \
      --mode joint \
      --lambda_type dynamic \
      --lambda_0 1.0 \
      --gamma 0.5 \
      > nohup_${TAG}_usg_${ds}_s${seed}.log 2>&1 &
    GPU=$(( (GPU + 1) % 8 ))
  done
done

# =====================================================================
# Part 2: 3D — Synapse with more epochs (300 -> see if helps)
# Current: 150 epochs, Dice=70.00. Try 300 epochs + lower LR.
# =====================================================================
echo "=== Part 2: 3D Synapse (300 epochs) ==="
for seed in "${SEEDS[@]}"; do
  echo "  Launching Synapse seed=$seed on GPU=$GPU"
  CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON train_multiclass.py \
    --name "${TAG}_synapse_ukanpp_300ep_s${seed}" \
    --dataset synapse \
    --data_dir "$DATA_DIR/synapse" \
    --num_classes 9 \
    --img_size 224 \
    --epochs 300 \
    --batch_size 24 \
    --dataseed $seed \
    --arch UKANPP \
    --grid_sizes 3,5,10 \
    --fusion_type softmax \
    --eval_interval 10 \
    --output_dir $OUTPUT_DIR \
    > nohup_${TAG}_synapse_s${seed}.log 2>&1 &
  GPU=$(( (GPU + 1) % 8 ))
done

# =====================================================================
# Part 3: 3D — ACDC with more epochs (300)
# Current: 150 epochs, Dice=82.59. Try 300 epochs.
# =====================================================================
echo "=== Part 3: 3D ACDC (300 epochs) ==="
for seed in "${SEEDS[@]}"; do
  echo "  Launching ACDC seed=$seed on GPU=$GPU"
  CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON train_multiclass.py \
    --name "${TAG}_acdc_ukanpp_300ep_s${seed}" \
    --dataset acdc \
    --data_dir "$DATA_DIR/acdc" \
    --num_classes 4 \
    --img_size 224 \
    --epochs 300 \
    --batch_size 24 \
    --dataseed $seed \
    --arch UKANPP \
    --grid_sizes 3,5,10 \
    --fusion_type softmax \
    --eval_interval 10 \
    --output_dir $OUTPUT_DIR \
    > nohup_${TAG}_acdc_s${seed}.log 2>&1 &
  GPU=$(( (GPU + 1) % 8 ))
done

echo ""
echo "=== All jobs launched ==="
echo "  TAG=$TAG"
echo "  Part 1: 9 jobs — USG {glas, cvc, isic2018} x 3 seeds"
echo "  Part 2: 3 jobs — Synapse 300ep x 3 seeds"
echo "  Part 3: 3 jobs — ACDC 300ep x 3 seeds"
echo "  Total: 15 jobs round-robin on 8 GPUs"
echo ""
echo "Monitor: tail -f outputs/${TAG}_*/log.csv"
