#!/bin/bash
# Table VI: MS-KAN Ablation on BUSI — using GPU 6,7
# 7 configs × 3 seeds = 21 runs, all parallel on 2 GPUs

PYTHON="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/.venv/bin/python"
DATA_DIR="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/U-KAN/data"
OUTPUT_DIR="outputs"

SEEDS=(2981 6142 1187)

CONFIGS=(
  "3|softmax"
  "5|softmax"
  "10|softmax"
  "3,5|softmax"
  "3,5,10|concat"
  "3,5,10|sum"
  "3,5,10|softmax"
)

GPUS=(6 7)
IDX=0

for config in "${CONFIGS[@]}"; do
  IFS='|' read -r grids fusion <<< "$config"
  grid_tag="${grids//,/_}"
  for seed in "${SEEDS[@]}"; do
    GPU=${GPUS[$((IDX % 2))]}

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
      --name "ablation_g${grid_tag}_${fusion}_seed${seed}" \
      --dataset busi \
      --data_dir "$DATA_DIR" \
      --arch UKANPP \
      --epochs 400 \
      --batch_size 8 \
      --input_h 256 --input_w 256 \
      --dataseed $seed \
      --output_dir $OUTPUT_DIR \
      --grid_sizes "$grids" \
      --fusion_type "$fusion" &

    IDX=$(( IDX + 1 ))
  done
done

echo "Launched $IDX ablation runs on GPU 6,7"
wait
echo "=== Table VI ablation experiments complete ==="
