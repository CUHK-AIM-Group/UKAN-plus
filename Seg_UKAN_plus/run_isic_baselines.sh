#!/bin/bash
# Run baseline models on ISIC 2018 using segmentation_models_pytorch
# Only 1 seed (2981) since these are non-main-table baselines

PYTHON="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/.venv/bin/python"
DATA_DIR="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/U-KAN/data"
WORK_DIR="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/U-KAN/Seg_UKAN"
LOG_DIR="$WORK_DIR/logs"
COMMON="--dataset isic2018 --data_dir $DATA_DIR --epochs 400 --batch_size 8 --input_h 256 --input_w 256 --output_dir outputs"

GPU=0

# Models to run (using train_baseline.py)
for MODEL in unet attunet unetpp unext; do
  for SEED in 2981 6142 1187; do
    nohup bash -c "
      cd $WORK_DIR
      CUDA_VISIBLE_DEVICES=$GPU $PYTHON train_baseline.py \
        --name baseline_${MODEL}_isic_seed${SEED} \
        --arch $MODEL \
        --dataseed $SEED \
        $COMMON
      echo 'DONE: ${MODEL}_isic_seed${SEED}'
    " > $LOG_DIR/baseline_${MODEL}_isic_seed${SEED}.log 2>&1 &
    echo "${MODEL} seed${SEED} -> GPU $GPU, PID $!"
    GPU=$(( (GPU + 1) % 8 ))
  done
done

echo "All baseline jobs launched."
