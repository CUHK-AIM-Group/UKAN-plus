#!/bin/bash
# Table I (ISIC 2018 column): baselines trained in this codebase, 3 seeds each
# U-Net, Att-UNet, U-Net++ (segmentation_models_pytorch) and U-NeXt, Rolling-UNet, U-Mamba
# (official implementations; set UNEXT_DIR, ROLLING_UNET_DIR, UMAMBA_DIR, see baselines.py)
# with train_baseline.py, and U-KAN with train.py. TransUNet on ISIC 2018 is in run_baselines.sh.
# All models are trained from scratch (--no_pretrained). Prepare the data with
# prepare_isic2018.py for the official split; the test metrics are written to <run>/test.csv.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../data}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}}"
WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}}"
LOG_DIR="$WORK_DIR/logs"
COMMON="--dataset isic2018 --data_dir $DATA_DIR --epochs 400 --batch_size 8 --input_h 256 --input_w 256 --output_dir outputs"

GPU=0
mkdir -p "$LOG_DIR"

for MODEL in unet attunet unetpp unext rollingunet umamba; do
  for SEED in 2981 6142 1187; do
    nohup bash -c "
      cd $WORK_DIR
      CUDA_VISIBLE_DEVICES=$GPU $PYTHON train_baseline.py \
        --name baseline_${MODEL}_isic_seed${SEED} \
        --arch $MODEL \
        --dataseed $SEED \
        --no_pretrained \
        $COMMON
      echo 'DONE: ${MODEL}_isic_seed${SEED}'
    " > $LOG_DIR/baseline_${MODEL}_isic_seed${SEED}.log 2>&1 &
    echo "${MODEL} seed${SEED} -> GPU $GPU, PID $!"
    GPU=$(( (GPU + 1) % 8 ))
  done
done

for SEED in 2981 6142 1187; do
  nohup bash -c "
    cd $WORK_DIR
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
      --name isic2018_ukan_seed${SEED} \
      --arch UKAN \
      --dataseed $SEED \
      $COMMON
    echo 'DONE: ukan_isic_seed${SEED}'
  " > $LOG_DIR/isic2018_ukan_seed${SEED}.log 2>&1 &
  echo "ukan seed${SEED} -> GPU $GPU, PID $!"
  GPU=$(( (GPU + 1) % 8 ))
done

echo "All baseline jobs launched."
