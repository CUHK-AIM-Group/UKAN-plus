#!/bin/bash
# Table XI: Diffusion Generation Quality (FID/IS) on BUSI/GlaS/CVC
# U-KAN++ (UKan_PP_Hybrid) on 3 datasets

PYTHON="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/.venv/bin/python"
SAVE_ROOT="./Output/"
GPU=0

# BUSI: 5000 epochs (as per original training script)
CUDA_VISIBLE_DEVICES=$GPU $PYTHON Main.py \
  --model UKan_PP_Hybrid \
  --exp_nme UKan_PP_busi \
  --batch_size 32 \
  --channel 64 \
  --dataset busi \
  --epoch 5000 \
  --save_root ${SAVE_ROOT} &
GPU=$((GPU + 1))

# GlaS: 1000 epochs
CUDA_VISIBLE_DEVICES=$GPU $PYTHON Main.py \
  --model UKan_PP_Hybrid \
  --exp_nme UKan_PP_glas \
  --batch_size 32 \
  --channel 64 \
  --dataset glas \
  --epoch 1000 \
  --save_root ${SAVE_ROOT} &
GPU=$((GPU + 1))

# CVC: 1000 epochs
CUDA_VISIBLE_DEVICES=$GPU $PYTHON Main.py \
  --model UKan_PP_Hybrid \
  --exp_nme UKan_PP_cvc \
  --batch_size 32 \
  --channel 64 \
  --dataset cvc \
  --epoch 1000 \
  --save_root ${SAVE_ROOT} &
GPU=$((GPU + 1))

wait

echo "=== Diffusion training complete. Computing FID/IS ==="

# Compute FID for each dataset
GPU=0
for ds in busi glas cvc; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m pytorch_fid \
    "data/${ds}/images_64/" \
    "${SAVE_ROOT}/UKan_PP_${ds}/Gens" \
    > "${SAVE_ROOT}/UKan_PP_${ds}/FID.txt" 2>&1 &
  GPU=$((GPU + 1))
done
wait

echo "=== FID computation complete ==="
echo "Results:"
for ds in busi glas cvc; do
  echo "--- ${ds} ---"
  cat "${SAVE_ROOT}/UKan_PP_${ds}/FID.txt"
done
