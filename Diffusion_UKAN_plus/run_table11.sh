#!/bin/bash
# Standalone diffusion model (UKan_PP_Hybrid) on BUSI/GlaS/CVC: FID and IS
# DDPM with T = 1000 at 64x64
# Table VI and Fig. 8 use the generation path of the joint model: Seg_UKAN_plus/run_usg_fid.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
SAVE_ROOT="./Output/"
cd "$SCRIPT_DIR"
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

echo "=== FID computation complete. Computing IS ==="

# Compute IS for each dataset
GPU=0
for ds in busi glas cvc; do
  GENS="${SCRIPT_DIR}/${SAVE_ROOT}/UKan_PP_${ds}/Gens"
  (cd inception-score-pytorch && CUDA_VISIBLE_DEVICES=$GPU $PYTHON inception_score.py --data-root "$GENS") \
    > "${SAVE_ROOT}/UKan_PP_${ds}/IS.txt" 2>&1 &
  GPU=$((GPU + 1))
done
wait

echo "=== IS computation complete ==="
echo "Results:"
for ds in busi glas cvc; do
  echo "--- ${ds} ---"
  cat "${SAVE_ROOT}/UKan_PP_${ds}/FID.txt"
  cat "${SAVE_ROOT}/UKan_PP_${ds}/IS.txt"
done
