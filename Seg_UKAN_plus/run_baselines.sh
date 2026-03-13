#!/bin/bash
# Train baseline methods for qualitative comparison and table filling
set -e

PYTHON="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/.venv/bin/python"
SCRIPT="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/U-KAN/Seg_UKAN/train_baseline.py"
cd /mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/U-KAN/Seg_UKAN

GPU=0

launch() {
  local name=$1 arch=$2 ds=$3 seed=$4
  if [ -f "outputs/${name}/model.pth" ]; then
    echo "SKIP: ${name} (already trained)"
    return
  fi
  mkdir -p "outputs"
  echo "GPU ${GPU}: Training ${name}"
  CUDA_VISIBLE_DEVICES=${GPU} nohup $PYTHON $SCRIPT \
    --name "$name" --arch "$arch" --dataset "$ds" \
    --dataseed "$seed" --epochs 400 \
    > "outputs/${name}.log" 2>&1 &
  GPU=$(( (GPU + 1) % 8 ))
}

# ===== Group 1: Qualitative figures (seed 2981, 3 methods × 4 datasets = 12 jobs) =====
for arch in unet attunet transunet; do
  for ds in busi isic2018 glas cvc; do
    launch "baseline_${arch}_${ds}_s2981" "$arch" "$ds" 2981
  done
done

# ===== Group 2: ISIC table (additional seeds 6142, 1187 for 4 methods = 8 jobs) =====
for arch in unet attunet unetpp transunet; do
  for seed in 6142 1187; do
    launch "baseline_${arch}_isic2018_s${seed}" "$arch" "isic2018" "$seed"
  done
done

# ===== Group 3: TransUNet table (additional seeds for BUSI/GlaS/CVC = 6 jobs) =====
for ds in busi glas cvc; do
  for seed in 6142 1187; do
    launch "baseline_transunet_${ds}_s${seed}" "transunet" "$ds" "$seed"
  done
done

echo ""
echo "All training jobs launched. Total: up to 26 jobs."
echo "Monitor: ps aux | grep train_baseline | grep -v grep | wc -l"
