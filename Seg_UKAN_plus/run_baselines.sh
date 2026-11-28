#!/bin/bash
# Train baseline methods for qualitative comparison and table filling
# TransUNet is the official R50-ViT-B_16 model: set TRANSUNET_DIR (and optionally
# TRANSUNET_WEIGHTS), see baselines.py. GlaS is trained at 512x512, the others at 256x256.
# ISIC 2018 models are trained from scratch (--no_pretrained).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
SCRIPT="${SCRIPT_DIR}/train_baseline.py"
cd "$SCRIPT_DIR"

GPU=0

launch() {
  local name=$1 arch=$2 ds=$3 seed=$4
  local size=256 extra=""
  [ "$ds" = glas ] && size=512
  [ "$ds" = isic2018 ] && extra="--no_pretrained"
  if [ -f "outputs/${name}/model.pth" ]; then
    echo "SKIP: ${name} (already trained)"
    return
  fi
  mkdir -p "outputs"
  echo "GPU ${GPU}: Training ${name}"
  CUDA_VISIBLE_DEVICES=${GPU} nohup $PYTHON $SCRIPT \
    --name "$name" --arch "$arch" --dataset "$ds" \
    --input_h "$size" --input_w "$size" $extra \
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
