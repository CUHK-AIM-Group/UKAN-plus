#!/bin/bash
# Table VII: anatomical proxy of generated images (anatomical_proxy.py)
# 1) U-Net segmenters trained on the real training splits of BUSI, GlaS, CVC-ClinicDB
#    (train_baseline.py, seed 2981); each checkpoint is fixed before any generated image is scored
# 2) 1000 new unconditional samples per dataset from the joint model (Diffusion UKAN++),
#    sampled with seed 1; the FID/IS samples of run_usg_fid.sh use seed 0
# 3) Detection rates with the frozen segmenter (threshold 0.5, component >= 0.5% of the image)
#    for Diffusion UKAN++ and, if provided, for Diffusion UKAN and Medfusion:
#    DUKAN_PROXY_ROOT/<dataset> and MEDFUSION_PROXY_ROOT/<dataset> hold 1000 unconditional
#    samples per dataset generated separately from the FID/IS samples
# Requires the joint runs of run_table7_usg.sh and run_batch3_usg_3d.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../data}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}}"
OUTPUT_DIR="outputs"
JOINT_SEED="${JOINT_SEED:-2981}"
cd "$SCRIPT_DIR"

input_size() { [ "$1" = glas ] && echo 512 || echo 256; }

GPU=0
for ds in busi glas cvc; do
  size=$(input_size "$ds")
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON train_baseline.py \
    --name "proxy_unet_${ds}" --arch unet --dataset "$ds" --data_dir "$DATA_DIR" \
    --input_h "$size" --input_w "$size" --dataseed 2981 --output_dir $OUTPUT_DIR &
  GPU=$(( (GPU + 1) % 8 ))
done
wait

for ds in busi glas cvc; do
  RUN="${OUTPUT_DIR}/usg_joint_dynamic_${ds}_seed${JOINT_SEED}"
  $PYTHON usg_sample.py --exp_dir "$RUN" --n_samples 1000 --seed 1 --out_dir "${RUN}/Gens_proxy"
  GENS=(--gen "Diffusion UKAN++=${RUN}/Gens_proxy")
  [ -n "$DUKAN_PROXY_ROOT" ] && GENS+=(--gen "Diffusion UKAN=${DUKAN_PROXY_ROOT}/${ds}")
  [ -n "$MEDFUSION_PROXY_ROOT" ] && GENS+=(--gen "Medfusion=${MEDFUSION_PROXY_ROOT}/${ds}")
  $PYTHON anatomical_proxy.py --segmenter "${OUTPUT_DIR}/proxy_unet_${ds}" "${GENS[@]}"
done
echo "=== Anatomical proxy complete ==="
