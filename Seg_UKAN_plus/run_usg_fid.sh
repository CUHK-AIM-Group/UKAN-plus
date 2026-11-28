#!/bin/bash
# Table VI and Fig. 8 (generation path of the joint model) and the FID/IS entries of Table V
# Joint and generation-only runs on BUSI, GlaS, CVC x 3 seeds = 18 runs
# Each run: 2048 samples at 64x64, FID (pytorch-fid) and IS
# Requires the runs from run_table7_usg.sh and run_batch3_usg_3d.sh
# Reference images: images_64 folders prepared with Diffusion_UKAN_plus/tools, the same
# preparation as the training views of usg_train.py --gen_view ukan (default)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
OUTPUT_DIR="outputs"
REF_ROOT="${REF_ROOT:-${SCRIPT_DIR}/../Diffusion_UKAN_plus/data}"
IS_DIR="${SCRIPT_DIR}/../Diffusion_UKAN_plus/inception-score-pytorch"
cd "$SCRIPT_DIR"

SEEDS=(2981 6142 1187)
GPU=0

for ds in busi glas cvc; do
  for cfg in joint_dynamic genonly; do
    for seed in "${SEEDS[@]}"; do
      RUN="${OUTPUT_DIR}/usg_${cfg}_${ds}_seed${seed}"
      (
        export CUDA_VISIBLE_DEVICES=$GPU
        $PYTHON usg_sample.py --exp_dir "$RUN" &&
        $PYTHON -m pytorch_fid "${REF_ROOT}/${ds}/images_64/" "${RUN}/Gens" > "${RUN}/FID.txt" 2>&1 &&
        (cd "$IS_DIR" && $PYTHON inception_score.py --data-root "${SCRIPT_DIR}/${RUN}/Gens") > "${RUN}/IS.txt" 2>&1
      ) &
      GPU=$(( (GPU + 1) % 8 ))
    done
  done
done

wait
echo "=== Generation evaluation complete ==="
for ds in busi glas cvc; do
  for cfg in joint_dynamic genonly; do
    for seed in "${SEEDS[@]}"; do
      RUN="${OUTPUT_DIR}/usg_${cfg}_${ds}_seed${seed}"
      echo "--- usg_${cfg}_${ds}_seed${seed} ---"
      tail -n 1 "${RUN}/FID.txt"
      tail -n 1 "${RUN}/IS.txt"
    done
  done
done
