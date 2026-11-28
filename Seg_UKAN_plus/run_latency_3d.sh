#!/bin/bash
# Table X: 3D segmentation latency on one GPU (benchmark_3d_latency.py)
# U-KAN, U-KAN++, the widened single-scale KAN control, UNETR++ and 3D U-Net;
# FP16, batch size 1, 5 warm-up + 10 timed repetitions;
# synthetic 96^3 patch forward and BraTS ROI sliding-window inference (96^3, 50% overlap)
# UNETR++ is built from the official repository: set UNETR_PP_DIR (see baselines.py)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../data}"
BRATS_DIR="${BRATS_DIR:-${DATA_ROOT}/brats2021}"
cd "$SCRIPT_DIR"

CUDA_VISIBLE_DEVICES=${GPU:-0} $PYTHON benchmark_3d_latency.py \
  --brats_dir "$BRATS_DIR" --brats_split val --case_index 0 \
  --out_csv outputs_3d/latency_3d.csv
