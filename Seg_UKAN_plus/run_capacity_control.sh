#!/bin/bash
# Table IX: Capacity-controlled routing ablation on BUSI
# Widened single-scale KAN (grid size 5, channels 128,320,352) x 3 seeds = 3 runs
# Segmentation-only, STAG and USG disabled
# The FR-KAN row is the FR-KAN-only BUSI configuration (busi_frkan_seed* in run_table1.sh)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../data}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}}"
OUTPUT_DIR="outputs"
cd "$SCRIPT_DIR"

SEEDS=(2981 6142 1187)
GPU=0

$PYTHON - <<'EOF'
import archs
count = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
single = archs.UKANPP(1, 3, False, embed_dims=[128, 320, 352], ms_stages='none')
frkan = archs.UKANPP(1, 3, False, embed_dims=[128, 160, 256], grid_sizes=[3, 5, 10])
print('Widened single-scale KAN: %.2fM parameters' % count(single))
print('FR-KAN (all stages):      %.2fM parameters' % count(frkan))
EOF

for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
    --name "busi_single_scale_352_seed${seed}" \
    --dataset busi \
    --data_dir "$DATA_DIR" \
    --arch UKANPP \
    --ms_stages none \
    --input_list 128,320,352 \
    --epochs 400 \
    --batch_size 8 \
    --input_h 256 --input_w 256 \
    --dataseed $seed \
    --output_dir $OUTPUT_DIR &
  GPU=$(( (GPU + 1) % 8 ))
done

wait
echo "=== Capacity-controlled routing ablation complete ==="
