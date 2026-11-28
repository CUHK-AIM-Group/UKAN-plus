#!/bin/bash
# Redesigned MS-KAN Ablation on BUSI (v2)
# 8 independent configs × 3 seeds = 24 runs
#
# (a) Activation function
#   ① U-KAN baseline (G=5, single KAN)          → arch=UKAN
#   ② MLP (no KAN, multi-branch with MLP)        → arch=UKANPP, --no_kan
# (b) MS-KAN position
#   ③ MS-KAN only encoder                        → --ms_stages enc
#   ④ MS-KAN only bottleneck                     → --ms_stages btl
#   ⑤ MS-KAN only decoder                        → --ms_stages dec
#   ⑥ MS-KAN all stages (ours, = ⑨ = ⑩)         → --ms_stages all (default)
# (c) Fusion strategy
#   ⑦ Sum fusion                                 → --fusion_type sum
#   ⑧ Concat+1×1 fusion                          → --fusion_type concat
#   ⑨ Softmax attn fusion (= ⑥)                  → already covered
# (d) Cross-branch interaction
#   ⑩ Independent branches (= ⑥)                 → already covered
#   ⑪ Cross-branch attention                     → --cross_branch_attn

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../data}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}}"
OUTPUT_DIR="outputs"
cd "$SCRIPT_DIR"

SEEDS=(2981 6142 1187)
COMMON_ARGS="--dataset busi --data_dir $DATA_DIR --epochs 400 --batch_size 8 --input_h 256 --input_w 256 --output_dir $OUTPUT_DIR"

GPU=0
BATCH=0

run_job() {
    local name=$1
    shift
    for seed in "${SEEDS[@]}"; do
        echo "[GPU=$GPU] ${name}_seed${seed}"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
            --name "${name}_seed${seed}" \
            --dataseed $seed \
            $COMMON_ARGS \
            "$@" &

        BATCH=$(( BATCH + 1 ))
        GPU=$(( (GPU + 1) % 8 ))

        # Wait every 8 concurrent jobs
        if [ $((BATCH % 8)) -eq 0 ]; then
            wait
        fi
    done
}

# ① U-KAN baseline (single-scale G=5 KAN)
run_job "ablv2_ukan_baseline" \
    --arch UKAN

# ② MLP (no KAN, multi-branch with linear layers)
run_job "ablv2_mlp_no_kan" \
    --arch UKANPP --no_kan --grid_sizes "3,5,10"

# ③ MS-KAN only encoder
run_job "ablv2_ms_enc_only" \
    --arch UKANPP --ms_stages enc --grid_sizes "3,5,10" --fusion_type softmax

# ④ MS-KAN only bottleneck
run_job "ablv2_ms_btl_only" \
    --arch UKANPP --ms_stages btl --grid_sizes "3,5,10" --fusion_type softmax

# ⑤ MS-KAN only decoder
run_job "ablv2_ms_dec_only" \
    --arch UKANPP --ms_stages dec --grid_sizes "3,5,10" --fusion_type softmax

# ⑥ MS-KAN all stages — softmax — independent branches (ours, = ⑨ = ⑩)
run_job "ablv2_ms_all_softmax" \
    --arch UKANPP --ms_stages all --grid_sizes "3,5,10" --fusion_type softmax

# ⑦ Sum fusion
run_job "ablv2_ms_all_sum" \
    --arch UKANPP --ms_stages all --grid_sizes "3,5,10" --fusion_type sum

# ⑧ Concat+1×1 fusion
run_job "ablv2_ms_all_concat" \
    --arch UKANPP --ms_stages all --grid_sizes "3,5,10" --fusion_type concat

# ⑪ Cross-branch attention
run_job "ablv2_ms_all_crossattn" \
    --arch UKANPP --ms_stages all --grid_sizes "3,5,10" --fusion_type softmax --cross_branch_attn

wait
echo "=== MS-KAN v2 ablation complete (8 configs × 3 seeds = 24 runs, plus UKAN baseline 3 runs = 27 total) ==="
