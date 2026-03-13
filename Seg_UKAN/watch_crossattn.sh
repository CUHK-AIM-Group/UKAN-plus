#!/bin/bash
# Monitor and auto-restart crossattn runs if they die
cd /mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/U-KAN/Seg_UKAN

PYTHON="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/.venv/bin/python"
DATA_DIR="/mnt/bn/ic-vlm-hl/personal/lichenxin/proj/PAMI_EXT.U-KAN/code/U-KAN/data"
COMMON="--dataset busi --data_dir $DATA_DIR --epochs 400 --batch_size 8 --input_h 256 --input_w 256 --output_dir outputs --arch UKANPP --ms_stages all --grid_sizes 3,5,10 --fusion_type softmax --cross_branch_attn"

while true; do
    all_done=true
    for seed in 2981 6142 1187; do
        name="ablv2_ms_all_crossattn_seed${seed}"
        log="outputs/${name}/log.csv"
        if [ -f "$log" ]; then
            ep=$(( $(wc -l < "$log") - 1 ))
            if [ "$ep" -ge 400 ]; then
                echo "[$(date)] $name: DONE ($ep/400)"
                continue
            fi
        fi
        all_done=false
        # Check if process is running
        if ! pgrep -f "train.py.*${name}" > /dev/null 2>&1; then
            echo "[$(date)] $name not running, restarting on GPU $((seed % 3))..."
            gpu=$((seed % 3))
            CUDA_VISIBLE_DEVICES=$gpu nohup $PYTHON train.py --name "$name" --dataseed $seed $COMMON > /dev/null 2>&1 &
        else
            if [ -f "$log" ]; then
                echo "[$(date)] $name: $ep/400"
            else
                echo "[$(date)] $name: initializing..."
            fi
        fi
    done
    if $all_done; then
        echo "[$(date)] === All crossattn runs complete! ==="
        exit 0
    fi
    sleep 120
done
