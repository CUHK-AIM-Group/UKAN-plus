#!/usr/bin/env python3
"""Collect baseline training results for table filling."""
import os
import pandas as pd
import numpy as np
from glob import glob

OUTPUTS_DIR = 'outputs'

# Collect results
results = []
for log_path in glob(os.path.join(OUTPUTS_DIR, 'baseline_*/log.csv')):
    name = os.path.basename(os.path.dirname(log_path))
    parts = name.replace('baseline_', '').split('_')

    # Parse: baseline_{arch}_{dataset}_s{seed}
    if len(parts) >= 3 and parts[-1].startswith('s'):
        seed = parts[-1][1:]
        dataset = parts[-2]
        arch = '_'.join(parts[:-2])

        df = pd.read_csv(log_path)
        best_idx = df['val_iou'].idxmax()
        best_dice = df.loc[best_idx, 'val_dice'] * 100
        best_iou = df.loc[best_idx, 'val_iou'] * 100

        results.append({
            'arch': arch,
            'dataset': dataset,
            'seed': seed,
            'dice': best_dice,
            'iou': best_iou
        })

df_results = pd.DataFrame(results)

# Group by arch and dataset
print("=" * 80)
print("BASELINE RESULTS (Dice / IoU)")
print("=" * 80)
for arch in ['unet', 'attunet', 'unetpp', 'transunet']:
    for ds in ['busi', 'isic2018', 'glas', 'cvc']:
        subset = df_results[(df_results['arch'] == arch) & (df_results['dataset'] == ds)]
        if len(subset) > 0:
            dice_mean = subset['dice'].mean()
            dice_std = subset['dice'].std()
            iou_mean = subset['iou'].mean()
            iou_std = subset['iou'].std()
            print(f"{arch:12s} {ds:10s}: {dice_mean:5.2f}±{dice_std:4.2f} / {iou_mean:5.2f}±{iou_std:4.2f}")
