#!/usr/bin/env python3
"""Collect baseline training results for table filling."""
import os
import pandas as pd
import numpy as np
import yaml
from glob import glob

OUTPUTS_DIR = 'outputs'

# Collect results
results = []
for log_path in glob(os.path.join(OUTPUTS_DIR, 'baseline_*/log.csv')):
    exp_dir = os.path.dirname(log_path)
    config_path = os.path.join(exp_dir, 'config.yml')
    if not os.path.exists(config_path):
        continue
    with open(config_path) as f:
        config = yaml.safe_load(f)

    test_path = os.path.join(exp_dir, 'test.csv')
    if os.path.exists(test_path):
        # separate test split: metrics of the best validation checkpoint
        row = pd.read_csv(test_path).iloc[0]
        best_dice, best_iou = row['dice'] * 100, row['iou'] * 100
    else:
        df = pd.read_csv(log_path)
        best_idx = df['val_iou'].idxmax()
        best_dice = df.loc[best_idx, 'val_dice'] * 100
        best_iou = df.loc[best_idx, 'val_iou'] * 100

    results.append({
        'arch': config['arch'],
        'dataset': config['dataset'],
        'seed': str(config['dataseed']),
        'dice': best_dice,
        'iou': best_iou
    })

df_results = pd.DataFrame(results)

# Group by arch and dataset
print("=" * 80)
print("BASELINE RESULTS (Dice / IoU)")
print("=" * 80)
for arch in ['unet', 'attunet', 'unetpp', 'transunet', 'unext', 'rollingunet', 'umamba']:
    for ds in ['busi', 'isic2018', 'glas', 'cvc']:
        subset = df_results[(df_results['arch'] == arch) & (df_results['dataset'] == ds)]
        if len(subset) > 0:
            dice_mean = subset['dice'].mean()
            dice_std = subset['dice'].std()
            iou_mean = subset['iou'].mean()
            iou_std = subset['iou'].std()
            print(f"{arch:12s} {ds:10s}: {dice_mean:5.2f}±{dice_std:4.2f} / {iou_mean:5.2f}±{iou_std:4.2f}")
