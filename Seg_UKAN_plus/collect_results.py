#!/usr/bin/env python3
import os, yaml, pandas as pd
from glob import glob

results = []
for ckpt_dir in sorted(glob('outputs/baseline_*')):
    if not os.path.isdir(ckpt_dir):
        continue
    
    config_path = os.path.join(ckpt_dir, 'config.yml')
    log_path = os.path.join(ckpt_dir, 'log.csv')
    
    if not os.path.exists(config_path) or not os.path.exists(log_path):
        continue
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    test_path = os.path.join(ckpt_dir, 'test.csv')
    if os.path.exists(test_path):
        # separate test split: metrics of the best validation checkpoint
        row = pd.read_csv(test_path).iloc[0]
        val_dice, val_iou = row['dice'] * 100, row['iou'] * 100
    else:
        df = pd.read_csv(log_path)
        best_idx = df['val_iou'].idxmax()
        val_dice = df.loc[best_idx, 'val_dice'] * 100
        val_iou = df.loc[best_idx, 'val_iou'] * 100
    
    results.append({
        'arch': config['arch'],
        'dataset': config['dataset'],
        'seed': str(config['dataseed']),
        'dice': val_dice,
        'iou': val_iou
    })
    print(f"{os.path.basename(ckpt_dir)}: Dice={val_dice:.2f}, IoU={val_iou:.2f}")

df = pd.DataFrame(results)
print("\n" + "="*80)
print("SUMMARY (Dice / IoU)")
print("="*80)
for arch in ['unet', 'attunet', 'unetpp', 'transunet', 'unext', 'rollingunet', 'umamba']:
    for ds in ['busi', 'isic2018', 'glas', 'cvc']:
        subset = df[(df['arch'] == arch) & (df['dataset'] == ds)]
        if len(subset) > 0:
            dice_mean, dice_std = subset['dice'].mean(), subset['dice'].std()
            iou_mean, iou_std = subset['iou'].mean(), subset['iou'].std()
            n = len(subset)
            print(f"{arch:12s} {ds:10s} (n={n}): {dice_mean:5.2f}±{dice_std:4.2f} / {iou_mean:5.2f}±{iou_std:4.2f}")
