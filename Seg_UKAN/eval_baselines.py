#!/usr/bin/env python3
"""Evaluate all baseline models to get validation metrics."""
import os, yaml, torch
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from albumentations import Resize, Compose
from albumentations.augmentations import transforms
import segmentation_models_pytorch as smp
from dataset import Dataset
from metrics import iou_score

ARCH_MAP = {
    'unet': lambda: smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1),
    'attunet': lambda: smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, decoder_attention_type='scse'),
    'unetpp': lambda: smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1),
    'transunet': lambda: smp.Unet(encoder_name='mit_b2', encoder_weights='imagenet', in_channels=3, classes=1),
}

results = []
for ckpt_dir in sorted(glob('outputs/baseline_*_s*')):
    if not os.path.isdir(ckpt_dir) or not os.path.exists(os.path.join(ckpt_dir, 'model.pth')):
        continue

    name = os.path.basename(ckpt_dir)
    parts = name.replace('baseline_', '').split('_')
    if len(parts) < 3 or not parts[-1].startswith('s'):
        print(f"Skipping {name}: invalid format")
        continue

    seed_part = parts[-1]
    if seed_part.startswith('seed'):
        seed = seed_part[4:]  # Remove 'seed' prefix
    elif seed_part.startswith('s'):
        seed = seed_part[1:]  # Remove 's' prefix
    else:
        seed = seed_part
    dataset = parts[-2]
    arch = '_'.join(parts[:-2])

    print(f"Processing {name}: arch={arch}, dataset={dataset}, seed={seed}")

    config_path = os.path.join(ckpt_dir, 'config.yml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load model
    model = ARCH_MAP[arch]().cuda()
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'model.pth')))
    model.eval()

    # Prepare validation data
    img_ext = '.png'
    mask_ext = '_mask.png' if dataset == 'busi' else '.png'
    img_ids = sorted(glob(os.path.join(config['data_dir'], dataset, 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    _, val_ids = train_test_split(img_ids, test_size=0.2, random_state=int(seed))

    val_transform = Compose([Resize(config['input_h'], config['input_w']), transforms.Normalize()])
    val_dataset = Dataset(img_ids=val_ids,
                          img_dir=os.path.join(config['data_dir'], dataset, 'images'),
                          mask_dir=os.path.join(config['data_dir'], dataset, 'masks'),
                          img_ext=img_ext, mask_ext=mask_ext,
                          num_classes=config['num_classes'], transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # Evaluate
    ious, dices = [], []
    with torch.no_grad():
        for inp, target, _ in val_loader:
            inp, target = inp.cuda(), target.cuda()
            output = model(inp)
            iou, dice, _ = iou_score(output, target)
            ious.append(iou)
            dices.append(dice)

    val_iou = np.mean(ious) * 100
    val_dice = np.mean(dices) * 100

    results.append({'arch': arch, 'dataset': dataset, 'seed': seed, 'dice': val_dice, 'iou': val_iou})
    print(f"{name}: Dice={val_dice:.2f}, IoU={val_iou:.2f}")

# Summary
import pandas as pd
df = pd.DataFrame(results)
print("\n" + "="*80)
print("SUMMARY (Dice / IoU)")
print("="*80)
for arch in ['unet', 'attunet', 'unetpp', 'transunet']:
    for ds in ['busi', 'isic2018', 'glas', 'cvc']:
        subset = df[(df['arch'] == arch) & (df['dataset'] == ds)]
        if len(subset) > 0:
            dice_mean, dice_std = subset['dice'].mean(), subset['dice'].std()
            iou_mean, iou_std = subset['iou'].mean(), subset['iou'].std()
            print(f"{arch:12s} {ds:10s}: {dice_mean:5.2f}±{dice_std:4.2f} / {iou_mean:5.2f}±{iou_std:4.2f}")
