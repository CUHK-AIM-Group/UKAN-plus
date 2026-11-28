#!/usr/bin/env python3
"""Generate predictions for baseline models (train_baseline.py checkpoints)."""
import argparse, os, yaml
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from albumentations import Resize, Compose
from albumentations.augmentations import transforms
import cv2
from baselines import build_baseline_2d
from dataset import Dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data_dir', default='../data')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load config
    exp_dir = os.path.dirname(args.checkpoint)
    with open(os.path.join(exp_dir, 'config.yml')) as f:
        config = yaml.safe_load(f)

    # Model
    model = build_baseline_2d(config['arch'], config['input_h'], pretrained=False).cuda()
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    # Dataset
    dataset_name = args.dataset
    img_ext = '.png'
    mask_ext = '_mask.png' if dataset_name == 'busi' else '.png'

    img_ids = sorted(glob(os.path.join(args.data_dir, dataset_name, 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    val_transform = Compose([Resize(config['input_h'], config['input_w']), transforms.Normalize()])
    val_dataset = Dataset(img_ids=img_ids,
                          img_dir=os.path.join(args.data_dir, dataset_name, 'images'),
                          mask_dir=os.path.join(args.data_dir, dataset_name, 'masks'),
                          img_ext=img_ext, mask_ext=mask_ext,
                          num_classes=config['num_classes'], transform=val_transform)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Output dir
    out_dir = os.path.join(exp_dir, 'out_val')
    os.makedirs(out_dir, exist_ok=True)

    # Generate predictions
    with torch.no_grad():
        for inp, _, meta in tqdm(val_loader, desc='Generating predictions'):
            inp = inp.cuda()
            output = torch.sigmoid(model(inp))
            output = output.cpu().numpy()[0, 0]
            output = (output * 255).astype(np.uint8)

            img_id = meta['img_id'][0]
            cv2.imwrite(os.path.join(out_dir, img_id + '.jpg'), output)

    print(f"Saved predictions to {out_dir}")

if __name__ == '__main__':
    main()
