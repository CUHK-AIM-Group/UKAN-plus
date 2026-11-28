"""Table VII: anatomical proxy assessment of generated images.

A U-Net segmenter trained with train_baseline.py on the real training split is
frozen and applied to the generated images of every generator with the same
preprocessing (the validation transform of train_baseline.py). The
target-class probability is thresholded at 0.5, and an image counts as a
detection when the binary prediction contains a connected component covering
at least 0.5% of the image. The detection rate is the proportion of detected
images among the first --n_images (1000) images of each directory, reported
with one decimal place.

Usage:
    python anatomical_proxy.py --segmenter outputs/proxy_unet_busi \
        --gen "Diffusion UKAN++=outputs/usg_joint_dynamic_busi_seed2981/Gens_proxy" \
        --gen "Diffusion UKAN=/path/to/diffusion_ukan/busi" --gen "Medfusion=/path/to/medfusion/busi"
"""
import argparse
import csv
import os
from glob import glob

import cv2
import numpy as np
import torch
import yaml
from albumentations import Resize
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from scipy.ndimage import generate_binary_structure, label as connected_components

from baselines import build_baseline_2d

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def load_segmenter(exp_dir, device):
    with open(os.path.join(exp_dir, 'config.yml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = build_baseline_2d(config['arch'], config['input_h'], pretrained=False)
    model.load_state_dict(torch.load(os.path.join(exp_dir, 'model.pth'), map_location='cpu'))
    transform = Compose([Resize(config['input_h'], config['input_w']), transforms.Normalize()])
    return model.to(device).eval(), transform


def preprocess(path, transform):
    img = cv2.imread(path)  # BGR, as in dataset.Dataset
    img = transform(image=img)['image'].astype('float32') / 255
    return img.transpose(2, 0, 1)


def largest_component_fraction(binary, structure):
    labels, n = connected_components(binary, structure=structure)
    if n == 0:
        return 0.0
    return np.bincount(labels.ravel())[1:].max() / binary.size


@torch.no_grad()
def evaluate_dir(model, transform, gen_dir, args, structure):
    paths = sorted(p for p in glob(os.path.join(gen_dir, '*')) if p.lower().endswith(IMG_EXTS))
    if len(paths) < args.n_images:
        print(f'  warning: {gen_dir} has {len(paths)} images (< {args.n_images})')
    paths = paths[:args.n_images]
    rows = []
    for i in range(0, len(paths), args.batch_size):
        batch = paths[i:i + args.batch_size]
        x = torch.from_numpy(np.stack([preprocess(p, transform) for p in batch])).to(args.device)
        prob = torch.sigmoid(model(x))[:, 0].cpu().numpy()
        for p, pr in zip(batch, prob):
            frac = largest_component_fraction(pr > args.threshold, structure)
            rows.append({'image': os.path.basename(p), 'largest_component_fraction': frac,
                         'detected': int(frac >= args.min_area)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--segmenter', required=True, help='train_baseline.py experiment directory')
    parser.add_argument('--gen', action='append', required=True, metavar='NAME=DIR',
                        help='generator name and directory of generated images (repeatable)')
    parser.add_argument('--n_images', type=int, default=1000)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--min_area', type=float, default=0.005, help='fraction of the image area')
    parser.add_argument('--connectivity', type=int, default=8, choices=[4, 8])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--out_csv', default=None, help='default: <segmenter>/anatomical_proxy.csv')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    model, transform = load_segmenter(args.segmenter, args.device)
    structure = generate_binary_structure(2, 2 if args.connectivity == 8 else 1)
    out_csv = args.out_csv or os.path.join(args.segmenter, 'anatomical_proxy.csv')
    summary = []
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['generator', 'image', 'largest_component_fraction', 'detected'])
        writer.writeheader()
        for spec in args.gen:
            name, gen_dir = spec.split('=', 1)
            rows = evaluate_dir(model, transform, gen_dir, args, structure)
            for row in rows:
                writer.writerow({'generator': name, **row})
            rate = 100.0 * sum(r['detected'] for r in rows) / max(len(rows), 1)
            summary.append((name, rate, len(rows)))

    print(f'Segmenter: {args.segmenter}')
    for name, rate, n in summary:
        print(f'  {name}: {rate:.1f}% ({n} images)')
    print(f'Per-image results: {out_csv}')


if __name__ == '__main__':
    main()
