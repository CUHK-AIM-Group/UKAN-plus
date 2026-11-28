"""Sliding-window evaluation of 3D checkpoints from train_3d.py (Tables II-IV).

Whole volumes are segmented with sliding windows (window = training crop
size, 50% overlap by default). Synapse reports per-organ Dice and HD95, ACDC
per-structure Dice and BraTS ET/TC/WT Dice; averages weight all classes or
regions equally.

Per-case conventions: Synapse and ACDC follow TransUNet (prediction empty:
Dice 0, HD95 0; prediction non-empty and ground truth empty: Dice 1, HD95 0).
BraTS uses the smoothed Dice of eval_brats_perclass.py (both empty: 1).

HD95 is computed between surface voxels. The TransUNet Synapse volumes carry
no voxel spacing, so HD95 is in voxels unless --spacing or --spacing_json
gives the spacing in mm (D,H,W order).

Usage:
    python eval_3d.py --exp_dir outputs_3d/synapse_ukanpp3d_seed2981 [--spacing_json spacing.json]
    python eval_3d.py --exp_dir outputs_3d/brats_ukanpp3d_seed* --data_dir /data/brats2021
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
import yaml
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

from archs_3d import UKANPP3D
from dataset_3d import get_dataset
from sliding_window import sliding_window_inference
from train_3d import DATASET_CONFIGS

CLASS_NAMES = {
    'synapse': ['Aorta', 'Gallbladder', 'Kidney_L', 'Kidney_R', 'Liver', 'Pancreas', 'Spleen', 'Stomach'],
    'acdc': ['RV', 'Myo', 'LV'],
    'brats2021': ['ET', 'TC', 'WT'],
}
DEFAULT_SPLIT = {'synapse': 'test', 'acdc': 'test', 'brats2021': 'val'}


def surface(mask):
    return mask ^ binary_erosion(mask, structure=generate_binary_structure(mask.ndim, 1))


def hd95(pred, gt, spacing=None):
    """95th percentile symmetric surface distance (same definition as medpy.metric.binary.hd95)."""
    box = tuple(slice(i.min(), i.max() + 1) for i in np.nonzero(pred | gt))
    pred, gt = pred[box], gt[box]
    sp, sg = surface(pred), surface(gt)
    d_pg = distance_transform_edt(~sg, sampling=spacing)[sp]
    d_gp = distance_transform_edt(~sp, sampling=spacing)[sg]
    return float(np.percentile(np.hstack([d_pg, d_gp]), 95))


def transunet_metrics(pred, gt, spacing, compute_hd):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = 2.0 * (pred & gt).sum() / (pred.sum() + gt.sum())
        return dice, hd95(pred, gt, spacing) if compute_hd else float('nan')
    if pred.sum() > 0:
        return 1.0, 0.0
    return 0.0, 0.0


def smoothed_dice(pred, gt, smooth=1e-5):
    return (2.0 * (pred & gt).sum() + smooth) / (pred.sum() + gt.sum() + smooth)


def class_masks(dataset, label):
    if dataset == 'brats2021':
        return [label == 3, (label == 1) | (label == 3), label >= 1]
    return [label == c for c in range(1, len(CLASS_NAMES[dataset]) + 1)]


def build_model(config, ds_cfg):
    return UKANPP3D(
        num_classes=ds_cfg['num_classes'], input_channels=ds_cfg['input_channels'],
        embed_dims=[int(x) for x in config['embed_dims'].split(',')],
        grid_sizes=[int(x) for x in config['grid_sizes'].split(',')],
        fusion_type=config.get('fusion_type', 'softmax'), no_kan=config.get('no_kan', False),
        use_stag=config.get('use_stag', False), stag_stages=config.get('stag_stages', 'dec'),
        stag_alpha=config.get('stag_alpha', 0.5), stag_norm=config.get('stag_norm', 'minmax'))


def evaluate(exp_dir, args):
    with open(os.path.join(exp_dir, 'config.yml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    dataset = config['dataset']
    ds_cfg = DATASET_CONFIGS[dataset]
    window = tuple(int(x) for x in config['crop_size'].split(',')) if config.get('crop_size') \
        else ds_cfg['crop_size']
    names = CLASS_NAMES[dataset]
    compute_hd = dataset == 'synapse' and not args.no_hd95 or args.hd95_all
    spacings = {}
    if args.spacing_json:
        with open(args.spacing_json) as f:
            spacings = json.load(f)
    default_spacing = [float(x) for x in args.spacing.split(',')] if args.spacing else None
    unit = 'mm' if (default_spacing or spacings) else 'voxels'

    model = build_model(config, ds_cfg)
    model.load_state_dict(torch.load(os.path.join(exp_dir, args.checkpoint), map_location='cpu'))
    model = model.to(args.device).eval()

    data = get_dataset(dataset, args.data_dir or config['data_dir'], split=args.split or DEFAULT_SPLIT[dataset],
                       crop_size=window, augment=False, full_volume=True)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=args.num_workers)

    rows = []
    for image, label, case in loader:
        logits = sliding_window_inference(image.to(args.device), model, window, args.overlap,
                                          args.sw_batch_size)
        pred = logits.argmax(1)[0].cpu().numpy()
        gt = label[0].numpy()
        spacing = spacings.get(case[0], default_spacing)
        row = {'case': case[0]}
        for name, p, g in zip(names, class_masks(dataset, pred), class_masks(dataset, gt)):
            if dataset == 'brats2021':
                row[f'{name}_dice'] = smoothed_dice(p, g)
                if compute_hd:
                    row[f'{name}_hd95'] = transunet_metrics(p, g, spacing, True)[1]
            else:
                row[f'{name}_dice'], hd = transunet_metrics(p, g, spacing, compute_hd)
                if compute_hd:
                    row[f'{name}_hd95'] = hd
        rows.append(row)
        print(f"  {case[0]}: mean Dice {np.mean([row[f'{n}_dice'] for n in names]) * 100:.2f}")

    summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'case'}
    summary['Avg_dice'] = float(np.mean([summary[f'{n}_dice'] for n in names]))
    if compute_hd:
        summary['Avg_hd95'] = float(np.mean([summary[f'{n}_hd95'] for n in names]))

    with open(os.path.join(exp_dir, 'eval_3d_cases.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(exp_dir, 'eval_3d.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    print(f"{os.path.basename(exp_dir.rstrip('/'))} ({dataset}, {len(rows)} cases): "
          f"Avg Dice {summary['Avg_dice'] * 100:.2f}"
          + (f", Avg HD95 {summary['Avg_hd95']:.2f} {unit}" if compute_hd else ''))
    print('  ' + ', '.join(f"{n} {summary[f'{n}_dice'] * 100:.2f}" for n in names))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--exp_dir', nargs='+', required=True)
    parser.add_argument('--data_dir', default=None, help='defaults to data_dir in config.yml')
    parser.add_argument('--split', default=None, help='default: test (Synapse, ACDC), val (BraTS)')
    parser.add_argument('--checkpoint', default='model.pth')
    parser.add_argument('--overlap', default=0.5, type=float)
    parser.add_argument('--sw_batch_size', default=2, type=int)
    parser.add_argument('--spacing', default=None, help='voxel spacing in mm, "D,H,W"')
    parser.add_argument('--spacing_json', default=None, help='JSON {case: [D, H, W]} spacing in mm')
    parser.add_argument('--no_hd95', action='store_true', help='skip HD95 for Synapse')
    parser.add_argument('--hd95_all', action='store_true', help='also compute HD95 for ACDC and BraTS')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    for exp_dir in args.exp_dir:
        evaluate(exp_dir, args)


if __name__ == '__main__':
    main()
