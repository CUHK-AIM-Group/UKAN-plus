"""
STAG interpretability evaluation — Figs. 10 and 11.

Compares the STAG map with Grad-CAM, Grad-CAM++ and Score-CAM on the
validation split of a trained STAG checkpoint (train.py or usg_train.py):
Dice-based deletion/insertion faithfulness and localization (IoU@0.5,
Pointing Game). Deletion and insertion ranks are computed among the four
methods within the evaluated dataset.

Usage:
    python sam_eval.py \
        --checkpoint outputs/usg_joint_dynamic_busi_seed2981 \
        --output_dir results/stag_busi
"""
import argparse
import json
import os
import yaml
import numpy as np
import torch
import cv2
from glob import glob
from tqdm import tqdm
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import Resize

import archs
from dataset import Dataset, split_ids
from sam import (compute_stag, compute_cam, perturbation_curves,
                 compute_auc, compute_aopc,
                 iou_at_threshold, pointing_game)

METHODS = ['stag', 'gradcam', 'gradcampp', 'scorecam']
LABELS = {'stag': 'STAG', 'gradcam': 'Grad-CAM', 'gradcampp': 'Grad-CAM++', 'scorecam': 'Score-CAM'}


def load_model(checkpoint_dir, device='cuda', require_stag=True):
    """Load a train.py or usg_train.py checkpoint as a UKANPP segmentation model."""
    config_path = os.path.join(checkpoint_dir, 'config.yml')
    ckpt_path = os.path.join(checkpoint_dir, 'model.pth')

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    state = torch.load(ckpt_path, map_location=device)

    if 'arch' not in config:
        # usg_train.py checkpoint: keep the segmentation model of the joint network
        config.update(arch='UKANPP', num_classes=1, input_channels=3,
                      deep_supervision=False, input_list=[128, 160, 256])
        state = {k[len('seg_model.'):]: v for k, v in state.items()
                 if k.startswith('seg_model.')}
    if config['arch'] != 'UKANPP':
        raise ValueError('STAG evaluation requires a UKANPP checkpoint')

    model = archs.UKANPP(
        config['num_classes'], config['input_channels'], config['deep_supervision'],
        embed_dims=config['input_list'], no_kan=config.get('no_kan', False),
        grid_sizes=[int(g) for g in config.get('grid_sizes', '3,5,10').split(',')],
        fusion_type=config.get('fusion_type', 'softmax'),
        ms_stages=config.get('ms_stages', 'all'),
        cross_branch_attn=config.get('cross_branch_attn', False),
        use_stag=config.get('use_stag', False),
        stag_stages=config.get('stag_stages', 'dec'),
        stag_alpha=config.get('stag_alpha', 0.5),
        stag_detach=config.get('stag_detach', False),
        stag_norm=config.get('stag_norm', 'minmax'))
    model.load_state_dict(state)
    if require_stag and model.stag_dec1 is None and model.stag_dec2 is None:
        raise ValueError('the checkpoint was trained without STAG')
    model = model.to(device).eval()
    return model, config


def get_val_loader(config):
    """Build the evaluation dataloader (test split, see dataset.split_ids) from config."""
    dataset_name = config['dataset']
    img_ext = '.png'
    mask_ext_map = {'busi': '_mask.png'}
    mask_ext = mask_ext_map.get(dataset_name, '.png')

    img_ids = sorted(glob(os.path.join(
        config['data_dir'], dataset_name, 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    val_ids = split_ids(img_ids, config['data_dir'], dataset_name, 'test', config['dataseed'])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    ds = Dataset(
        img_ids=val_ids,
        img_dir=os.path.join(config['data_dir'], dataset_name, 'images'),
        mask_dir=os.path.join(config['data_dir'], dataset_name, 'masks'),
        img_ext=img_ext, mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)

    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    return loader


def explanation_maps(model, inp):
    maps = {'stag': compute_stag(model, inp)}
    for m in METHODS[1:]:
        maps[m] = compute_cam(model, inp, m)
    return maps


def evaluate(model, loader, device, num_samples=50, steps=100):
    """Faithfulness curves and localization scores for every method."""
    curves = {m: {'del': [], 'ins': []} for m in METHODS}
    loc = {m: {'iou05': [], 'pointing': []} for m in METHODS}

    total = min(num_samples, len(loader))
    for i, (inp, mask, meta) in enumerate(tqdm(loader, desc='Evaluating', total=total)):
        if i >= total:
            break
        inp = inp.to(device)
        gt = (mask[0, 0].numpy() > 0.5).astype(np.float32)

        for m, smap in explanation_maps(model, inp).items():
            d, s = perturbation_curves(model, inp, gt, smap, steps=steps)
            curves[m]['del'].append(d)
            curves[m]['ins'].append(s)
            # Localization is undefined for images without a target structure
            if gt.sum() > 0:
                loc[m]['iou05'].append(iou_at_threshold(smap, gt, 0.5))
                loc[m]['pointing'].append(pointing_game(smap, gt))
    return curves, loc


def summarize(curves, loc, output_dir):
    summary = {}
    for m in METHODS:
        summary[m] = {
            'deletion_auc': float(np.mean([compute_auc(c) for c in curves[m]['del']])),
            'insertion_auc': float(np.mean([compute_auc(c) for c in curves[m]['ins']])),
            'aopc': float(np.mean([compute_aopc(c) for c in curves[m]['del']])),
            'iou05': 100.0 * float(np.mean(loc[m]['iou05'])) if loc[m]['iou05'] else float('nan'),
            'pointing_game': 100.0 * float(np.mean(loc[m]['pointing'])) if loc[m]['pointing'] else float('nan'),
            'num_images': len(curves[m]['del']),
            'num_localization_images': len(loc[m]['iou05']),
        }
    del_order = sorted(METHODS, key=lambda m: summary[m]['deletion_auc'])
    ins_order = sorted(METHODS, key=lambda m: -summary[m]['insertion_auc'])
    for m in METHODS:
        summary[m]['deletion_rank'] = del_order.index(m) + 1
        summary[m]['insertion_rank'] = ins_order.index(m) + 1

    print("\n{:<12} {:>9} {:>9} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        'Method', 'Del AUC↓', 'Ins AUC↑', 'AOPC↑', 'Del rk', 'Ins rk', 'IoU@0.5', 'PG (%)'))
    print("-" * 78)
    for m in METHODS:
        r = summary[m]
        print("{:<12} {:>9.2f} {:>9.2f} {:>8.2f} {:>8d} {:>8d} {:>8.2f} {:>8.2f}".format(
            LABELS[m], r['deletion_auc'], r['insertion_auc'], r['aopc'],
            r['deletion_rank'], r['insertion_rank'], r['iou05'], r['pointing_game']))

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    np.savez(os.path.join(output_dir, 'curves.npz'), **{
        '{}_{}'.format(m, k): np.array(v) for m in METHODS for k, v in curves[m].items()})
    return summary


def plot_curves(curves, output_dir):
    """Mean deletion and insertion curves on the Dice-percentage axis."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print("Could not plot (matplotlib issue): {}".format(e))
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for m in METHODS:
        for ax, key in zip(axes, ['del', 'ins']):
            mean = np.mean(curves[m][key], axis=0)
            ax.plot(np.linspace(0, 1, len(mean)), mean, label=LABELS[m])
    axes[0].set_title('Deletion')
    axes[0].set_xlabel('Fraction of pixels removed')
    axes[1].set_title('Insertion')
    axes[1].set_xlabel('Fraction of pixels restored')
    for ax in axes:
        ax.set_ylabel('Dice (%)')
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'perturbation_curves.pdf'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'perturbation_curves.png'), dpi=150)
    plt.close()


def visualize(model, loader, device, output_dir, num_vis=8):
    """Save explanation-map overlays and GT masks for the first images."""
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    for i, (inp, mask, meta) in enumerate(loader):
        if i >= num_vis:
            break
        img_id = meta['img_id'][0]
        img_np = inp[0].permute(1, 2, 0).numpy()
        img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)

        for m, smap in explanation_maps(model, inp.to(device)).items():
            heatmap = cv2.applyColorMap((smap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
            cv2.imwrite(os.path.join(vis_dir, '{}_{}.png'.format(img_id, m)), overlay)

        gt_np = (mask[0, 0].numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(vis_dir, '{}_gt.png'.format(img_id)), gt_np)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='Path to checkpoint directory (contains config.yml + model.pth)')
    parser.add_argument('--output_dir', default='results/stag',
                        help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of validation images for faithfulness/localization')
    parser.add_argument('--num_vis', type=int, default=8,
                        help='Number of images for visualization')
    parser.add_argument('--steps', type=int, default=100,
                        help='Number of steps for deletion/insertion curves')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model from {}".format(args.checkpoint))
    model, config = load_model(args.checkpoint, args.device)

    print("Building validation loader for {}".format(config['dataset']))
    loader = get_val_loader(config)

    curves, loc = evaluate(model, loader, args.device,
                           num_samples=args.num_samples, steps=args.steps)
    summarize(curves, loc, args.output_dir)
    plot_curves(curves, args.output_dir)
    visualize(model, loader, args.device, args.output_dir, num_vis=args.num_vis)

    print("\nDone! Results saved to {}".format(args.output_dir))


if __name__ == '__main__':
    main()
