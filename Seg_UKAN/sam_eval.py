"""
SAM Evaluation Script — Tables IX, X and Figures 1, 2.

Usage:
    python sam_eval.py \
        --checkpoint outputs/busi_ukanpp_seed2981/model.pth \
        --dataset busi --data_dir ../data \
        --output_dir results/sam_busi
"""
import argparse
import os
import yaml
import numpy as np
import torch
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import Resize

import archs
from dataset import Dataset
from sam import (compute_sam, compute_gradcam, compute_random_baseline,
                 deletion_curve, insertion_curve,
                 compute_auc, compute_aopc,
                 iou_at_threshold, pointing_game)


def load_model(checkpoint_dir, device='cuda'):
    """Load model from checkpoint directory."""
    config_path = os.path.join(checkpoint_dir, 'config.yml')
    ckpt_path = os.path.join(checkpoint_dir, 'model.pth')

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config['arch'] == 'UKANPP':
        grid_sizes = [int(g) for g in config.get('grid_sizes', '3,5,10').split(',')]
        model = archs.__dict__[config['arch']](
            config['num_classes'], config['input_channels'],
            config['deep_supervision'],
            embed_dims=config['input_list'],
            no_kan=config.get('no_kan', False),
            grid_sizes=grid_sizes,
            fusion_type=config.get('fusion_type', 'softmax'))
    else:
        model = archs.__dict__[config['arch']](
            config['num_classes'], config['input_channels'],
            config['deep_supervision'],
            embed_dims=config['input_list'],
            no_kan=config.get('no_kan', False))

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device).eval()
    return model, config


def get_val_loader(config):
    """Build validation dataloader from config."""
    dataset_name = config['dataset']
    img_ext = '.png'
    mask_ext_map = {'busi': '_mask.png'}
    mask_ext = mask_ext_map.get(dataset_name, '.png')

    img_ids = sorted(glob(os.path.join(
        config['data_dir'], dataset_name, 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    _, val_ids = train_test_split(img_ids, test_size=0.2,
                                  random_state=config['dataseed'])

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


def evaluate_faithfulness(model, loader, device, output_dir, num_samples=50,
                          steps=100):
    """Table IX: Deletion/Insertion AUC, AOPC for SAM vs GradCAM vs Random."""
    results = {m: {'del_auc': [], 'ins_auc': [], 'aopc': []}
               for m in ['sam', 'gradcam', 'random']}

    count = 0
    for inp, mask, meta in tqdm(loader, desc='Faithfulness', total=min(num_samples, len(loader))):
        if count >= num_samples:
            break
        inp = inp.to(device)

        # Compute saliency maps
        sam_map = compute_sam(model, inp)
        gcam_map = compute_gradcam(model, inp)
        rand_map = compute_random_baseline(inp.shape[2], inp.shape[3])

        for name, smap in [('sam', sam_map), ('gradcam', gcam_map), ('random', rand_map)]:
            del_scores = deletion_curve(model, inp, smap, steps=steps)
            ins_scores = insertion_curve(model, inp, smap, steps=steps)
            results[name]['del_auc'].append(compute_auc(del_scores))
            results[name]['ins_auc'].append(compute_auc(ins_scores))
            results[name]['aopc'].append(compute_aopc(del_scores))

        count += 1

    # Aggregate
    print("\n=== Table IX: Faithfulness ===")
    print("{:<12} {:>10} {:>10} {:>10}".format('Method', 'Del AUC↓', 'Ins AUC↑', 'AOPC↑'))
    print("-" * 45)
    for m in ['sam', 'gradcam', 'random']:
        r = results[m]
        print("{:<12} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            m.upper(),
            np.mean(r['del_auc']), np.mean(r['ins_auc']),
            np.mean(r['aopc'])))

    # Save raw results
    np.savez(os.path.join(output_dir, 'faithfulness.npz'), **{
        k + '_' + mk: np.array(v)
        for k, mv in results.items() for mk, v in mv.items()})
    return results


def evaluate_localization(model, loader, device, output_dir, num_samples=50):
    """Table X: IoU@0.5, Pointing Game for SAM vs GradCAM vs Random."""
    results = {m: {'iou05': [], 'pointing': []}
               for m in ['sam', 'gradcam', 'random']}

    count = 0
    for inp, mask, meta in tqdm(loader, desc='Localization', total=min(num_samples, len(loader))):
        if count >= num_samples:
            break
        inp = inp.to(device)
        gt = mask[0, 0].numpy()  # (H, W) binary

        sam_map = compute_sam(model, inp)
        gcam_map = compute_gradcam(model, inp)
        rand_map = compute_random_baseline(inp.shape[2], inp.shape[3])

        for name, smap in [('sam', sam_map), ('gradcam', gcam_map), ('random', rand_map)]:
            results[name]['iou05'].append(iou_at_threshold(smap, gt, 0.5))
            results[name]['pointing'].append(pointing_game(smap, gt))

        count += 1

    print("\n=== Table X: Localization ===")
    print("{:<12} {:>10} {:>14}".format('Method', 'IoU@0.5↑', 'Pointing Game↑'))
    print("-" * 40)
    for m in ['sam', 'gradcam', 'random']:
        r = results[m]
        print("{:<12} {:>10.4f} {:>14.4f}".format(
            m.upper(), np.mean(r['iou05']), np.mean(r['pointing'])))

    np.savez(os.path.join(output_dir, 'localization.npz'), **{
        k + '_' + mk: np.array(v)
        for k, mv in results.items() for mk, v in mv.items()})
    return results


def visualize_sam(model, loader, device, output_dir, num_vis=8):
    """Fig 1: SAM heatmap visualization. Fig 2: Deletion/Insertion curves."""
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    count = 0
    all_del = {'sam': [], 'gradcam': []}
    all_ins = {'sam': [], 'gradcam': []}

    for inp, mask, meta in loader:
        if count >= num_vis:
            break
        inp_cuda = inp.to(device)
        img_id = meta['img_id'][0]

        sam_map = compute_sam(model, inp_cuda)
        gcam_map = compute_gradcam(model, inp_cuda)

        # Save heatmaps as colormapped images
        for name, smap in [('sam', sam_map), ('gradcam', gcam_map)]:
            heatmap = (smap * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Overlay on original image
            img_np = inp[0].permute(1, 2, 0).numpy()
            img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)
            overlay = cv2.addWeighted(img_np, 0.5, heatmap_color, 0.5, 0)
            cv2.imwrite(os.path.join(vis_dir, '{}_{}.png'.format(img_id, name)), overlay)

        # GT mask
        gt_np = (mask[0, 0].numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(vis_dir, '{}_gt.png'.format(img_id)), gt_np)

        # Deletion/insertion curves for Fig 2
        for name, smap in [('sam', sam_map), ('gradcam', gcam_map)]:
            del_s = deletion_curve(model, inp_cuda, smap, steps=50)
            ins_s = insertion_curve(model, inp_cuda, smap, steps=50)
            all_del[name].append(del_s)
            all_ins[name].append(ins_s)

        count += 1

    # Save curve data for plotting
    np.savez(os.path.join(output_dir, 'curves.npz'),
             sam_del=np.array(all_del['sam']),
             sam_ins=np.array(all_ins['sam']),
             gcam_del=np.array(all_del['gradcam']),
             gcam_ins=np.array(all_ins['gradcam']))

    # Plot average curves (Fig 2)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for name, style in [('sam', '-'), ('gradcam', '--')]:
            del_avg = np.mean(all_del[name], axis=0)
            ins_avg = np.mean(all_ins[name], axis=0)
            x = np.linspace(0, 1, len(del_avg))
            axes[0].plot(x, del_avg, style, label=name.upper())
            axes[1].plot(x, ins_avg, style, label=name.upper())

        axes[0].set_title('Deletion Curve')
        axes[0].set_xlabel('Fraction of pixels removed')
        axes[0].set_ylabel('Model confidence')
        axes[0].legend()

        axes[1].set_title('Insertion Curve')
        axes[1].set_xlabel('Fraction of pixels revealed')
        axes[1].set_ylabel('Model confidence')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig2_curves.pdf'), dpi=150)
        plt.savefig(os.path.join(output_dir, 'fig2_curves.png'), dpi=150)
        plt.close()
        print("Fig 2 saved to {}".format(output_dir))
    except Exception as e:
        print("Could not plot (matplotlib issue): {}".format(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='Path to checkpoint directory (contains config.yml + model.pth)')
    parser.add_argument('--output_dir', default='results/sam',
                        help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples for faithfulness/localization eval')
    parser.add_argument('--num_vis', type=int, default=8,
                        help='Number of samples for visualization')
    parser.add_argument('--steps', type=int, default=100,
                        help='Number of steps for deletion/insertion curves')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model from {}".format(args.checkpoint))
    model, config = load_model(args.checkpoint, args.device)

    print("Building validation loader for {}".format(config['dataset']))
    loader = get_val_loader(config)

    print("\n--- Faithfulness Evaluation (Table IX) ---")
    evaluate_faithfulness(model, loader, args.device, args.output_dir,
                          num_samples=args.num_samples, steps=args.steps)

    print("\n--- Localization Evaluation (Table X) ---")
    evaluate_localization(model, loader, args.device, args.output_dir,
                          num_samples=args.num_samples)

    print("\n--- Visualization (Fig 1 & 2) ---")
    visualize_sam(model, loader, args.device, args.output_dir,
                  num_vis=args.num_vis)

    print("\nDone! Results saved to {}".format(args.output_dir))


if __name__ == '__main__':
    main()
