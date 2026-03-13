"""Per-class evaluation for Synapse and ACDC experiments.

Loads saved model.pth checkpoints and computes per-class Dice (and HD95 for Synapse).
Outputs results to stdout and saves to eval_results.csv in each experiment directory.

Usage:
    python eval_perclass.py --experiments synapse_ukanpp_seed2981 synapse_ukanpp_seed6142 ...
    python eval_perclass.py --pattern "synapse_*"  # glob pattern
    python eval_perclass.py --pattern "acdc_*"
"""
import argparse
import os
import glob as glob_mod
import csv
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import yaml
from scipy.ndimage import distance_transform_edt

import archs


def compute_hd95(pred, gt):
    """Compute 95th percentile Hausdorff distance between binary masks."""
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 373.128  # max possible for 512x512 volume

    pred_border = pred ^ (pred & np.roll(pred, 1, axis=0) & np.roll(pred, 1, axis=1))
    gt_border = gt ^ (gt & np.roll(gt, 1, axis=0) & np.roll(gt, 1, axis=1))

    # Use distance transform for efficiency
    dt_pred = distance_transform_edt(~pred.astype(bool))
    dt_gt = distance_transform_edt(~gt.astype(bool))

    d_pred_to_gt = dt_gt[pred.astype(bool)] if pred.sum() > 0 else np.array([0])
    d_gt_to_pred = dt_pred[gt.astype(bool)] if gt.sum() > 0 else np.array([0])

    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return np.percentile(all_distances, 95)


def compute_hd95_3d(pred, gt):
    """Compute 95th percentile Hausdorff distance for 3D volumes."""
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 373.128

    dt_pred = distance_transform_edt(~pred.astype(bool))
    dt_gt = distance_transform_edt(~gt.astype(bool))

    d_pred_to_gt = dt_gt[pred.astype(bool)] if pred.sum() > 0 else np.array([0])
    d_gt_to_pred = dt_pred[gt.astype(bool)] if gt.sum() > 0 else np.array([0])

    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return np.percentile(all_distances, 95)


def load_model(config, model_path, device):
    """Load model from config and checkpoint."""
    grid_sizes = [int(g) for g in config.get('grid_sizes', '5').split(',')]
    embed_dims = [128, 160, 256]

    if config['arch'] == 'UKANPP':
        model = archs.UKANPP(
            num_classes=config['num_classes'], input_channels=3,
            deep_supervision=False, embed_dims=embed_dims,
            grid_sizes=grid_sizes, fusion_type=config.get('fusion_type', 'softmax'),
            no_kan=config.get('no_kan', False))
    else:
        model = archs.UKAN(
            num_classes=config['num_classes'], input_channels=3,
            deep_supervision=False, embed_dims=embed_dims,
            no_kan=config.get('no_kan', False))

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    return model


def eval_synapse(model, config, device, compute_hd=True):
    """Evaluate on Synapse test volumes with per-organ Dice and HD95."""
    import h5py
    data_dir = config['data_dir']
    img_size = config['img_size']
    num_classes = config['num_classes']

    test_dir = 'test_vol_h5' if os.path.isdir(
        os.path.join(data_dir, 'test_vol_h5')) else 'test_vol'
    list_file = os.path.join(data_dir, 'lists', 'test.txt')
    with open(list_file, 'r') as f:
        test_list = [l.strip() for l in f if l.strip()]

    organ_names = {1: 'Aorta', 2: 'Gallbladder', 3: 'Kidney_L', 4: 'Kidney_R',
                   5: 'Liver', 6: 'Pancreas', 7: 'Spleen', 8: 'Stomach'}

    all_dices = {c: [] for c in range(1, num_classes)}
    all_hd95s = {c: [] for c in range(1, num_classes)}

    with torch.no_grad():
        for name in test_list:
            filepath = os.path.join(data_dir, test_dir, name + '.npy.h5')
            data = h5py.File(filepath, 'r')
            image = np.array(data['image']).astype(np.float32)  # (D, H, W)
            label = np.array(data['label']).astype(np.int64)     # (D, H, W)

            pred_vol = np.zeros_like(label)
            for d in range(image.shape[0]):
                sl = image[d]
                sl_resized = cv2.resize(sl, (img_size, img_size),
                                        interpolation=cv2.INTER_LINEAR)
                inp = torch.from_numpy(
                    np.stack([sl_resized] * 3, axis=0)[np.newaxis]
                ).float().to(device)
                out = model(inp)
                pred = out.argmax(dim=1).squeeze().cpu().numpy()
                pred = cv2.resize(pred.astype(np.float32),
                                  (label.shape[2], label.shape[1]),
                                  interpolation=cv2.INTER_NEAREST).astype(np.int64)
                pred_vol[d] = pred

            for c in range(1, num_classes):
                pred_c = (pred_vol == c).astype(np.float32)
                gt_c = (label == c).astype(np.float32)
                inter = (pred_c * gt_c).sum()
                union = pred_c.sum() + gt_c.sum()
                if union == 0:
                    dice = 1.0 if inter == 0 else 0.0
                else:
                    dice = 2 * inter / union
                all_dices[c].append(dice)

                if compute_hd:
                    hd95 = compute_hd95_3d(pred_c.astype(bool), gt_c.astype(bool))
                    all_hd95s[c].append(hd95)

    results = {}
    mean_dice = 0
    mean_hd95 = 0
    for c in range(1, num_classes):
        d = np.mean(all_dices[c])
        name = organ_names.get(c, f'class_{c}')
        results[f'{name}_dice'] = d
        mean_dice += d
        if compute_hd and all_hd95s[c]:
            h = np.mean(all_hd95s[c])
            results[f'{name}_hd95'] = h
            mean_hd95 += h
    mean_dice /= (num_classes - 1)
    results['Mean_dice'] = mean_dice
    if compute_hd:
        mean_hd95 /= (num_classes - 1)
        results['Mean_hd95'] = mean_hd95
    return results


def eval_acdc(model, config, device):
    """Evaluate on ACDC valid set with per-structure per-patient Dice.

    Groups slices by patient (case_XXX), reconstructs pseudo-volume predictions,
    then computes per-class Dice at the patient level (standard ACDC protocol).
    """
    import re
    from collections import defaultdict

    data_dir = config['data_dir']
    img_size = config['img_size']
    num_classes = config['num_classes']

    structure_names = {1: 'RV', 2: 'Myo', 3: 'LV'}

    split_dir = os.path.join(data_dir, 'valid')
    sample_list = sorted([f for f in os.listdir(split_dir) if f.endswith('.npz')])

    # Group slices by patient+phase (e.g., case_019_sliceED)
    patient_slices = defaultdict(list)
    for fname in sample_list:
        # Extract patient+phase key: case_XXX_sliceXX
        m = re.match(r'(case_\d+_slice[A-Z]+)_(\d+)\.npz', fname)
        if m:
            key = m.group(1)
            slice_idx = int(m.group(2))
            patient_slices[key].append((slice_idx, fname))

    # Per-patient per-class Dice
    patient_dices = {c: [] for c in range(1, num_classes)}

    model.eval()
    with torch.no_grad():
        for patient_key in sorted(patient_slices.keys()):
            slices = sorted(patient_slices[patient_key], key=lambda x: x[0])

            all_preds = []
            all_labels = []

            for slice_idx, fname in slices:
                data = np.load(os.path.join(split_dir, fname))
                image = data['img'].astype(np.float32)
                label = data['label'].astype(np.int64)

                if image.shape[0] != img_size or image.shape[1] != img_size:
                    image = cv2.resize(image, (img_size, img_size),
                                       interpolation=cv2.INTER_LINEAR)
                    label = cv2.resize(label.astype(np.float32), (img_size, img_size),
                                       interpolation=cv2.INTER_NEAREST).astype(np.int64)

                inp = torch.from_numpy(
                    np.stack([image] * 3, axis=0)[np.newaxis]
                ).float().to(device)
                out = model(inp)
                pred = out.argmax(dim=1).squeeze().cpu().numpy()

                all_preds.append(pred)
                all_labels.append(label)

            # Stack into volume
            pred_vol = np.stack(all_preds, axis=0)   # (D, H, W)
            label_vol = np.stack(all_labels, axis=0)  # (D, H, W)

            for c in range(1, num_classes):
                pred_c = (pred_vol == c).astype(np.float32)
                gt_c = (label_vol == c).astype(np.float32)
                inter = (pred_c * gt_c).sum()
                union = pred_c.sum() + gt_c.sum()
                if union == 0:
                    dice = 1.0 if inter == 0 else 0.0
                else:
                    dice = 2 * inter / union
                patient_dices[c].append(dice)

    results = {}
    mean_dice = 0
    for c in range(1, num_classes):
        d = np.mean(patient_dices[c])
        name = structure_names.get(c, f'class_{c}')
        results[f'{name}_dice'] = d
        mean_dice += d
    mean_dice /= (num_classes - 1)
    results['Mean_dice'] = mean_dice
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', nargs='+', default=[])
    parser.add_argument('--pattern', type=str, default=None,
                        help='Glob pattern to match experiment directories')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--no_hd95', action='store_true',
                        help='Skip HD95 computation (much faster)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    experiments = list(args.experiments)
    if args.pattern:
        matched = sorted(glob_mod.glob(os.path.join(args.output_dir, args.pattern)))
        experiments.extend([os.path.basename(p) for p in matched if os.path.isdir(p)])

    if not experiments:
        print("No experiments specified. Use --experiments or --pattern.")
        return

    all_results = []

    for exp_name in experiments:
        exp_dir = os.path.join(args.output_dir, exp_name)
        config_path = os.path.join(exp_dir, 'config.yml')
        model_path = os.path.join(exp_dir, 'model.pth')

        if not os.path.exists(config_path) or not os.path.exists(model_path):
            print(f"Skipping {exp_name}: missing config or model")
            continue

        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        print(f"\n{'='*60}")
        print(f"Evaluating: {exp_name}")
        print(f"  arch={config['arch']}, dataset={config['dataset']}")
        print(f"{'='*60}")

        model = load_model(config, model_path, device)

        dataset = config['dataset']
        if dataset == 'synapse':
            results = eval_synapse(model, config, device, compute_hd=not args.no_hd95)
        elif dataset == 'acdc':
            results = eval_acdc(model, config, device)
        else:
            print(f"  Unsupported dataset: {dataset}")
            continue

        # Print results
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")

        # Save to CSV
        results['experiment'] = exp_name
        all_results.append(results)

        csv_path = os.path.join(exp_dir, 'eval_perclass.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            writer.writeheader()
            writer.writerow(results)
        print(f"  Saved to {csv_path}")

    # Print summary table
    if all_results:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        for r in all_results:
            exp = r.pop('experiment')
            metrics = ', '.join(f'{k}={v:.4f}' for k, v in r.items())
            print(f"  {exp}: {metrics}")


if __name__ == '__main__':
    main()
