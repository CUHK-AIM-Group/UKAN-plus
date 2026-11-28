"""Table X: 3D segmentation latency.

Protocol: one GPU, FP16 (autocast), batch size 1; 5 untimed warm-up iterations
followed by 10 timed repetitions, each enclosed by torch.cuda.synchronize().
Two timing scopes:
  * fixed-size forward pass of a synthetic 96^3 input;
  * BraTS ROI inference: sliding-window inference over one preprocessed BraTS
    volume with 96^3 windows and 50% overlap, including window aggregation and
    excluding loading, preprocessing and postprocessing.
Latency is reported as mean +- SD (ms) and relative to U-KAN in the same scope.

Models (4 input modalities, 4 output classes, random weights):
  ukan     single-scale U-KAN (UKANPP3D with grid 5, STAG off)
  ukanpp   U-KAN++ (grids 3/5/10, STAG on; USG is not executed at inference)
  widened  widened single-scale KAN control with the 352-channel bottleneck of
           Table IX; embed dims 64/170/352 match the U-KAN++ parameter count
           (16.65M vs 16.68M); STAG and USG off
  unetrpp  UNETR++ (official implementation, BraTS configuration; baselines.py)
  unet3d   3D U-Net (Cicek et al.)

Usage:
    python benchmark_3d_latency.py --brats_dir /data/brats2021
    python benchmark_3d_latency.py --roi_size 128,160,128   # synthetic ROI instead of a BraTS case
"""
import argparse
import csv
import os
import time
from contextlib import nullcontext

import numpy as np
import torch

import baselines
from archs_3d import UKANPP3D
from dataset_3d import get_dataset
from sliding_window import sliding_window_inference

LABELS = {'ukan': 'U-KAN', 'ukanpp': 'U-KAN++', 'widened': 'Widened single-scale KAN control',
          'unetrpp': 'UNETR++', 'unet3d': '3D U-Net'}


def build(name, window, widened_dims):
    common = dict(num_classes=4, input_channels=4)
    if name == 'ukan':
        return UKANPP3D(**common, embed_dims=[64, 128, 256], grid_sizes=[5])
    if name == 'ukanpp':
        return UKANPP3D(**common, embed_dims=[64, 128, 256], grid_sizes=[3, 5, 10],
                        use_stag=True, stag_stages='dec', stag_alpha=0.5)
    if name == 'widened':
        return UKANPP3D(**common, embed_dims=widened_dims, grid_sizes=[5])
    if name == 'unetrpp':
        return baselines.unetr_pp(in_channels=4, num_classes=4, window=window)
    if name == 'unet3d':
        return baselines.UNet3D(in_channels=4, num_classes=4)
    raise ValueError(name)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def time_ms(fn, device, warmup, repeats):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.mean(times)), float(np.std(times, ddof=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', default='ukan,ukanpp,widened,unetrpp,unet3d')
    parser.add_argument('--window', type=int, default=96)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--precision', default='fp16', choices=['fp16', 'fp32'])
    parser.add_argument('--widened_dims', default='64,170,352')
    parser.add_argument('--brats_dir', default=None, help='BraTS root (dataset_3d.py layout)')
    parser.add_argument('--brats_split', default='val')
    parser.add_argument('--case_index', type=int, default=0)
    parser.add_argument('--roi_size', default=None, help='synthetic ROI "D,H,W" when no BraTS case is used')
    parser.add_argument('--out_csv', default='latency_3d.csv')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    window = (args.window,) * 3
    widened_dims = [int(x) for x in args.widened_dims.split(',')]
    amp = (lambda: torch.autocast(device_type=device.type, dtype=torch.float16)) \
        if args.precision == 'fp16' else nullcontext

    roi, roi_name = None, None
    if args.brats_dir:
        data = get_dataset('brats2021', args.brats_dir, split=args.brats_split, augment=False, full_volume=True)
        image, _, roi_name = data[args.case_index]
        roi = image[None].to(device)
    elif args.roi_size:
        roi = torch.randn(1, 4, *[int(x) for x in args.roi_size.split(',')], device=device)
        roi_name = 'synthetic'
    if roi is not None:
        print(f'ROI: {roi_name}, shape {tuple(roi.shape[2:])}')
    patch = torch.randn(1, 4, *window, device=device)

    rows = []
    for name in args.models.split(','):
        try:
            model = build(name, window, widened_dims).to(device).eval()
        except (RuntimeError, ImportError) as err:
            print(f'{LABELS[name]}: skipped ({err})')
            continue
        params = sum(p.numel() for p in model.parameters()) / 1e6

        def forward_patch():
            with torch.no_grad(), amp():
                model(patch)

        def forward_roi():
            with torch.no_grad(), amp():
                sliding_window_inference(roi, model, window, args.overlap, sw_batch_size=1)

        scopes = [('Synthetic input', f'{args.window}^3 patch forward', forward_patch)]
        if roi is not None:
            scopes.append(('BraTS' if args.brats_dir else 'Synthetic ROI',
                           f'{args.window}^3 ROI sliding window', forward_roi))
        for dataset, scope, fn in scopes:
            mean, sd = time_ms(fn, device, args.warmup, args.repeats)
            rows.append({'dataset': dataset, 'method': LABELS[name], 'scope': scope,
                         'mean_ms': mean, 'sd_ms': sd, 'params_m': params})
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    for row in rows:
        ref = [r['mean_ms'] for r in rows if r['scope'] == row['scope'] and r['method'] == 'U-KAN']
        row['relative'] = row['mean_ms'] / ref[0] if ref else float('nan')
        print(f"{row['dataset']:15s} {row['method']:34s} {row['scope']:26s} "
              f"{row['mean_ms']:9.1f} +- {row['sd_ms']:6.1f} ms  {row['relative']:.2f}x  ({row['params_m']:.2f}M)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {args.out_csv}')


if __name__ == '__main__':
    main()
