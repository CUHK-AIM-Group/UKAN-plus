"""Fig. 6 (FR-KAN branch maps and spatial FFT power spectra) and Fig. 9
(spline response with lesion and normal-tissue activation ranges).

  freq    The outputs of the G = 3, 5, 10 branches of one FR-KAN layer
          (default: the second decoder block, the finest token grid) are reduced
          to channel-mean absolute activation maps, overlaid on the image (top
          row), and shown as centered log power spectra of the displayed maps
          (bottom row, shared color scale).
  spline  Panel (a) shows the image with its lesion and normal-tissue regions.
          Panel (b) plots one learned B-spline function phi(x) = sum_j c_j B_j(x)
          (G = 5, k = 3) of the first KAN layer of the G = 5 branch over its knot
          domain, with the 10th-90th percentile activation ranges of lesion and
          normal tokens shaded. Lesion tokens have > 50% lesion pixels, normal
          tokens none. Unless --in_channel/--out_channel are given, the input
          channel with the largest standardized lesion-normal median difference
          and the output with the largest response difference are used.

Usage:
    python plot_kan_figures.py freq --exp_dir outputs/usg_joint_dynamic_busi_seed2981 --image img.png
    python plot_kan_figures.py spline --exp_dir outputs/usg_joint_dynamic_busi_seed2981 \
        --image img.png --mask img_mask.png
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations import Resize
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose

from sam_eval import load_model


def load_inputs(config, image_path, mask_path=None):
    h, w = config['input_h'], config['input_w']
    raw = cv2.imread(image_path)
    x = Compose([Resize(h, w), transforms.Normalize()])(image=raw)['image'].astype('float32') / 255
    rgb = cv2.cvtColor(cv2.resize(raw, (w, h)), cv2.COLOR_BGR2RGB)
    mask = None
    if mask_path:
        mask = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), (w, h),
                          interpolation=cv2.INTER_NEAREST) > 127
    return torch.from_numpy(x.transpose(2, 0, 1))[None], rgb, mask


@torch.no_grad()
def capture_branches(model, x, block):
    """Inputs of fc1 and outputs of every branch of the FR-KAN layer in `block`."""
    layer = getattr(model, block)[0].layer
    if len(layer.branches) < 2:
        raise ValueError(f'{block} is not a multi-branch FR-KAN layer')
    rec, hooks = {}, []
    for i, branch in enumerate(layer.branches):
        hooks.append(branch['fc1'].register_forward_hook(
            lambda m, inp, out, i=i: rec.__setitem__(('in', i), inp[0])))
        hooks.append(branch['dw3'].register_forward_hook(
            lambda m, inp, out, i=i: rec.__setitem__(('out', i), out[0])))
    hooks.append(layer.register_forward_hook(lambda m, inp, out: rec.__setitem__('hw', inp[1:3])))
    model(x)
    for hook in hooks:
        hook.remove()
    return layer, rec


def plot_freq(args, model, config):
    x, rgb, _ = load_inputs(config, args.image)
    layer, rec = capture_branches(model, x.to(args.device), args.block)
    h, w = rec['hw']
    maps, spectra, grids = [], [], []
    for i, branch in enumerate(layer.branches):
        fmap = rec[('out', i)].abs().mean(-1).reshape(h, w).cpu().numpy()
        power = np.abs(np.fft.fftshift(np.fft.fft2(fmap - fmap.mean()))) ** 2
        maps.append(fmap)
        spectra.append(np.log10(power + 1e-12))
        grids.append(branch['fc1'].grid_size)
    vmin = min(s[s > -12].min() for s in spectra)
    vmax = max(s.max() for s in spectra)

    fig, axes = plt.subplots(2, len(maps), figsize=(3 * len(maps), 6))
    for col, (fmap, spec, g) in enumerate(zip(maps, spectra, grids)):
        disp = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
        axes[0, col].imshow(rgb)
        axes[0, col].imshow(cv2.resize(disp, rgb.shape[1::-1]), cmap='jet', alpha=0.5)
        axes[0, col].set_title(f'G = {g}')
        im = axes[1, col].imshow(spec, cmap='magma', vmin=vmin, vmax=vmax)
        for ax in axes[:, col]:
            ax.axis('off')
    fig.colorbar(im, ax=axes[1].tolist(), shrink=0.8, label='log10 power')
    fig.savefig(args.out, bbox_inches='tight', dpi=300)
    np.savez(os.path.splitext(args.out)[0] + '.npz', maps=np.stack(maps), log_power=np.stack(spectra),
             grid_sizes=np.array(grids))
    print(f'{args.block}: token grid {h}x{w}, branches G = {grids}; saved {args.out}')


def spline_values(kan, xs, in_ch, out_ch):
    inp = xs[:, None].expand(-1, kan.in_features).contiguous()
    bases = kan.b_splines(inp)[:, in_ch]                     # (n, G + k)
    return bases @ kan.scaled_spline_weight[out_ch, in_ch]   # (n,)


@torch.no_grad()
def plot_spline(args, model, config):
    x, rgb, mask = load_inputs(config, args.image, args.mask)
    layer, rec = capture_branches(model, x.to(args.device), args.block)
    h, w = rec['hw']
    sizes = [b['fc1'].grid_size for b in layer.branches]
    if args.grid_size not in sizes:
        raise ValueError(f'no G = {args.grid_size} branch in {args.block} (grids {sizes})')
    b = sizes.index(args.grid_size)
    kan = layer.branches[b]['fc1']
    acts = rec[('in', b)].reshape(h, w, -1)

    frac = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    lesion = acts[torch.from_numpy(frac > 0.5)]
    normal = acts[torch.from_numpy(frac == 0)]
    if len(lesion) == 0 or len(normal) == 0:
        raise ValueError('the mask yields no lesion or no normal tokens at this resolution')

    if args.in_channel is None:
        pooled = torch.sqrt((lesion.var(0) + normal.var(0)) / 2) + 1e-6
        in_ch = int(((lesion.median(0).values - normal.median(0).values).abs() / pooled).argmax())
    else:
        in_ch = args.in_channel
    if args.out_channel is None:
        med = torch.stack([lesion[:, in_ch].median(), normal[:, in_ch].median()])
        inp = med[:, None].expand(-1, kan.in_features).contiguous()
        resp = kan.b_splines(inp)[:, in_ch] @ kan.scaled_spline_weight[:, in_ch].T   # (2, out)
        out_ch = int((resp[0] - resp[1]).abs().argmax())
    else:
        out_ch = args.out_channel

    knots = kan.grid[in_ch]
    xs = torch.linspace(float(knots[0]), float(knots[-1]), 400, device=knots.device)
    phi = spline_values(kan, xs, in_ch, out_ch).cpu().numpy()
    lo, hi = args.percentiles
    ranges = {name: np.percentile(v[:, in_ch].cpu().numpy(), [lo, hi])
              for name, v in (('Normal', normal), ('Lesion', lesion))}

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9, 3.8))
    ax_a.imshow(rgb)
    ax_a.contour(mask, levels=[0.5], colors='red', linewidths=1.5)
    ax_a.set_title('(a) lesion (red contour) and normal tissue')
    ax_a.axis('off')
    ax_b.plot(xs.cpu().numpy(), phi, color='black')
    for (name, (r0, r1)), color in zip(ranges.items(), ('tab:blue', 'tab:red')):
        ax_b.axvspan(r0, r1, color=color, alpha=0.25, label=f'{name} ({lo}-{hi}th pct.)')
    ax_b.set_xlabel('feature activation')
    ax_b.set_ylabel(r'$\phi(x)$')
    ax_b.set_title(f'(b) B-spline (G = {args.grid_size}, k = {kan.spline_order})')
    ax_b.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches='tight', dpi=300)
    print(f'{args.block} branch G={args.grid_size} fc1: input channel {in_ch}, output {out_ch}; '
          f'normal {ranges["Normal"].round(3)}, lesion {ranges["Lesion"].round(3)} '
          f'({len(normal)} / {len(lesion)} tokens on a {h}x{w} grid); saved {args.out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='figure', required=True)
    for name in ('freq', 'spline'):
        p = sub.add_parser(name)
        p.add_argument('--exp_dir', required=True, help='train.py or usg_train.py run of U-KAN++')
        p.add_argument('--image', required=True)
        p.add_argument('--block', default='dblock2', choices=['block1', 'block2', 'dblock1', 'dblock2'])
        p.add_argument('--out', default=f'fig_{name}.pdf')
        p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    spline = sub.choices['spline']
    spline.add_argument('--mask', required=True)
    spline.add_argument('--grid_size', type=int, default=5)
    spline.add_argument('--in_channel', type=int, default=None)
    spline.add_argument('--out_channel', type=int, default=None)
    spline.add_argument('--percentiles', type=int, nargs=2, default=[10, 90])
    args = parser.parse_args()

    model, config = load_model(args.exp_dir, args.device, require_stag=False)
    (plot_freq if args.figure == 'freq' else plot_spline)(args, model, config)


if __name__ == '__main__':
    main()
