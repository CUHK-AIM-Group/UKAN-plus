"""Params and FLOPs of the U-KAN++ segmentation path (Table I).

FLOPs are counted with fvcore over the full forward pass from one 3-channel input
(512x512 by default) to the output mask: the convolutions, the base and spline
products and the B-spline basis evaluation of all three FR-KAN branches, the
routing and fusion, and the decoder with STAG gating. As in fvcore, one
multiply-accumulate counts as one FLOP, and normalization and interpolation use
fvcore's counts. Elementwise operations count one FLOP per output element and
reductions one FLOP per input element. The generation decoder of the joint model
is inactive during segmentation and is not counted.

Usage:
    python profile_flops.py                 # U-KAN++ (FR-KAN + STAG)
    python profile_flops.py --exp_dir outputs/usg_joint_dynamic_busi_seed2981
"""
import argparse
import os

import torch
import yaml
from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import elementwise_flop_counter

import archs
from usg_train import EMBED_DIMS

# fvcore leaves these ops uncounted by default. They carry the B-spline basis
# evaluation, the activations, the routing softmax and the STAG gate.
ELEMENTWISE_OPS = ['aten::add', 'aten::sub', 'aten::mul', 'aten::div', 'aten::abs',
                   'aten::ge', 'aten::lt', 'aten::__and__', 'aten::relu', 'aten::relu_',
                   'aten::silu', 'aten::sigmoid', 'aten::softmax']
REDUCTION_OPS = ['aten::mean', 'aten::amax', 'aten::amin', 'aten::max_pool2d']


def ukanpp_from_config(config):
    grid_sizes = [int(g) for g in str(config.get('grid_sizes', '3,5,10')).split(',')]
    kwargs = dict(grid_sizes=grid_sizes, fusion_type=config.get('fusion_type', 'softmax'),
                  use_stag=config.get('use_stag', False), stag_stages=config.get('stag_stages', 'dec'),
                  stag_alpha=config.get('stag_alpha', 0.5), stag_norm=config.get('stag_norm', 'minmax'))
    if 'input_list' in config:  # train.py run
        kwargs.update(embed_dims=config['input_list'], no_kan=config.get('no_kan', False),
                      ms_stages=config.get('ms_stages', 'all'),
                      cross_branch_attn=config.get('cross_branch_attn', False))
    else:  # usg_train.py run (segmentation path of the joint model)
        kwargs.update(embed_dims=EMBED_DIMS)
    return archs.UKANPP(num_classes=1, input_channels=3, deep_supervision=False, **kwargs)


def profile(name, model, size):
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    counter = FlopCountAnalysis(model, torch.randn(1, 3, size, size))
    for op in ELEMENTWISE_OPS:
        counter.set_op_handle(op, elementwise_flop_counter(0, 1))
    for op in REDUCTION_OPS:
        counter.set_op_handle(op, elementwise_flop_counter(1, 0))
    counter.unsupported_ops_warnings(False)
    counter.uncalled_modules_warnings(False)
    print(f'{name:28s} params {params:7.2f} M   FLOPs {counter.total() / 1e9:8.2f} G')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--exp_dir', default=None, help='train.py or usg_train.py run (config.yml)')
    parser.add_argument('--size', type=int, default=512)
    args = parser.parse_args()

    if args.exp_dir:
        with open(os.path.join(args.exp_dir, 'config.yml')) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        profile(os.path.basename(args.exp_dir.rstrip('/')), ukanpp_from_config(config), args.size)
    else:
        full = {'grid_sizes': '3,5,10', 'use_stag': True, 'stag_stages': 'dec', 'stag_alpha': 0.5}
        profile('U-KAN++ (FR-KAN + STAG)', ukanpp_from_config(full), args.size)


if __name__ == '__main__':
    main()
