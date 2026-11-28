"""Baseline networks for the comparison tables (Tables I and X).

The smp-based CNNs (U-Net, Att-UNet, U-Net++) and the 3D U-Net are built
here. TransUNet, U-NeXt, Rolling-UNet, U-Mamba and UNETR++ are imported from
their official repositories, which are not vendored. Clone each repository
and point the environment variable at it:

  TRANSUNET_DIR     https://github.com/Beckschen/TransUNet
                    (optional TRANSUNET_WEIGHTS: R50+ViT-B_16.npz initialization)
  UNEXT_DIR         https://github.com/jeya-maria-jose/UNeXt-pytorch
  ROLLING_UNET_DIR  https://github.com/Jiaoyang45/Rolling-Unet
  UMAMBA_DIR        https://github.com/bowang-lab/U-Mamba
                    (requires `pip install -e umamba` and mamba-ssm, CUDA only)
  UNETR_PP_DIR      https://github.com/Amshaker/unetr_plus_plus
"""
import importlib
import importlib.util
import os
import sys
import types
import warnings
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

BASELINES_2D = ('unet', 'attunet', 'unetpp', 'transunet', 'unext', 'rollingunet', 'umamba',
                'transunet_smp')


def _repo(env):
    path = os.environ.get(env)
    if not path or not os.path.isdir(path):
        raise RuntimeError(f'{env} must point to a clone of the official repository (see baselines.py)')
    return os.path.abspath(path)


@contextmanager
def _official_imports(repo_dir, shadowed=('utils', 'archs')):
    """Import from repo_dir while hiding this repository's modules of the same names."""
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.split('.')[0] in shadowed}
    sys.path.insert(0, repo_dir)
    try:
        yield
    finally:
        sys.path.remove(repo_dir)
        for k in [k for k in sys.modules if k.split('.')[0] in shadowed]:
            del sys.modules[k]
        sys.modules.update(saved)


def _load_file(repo_dir, filename, module_name):
    with _official_imports(repo_dir):
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(repo_dir, filename))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- 2D (Table I)

def smp_baseline(arch, pretrained=True, num_classes=1):
    import segmentation_models_pytorch as smp
    weights = 'imagenet' if pretrained else None
    if arch == 'unetpp':
        return smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights=weights,
                                in_channels=3, classes=num_classes)
    if arch == 'transunet_smp':
        # Earlier stand-in (U-Net with a MiT-B2 encoder), kept to load old checkpoints
        return smp.Unet(encoder_name='mit_b2', encoder_weights=weights, in_channels=3,
                        classes=num_classes)
    return smp.Unet(encoder_name='resnet34', encoder_weights=weights, in_channels=3,
                    classes=num_classes, decoder_attention_type='scse' if arch == 'attunet' else None)


def transunet(img_size, num_classes=1, pretrained=True):
    """TransUNet R50-ViT-B_16 with three skip connections (official configuration)."""
    with _official_imports(_repo('TRANSUNET_DIR'), shadowed=('networks',)):
        vit = importlib.import_module('networks.vit_seg_modeling')
    config = vit.CONFIGS['R50-ViT-B_16']
    config.n_classes, config.n_skip = num_classes, 3
    config.patches.grid = (img_size // 16, img_size // 16)
    model = vit.VisionTransformer(config, img_size=img_size, num_classes=num_classes)
    if pretrained:
        weights = os.environ.get('TRANSUNET_WEIGHTS')
        if weights:
            model.load_from(weights=np.load(weights))
        else:
            warnings.warn('TRANSUNET_WEIGHTS is not set; TransUNet starts from random initialization')
    return model


def unext(img_size, num_classes=1):
    try:
        import mmcv.cnn  # noqa: F401
    except ImportError:
        # UNeXt's archs.py imports mmcv.cnn.ConvModule without using it
        cnn = types.ModuleType('mmcv.cnn')
        cnn.ConvModule = None
        mmcv = types.ModuleType('mmcv')
        mmcv.cnn = cnn
        sys.modules.setdefault('mmcv', mmcv)
        sys.modules.setdefault('mmcv.cnn', cnn)
    archs = _load_file(_repo('UNEXT_DIR'), 'archs.py', 'official_unext_archs')
    return archs.UNext(num_classes=num_classes, input_channels=3, deep_supervision=False,
                       img_size=img_size)


def rolling_unet(img_size, num_classes=1, variant='S'):
    archs = _load_file(_repo('ROLLING_UNET_DIR'), 'archs.py', 'official_rolling_unet_archs')
    return getattr(archs, f'Rolling_Unet_{variant}')(
        num_classes=num_classes, input_channels=3, deep_supervision=False, img_size=img_size)


def umamba_bot_2d(img_size, num_classes=1):
    """U-Mamba-Bot 2D with the nnU-Net 2D topology for a square img_size patch:
    stages down to 4x4 (at most 7), 32 base features doubling up to 512,
    3x3 kernels, two blocks/convolutions per stage, instance norm, LeakyReLU."""
    repo = _repo('UMAMBA_DIR')
    if importlib.util.find_spec('nnunetv2') is None:
        sys.path.insert(0, os.path.join(repo, 'umamba'))
    from nnunetv2.nets.UMambaBot_2d import UMambaBot
    n_stages = min(int(np.log2(img_size // 4)) + 1, 7)
    return UMambaBot(
        input_channels=3, n_stages=n_stages,
        features_per_stage=[min(32 * 2 ** i, 512) for i in range(n_stages)],
        conv_op=nn.Conv2d, kernel_sizes=[[3, 3]] * n_stages,
        strides=[[1, 1]] + [[2, 2]] * (n_stages - 1),
        n_conv_per_stage=[2] * n_stages, num_classes=num_classes,
        n_conv_per_stage_decoder=[2] * (n_stages - 1), conv_bias=True,
        norm_op=nn.InstanceNorm2d, norm_op_kwargs={'eps': 1e-5, 'affine': True},
        nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=False)


def build_baseline_2d(arch, img_size=256, pretrained=True, num_classes=1):
    if arch in ('unet', 'attunet', 'unetpp', 'transunet_smp'):
        return smp_baseline(arch, pretrained, num_classes)
    if arch == 'transunet':
        return transunet(img_size, num_classes, pretrained)
    if arch == 'unext':
        return unext(img_size, num_classes)
    if arch == 'rollingunet':
        return rolling_unet(img_size, num_classes)
    if arch == 'umamba':
        return umamba_bot_2d(img_size, num_classes)
    raise ValueError(f'unknown 2D baseline: {arch}')


# ---------------------------------------------------------------- 3D (Table X)

class UNet3D(nn.Module):
    """3D U-Net (Cicek et al., 2016): per level two 3x3x3 conv-BN-ReLU with
    32-64, 64-128, 128-256 and 256-512 channels, 2x2x2 max pooling and
    transposed-convolution upsampling; convolutions are padded."""

    def __init__(self, in_channels=4, num_classes=4):
        super().__init__()

        def block(cin, cmid, cout):
            return nn.Sequential(
                nn.Conv3d(cin, cmid, 3, padding=1), nn.BatchNorm3d(cmid), nn.ReLU(inplace=True),
                nn.Conv3d(cmid, cout, 3, padding=1), nn.BatchNorm3d(cout), nn.ReLU(inplace=True))

        self.encoders = nn.ModuleList([block(in_channels, 32, 64), block(64, 64, 128),
                                       block(128, 128, 256)])
        self.bottom = block(256, 256, 512)
        self.ups = nn.ModuleList([nn.ConvTranspose3d(c, c, 2, stride=2) for c in (512, 256, 128)])
        self.decoders = nn.ModuleList([block(512 + 256, 256, 256), block(256 + 128, 128, 128),
                                       block(128 + 64, 64, 64)])
        self.pool = nn.MaxPool3d(2)
        self.head = nn.Conv3d(64, num_classes, 1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottom(x)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = decoder(torch.cat([up(x), skip], 1))
        return self.head(x)


def unetr_pp(in_channels=4, num_classes=4, window=(96, 96, 96)):
    """UNETR++ in its BraTS (tumor) configuration. The official model fixes its
    token counts for 128^3 inputs; they are recomputed here for `window`
    (each side divisible by 32)."""
    with _official_imports(_repo('UNETR_PP_DIR'), shadowed=('unetr_pp',)):
        tumor = importlib.import_module('unetr_pp.network_architecture.tumor.unetr_pp_tumor')
    grids = [tuple(s // f for s in window) for f in (4, 8, 16, 32)]
    tokens = [int(np.prod(g)) for g in grids]
    out_sizes = {8 ** 3: tokens[2], 16 ** 3: tokens[1], 32 ** 3: tokens[0], 128 ** 3: int(np.prod(window))}
    encoder_cls, up_cls = tumor.UnetrPPEncoder, tumor.UnetrUpBlock

    class SizedEncoder(encoder_cls):
        def __init__(self, **kwargs):
            super().__init__(input_size=tokens, in_channels=in_channels, **kwargs)

    class SizedUpBlock(up_cls):
        def __init__(self, out_size=0, **kwargs):
            super().__init__(out_size=out_sizes.get(out_size, out_size), **kwargs)

    tumor.UnetrPPEncoder, tumor.UnetrUpBlock = SizedEncoder, SizedUpBlock
    try:
        model = tumor.UNETR_PP(in_channels=in_channels, out_channels=num_classes, feature_size=16,
                               num_heads=4, depths=[3, 3, 3, 3], dims=[32, 64, 128, 256], do_ds=False)
    finally:
        tumor.UnetrPPEncoder, tumor.UnetrUpBlock = encoder_cls, up_cls
    model.feat_size = grids[3]
    return model
