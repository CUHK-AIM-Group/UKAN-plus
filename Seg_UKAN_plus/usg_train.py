"""
Unified Seg-Gen (USG) Training Script — Tables I, V and VI (sampling: usg_sample.py).

Shared MS-KAN encoder with:
  - Segmentation decoder head (from UKANPP)
  - Generation decoder head: DDPM noise predictor on the shared encoder
    features, with FR-KAN blocks, skip connections and a timestep embedding
  - Dynamic loss balancing: lambda(t) = lambda_0 * exp(-gamma * L_seg / L_gen)

In each iteration the segmentation branch receives the image and the generation
branch receives a noise-corrupted 64x64 version of the same image (--gen_view).

Modes:
  - seg_only:  standard segmentation (reuse UKANPP, same as train.py)
  - gen_only:  generation only; the segmentation decoder is not executed
  - joint:     unified framework with dynamic or fixed lambda

Usage:
    python usg_train.py --name usg_joint_dynamic_busi_seed2981 --dataset busi \
        --mode joint --lambda_type dynamic --lambda_0 1.0 --gamma 0.5 \
        --grid_sizes 3,5,10 --use_stag --dataseed 2981
"""
import argparse
import os
import random
import csv
import yaml
import math
from contextlib import contextmanager, nullcontext
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm

from albumentations.augmentations import transforms
from albumentations.augmentations import geometric
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize

import archs
from dataset import Dataset, split_ids
from losses import BCEDiceLoss
from metrics import iou_score
from utils import AverageMeter


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


EMBED_DIMS = [128, 160, 256]

# dataset.py reads BGR images with cv2, applies albumentations Normalize with
# these constants and then divides by 255.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)


def to_ddpm_space(inp, size):
    """Dataset image tensor -> x_0 in [-1, 1] at size x size (BGR channel order)."""
    mean = inp.new_tensor(NORM_MEAN).view(1, 3, 1, 1)
    std = inp.new_tensor(NORM_STD).view(1, 3, 1, 1)
    img = (inp * 255 * std + mean).clamp(0, 1)
    img = F.interpolate(img, size=(size, size), mode='bilinear',
                        align_corners=False, antialias=True)
    return img * 2 - 1


def to_model_input(x):
    """Image in [-1, 1] -> the normalized representation produced by dataset.py."""
    mean = x.new_tensor(NORM_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(NORM_STD).view(1, 3, 1, 1)
    return ((x + 1) / 2 - mean) / std / 255


# images_64 preparation of Diffusion_UKAN_plus/tools, whose folders are the FID/IS
# reference sets: center crop (BUSI, CVC) or random crop (GlaS), then resize.
UKAN_GEN_VIEWS = {'busi': ('center', 400), 'cvc': ('center', 288), 'glas': ('random', 64)}


def ukan_gen_view(img, view, size):
    """BGR uint8 image -> x_0 in [-1, 1] following UKAN_GEN_VIEWS, with random flips."""
    if view is not None:
        kind, crop = view
        H, W = img.shape[:2]
        if kind == 'center':
            y, x = H // 2 - crop // 2, W // 2 - crop // 2
        else:
            y, x = random.randint(0, max(H - crop, 0)), random.randint(0, max(W - crop, 0))
        img = img[max(y, 0):y + crop, max(x, 0):x + crop]
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if random.random() < 0.5:
        img = img[:, ::-1]
    if random.random() < 0.5:
        img = img[::-1]
    img = np.ascontiguousarray(img).transpose(2, 0, 1)
    return torch.from_numpy(img).float() / 127.5 - 1


class USGDataset(Dataset):
    """Segmentation sample plus x_0 of the generation branch for the same image."""
    def __init__(self, gen_view, gen_size, **kwargs):
        super().__init__(**kwargs)
        self.gen_view = gen_view
        self.gen_size = gen_size

    def __getitem__(self, idx):
        img, mask, meta = super().__getitem__(idx)
        if self.gen_view == 'resize':
            x_0 = to_ddpm_space(torch.from_numpy(img)[None], self.gen_size)[0]
        else:
            raw = cv2.imread(os.path.join(self.img_dir, meta['img_id'] + self.img_ext))
            x_0 = ukan_gen_view(raw, self.gen_view, self.gen_size)
        return img, mask, x_0, meta


@contextmanager
def frozen_bn_stats(module):
    """BatchNorm layers normalize with, and do not update, their running statistics."""
    bns = [m for m in module.modules()
           if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training]
    for m in bns:
        m.eval()
    try:
        yield
    finally:
        for m in bns:
            m.train()


# ---- Generation Decoder ----

class GenDecoder(nn.Module):
    """DDPM noise predictor built on the shared encoder features.

    Mirrors the segmentation decoder: FR-KAN blocks at the two token stages and
    convolutional upsampling with the encoder skip connections. A timestep
    embedding is added at the bottleneck and at every decoder stage. The encoder
    downsamples before its first skip, so the noisy image itself is concatenated
    at full resolution.
    """
    def __init__(self, embed_dims=EMBED_DIMS, grid_sizes=[3, 5, 10],
                 fusion_type='softmax', img_channels=3, T=1000):
        super().__init__()
        e0, e1, e2 = embed_dims
        tdim = e2

        self.register_buffer('time_freqs', self._sinusoidal_embedding(T, tdim))
        self.time_embed = nn.Sequential(
            nn.Linear(tdim, tdim * 4),
            nn.SiLU(),
            nn.Linear(tdim * 4, tdim),
        )
        # bottleneck, dec1, dec2, dec3, dec4, full resolution
        self.time_proj = nn.ModuleList(
            [nn.Linear(tdim, w) for w in (e2, e1, e0, e0 // 4, e0 // 8, e0 // 8)])

        def fr_kan(dim):
            return nn.ModuleList([archs.MSKANBlock(
                dim=dim, grid_sizes=grid_sizes, fusion_type=fusion_type)])

        self.decoder1 = archs.D_ConvLayer(e2, e1)
        self.dblock1 = fr_kan(e1)
        self.dnorm3 = nn.LayerNorm(e1)
        self.decoder2 = archs.D_ConvLayer(e1, e0)
        self.dblock2 = fr_kan(e0)
        self.dnorm4 = nn.LayerNorm(e0)
        self.decoder3 = archs.D_ConvLayer(e0, e0 // 4)
        self.decoder4 = archs.D_ConvLayer(e0 // 4, e0 // 8)
        self.decoder5 = archs.D_ConvLayer(e0 // 8, e0 // 8)
        self.head = archs.D_ConvLayer(e0 // 8 + img_channels, e0 // 8)
        self.final = nn.Conv2d(e0 // 8, img_channels, kernel_size=1)

    def _sinusoidal_embedding(self, T, dim):
        pos = torch.arange(T).float()
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim).float() * -emb)
        emb = pos[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (T, dim)

    def _kan_stage(self, out, blocks, norm):
        B, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in blocks:
            out = blk(out, H, W)
        out = norm(out)
        return out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

    def forward(self, bottleneck, skips, t, x_t):
        """
        Args:
            bottleneck: (B, C, h, w) bottleneck features of the noisy image x_t
            skips: encoder skip connections (t1, t2, t3, t4, ...)
            t: (B,) integer timesteps
            x_t: (B, 3, H, W) noisy image in [-1, 1] space

        Returns:
            noise_pred: (B, 3, H, W) predicted noise
        """
        t1, t2, t3, t4 = skips[:4]
        emb = self.time_embed(self.time_freqs[t])
        tb, td1, td2, td3, td4, td5 = [p(emb)[:, :, None, None] for p in self.time_proj]

        out = bottleneck + tb
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))
        out = self._kan_stage(out + t4 + td1, self.dblock1, self.dnorm3)
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        out = self._kan_stage(out + t3 + td2, self.dblock2, self.dnorm4)
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        out = out + t2 + td3
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        out = out + t1 + td4
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))
        out = self.head(torch.cat([out + td5, x_t], dim=1))
        return self.final(out)


# ---- DDPM Utilities ----

def linear_beta_schedule(T, beta_1=1e-4, beta_T=0.02):
    return torch.linspace(beta_1, beta_T, T)


class SimpleDDPM:
    """Minimal DDPM forward process for USG training."""
    def __init__(self, T=1000, device='cuda'):
        self.T = T
        betas = linear_beta_schedule(T).to(device)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_bar = torch.sqrt(alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * noise"""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alphas_bar[t][:, None, None, None]
        sqrt_omab = self.sqrt_one_minus_alphas_bar[t][:, None, None, None]
        return sqrt_ab * x_0 + sqrt_omab * noise, noise


# ---- USG Model Wrapper ----

class USGModel(nn.Module):
    """Unified Seg-Gen model: shared encoder + seg decoder + gen decoder."""
    def __init__(self, seg_model, gen_decoder):
        super().__init__()
        self.seg_model = seg_model     # UKANPP (full model)
        self.gen_decoder = gen_decoder  # GenDecoder

    def encode(self, x):
        """Run shared encoder, return bottleneck features and skip connections."""
        m = self.seg_model
        B = x.shape[0]

        out = F.relu(F.max_pool2d(m.encoder1(x), 2, 2))
        t1 = out
        out = F.relu(F.max_pool2d(m.encoder2(out), 2, 2))
        t2 = out
        out = F.relu(F.max_pool2d(m.encoder3(out), 2, 2))
        t3 = out

        out, H, W = m.patch_embed3(out)
        for blk in m.block1:
            out = blk(out, H, W)
        out = m.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        out, H, W = m.patch_embed4(out)
        for blk in m.block2:
            out = blk(out, H, W)
        out = m.norm4(out)
        bottleneck = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return bottleneck, (t1, t2, t3, t4, H, W)

    def decode_seg(self, bottleneck, skips):
        """Run segmentation decoder."""
        m = self.seg_model
        t1, t2, t3, t4, H, W = skips
        B = bottleneck.shape[0]
        m.last_stag_maps = {}

        out = F.relu(F.interpolate(m.decoder1(bottleneck), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t4)
        _, _, H2, W2 = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in m.dblock1:
            out = blk(out, H2, W2)
        out = m.dnorm3(out)
        out = out.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
        if m.stag_dec1 is not None:
            stag_map = m._collect_stag_map(m.dblock1)
            out, gate = m.stag_dec1(out, stag_map, H2, W2, detach=m.stag_detach)
            m.last_stag_maps['dec1'] = gate

        out = F.relu(F.interpolate(m.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t3)
        _, _, H3, W3 = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in m.dblock2:
            out = blk(out, H3, W3)
        out = m.dnorm4(out)
        out = out.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
        if m.stag_dec2 is not None:
            stag_map = m._collect_stag_map(m.dblock2)
            out, gate = m.stag_dec2(out, stag_map, H3, W3, detach=m.stag_detach)
            m.last_stag_maps['dec2'] = gate

        out = F.relu(F.interpolate(m.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(m.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(m.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        return m.final(out)

    def decode_gen(self, bottleneck, skips, t, x_t):
        """Run generation decoder."""
        return self.gen_decoder(bottleneck, skips, t, x_t)

    def forward(self, x, mode='seg', t=None):
        """mode='seg': x is a dataset image tensor; mode='gen': x is x_t in [-1, 1]."""
        if mode == 'seg':
            return self.decode_seg(*self.encode(x))
        elif mode == 'gen':
            bottleneck, skips = self.encode(to_model_input(x))
            return self.decode_gen(bottleneck, skips, t, x)
        raise ValueError(f'Unknown mode: {mode}')


def seg_metrics(model, loader, device):
    """Mean IoU and Dice of the segmentation path over a loader."""
    iou_meter = AverageMeter()
    dice_meter = AverageMeter()
    model.eval()
    with torch.no_grad():
        for inp, mask, _ in loader:
            inp = inp.to(device)
            mask = mask.to(device)
            seg_out = model(inp, mode='seg')
            iou, dice, _ = iou_score(seg_out, mask)
            iou_meter.update(iou, inp.size(0))
            dice_meter.update(dice, inp.size(0))
    return iou_meter.avg, dice_meter.avg


def build_model(config):
    """USG model from a training config (args of this script)."""
    grid_sizes = [int(g) for g in str(config['grid_sizes']).split(',')]
    seg_model = archs.UKANPP(
        num_classes=1, input_channels=3, deep_supervision=False,
        embed_dims=EMBED_DIMS, grid_sizes=grid_sizes,
        fusion_type=config['fusion_type'],
        use_stag=config['use_stag'], stag_stages=config['stag_stages'],
        stag_alpha=config['stag_alpha'], stag_norm=config['stag_norm'])
    gen_decoder = GenDecoder(embed_dims=EMBED_DIMS, grid_sizes=grid_sizes,
                             fusion_type=config['fusion_type'], T=config['T'])
    return USGModel(seg_model, gen_decoder)


def parse_args():
    parser = argparse.ArgumentParser()
    # Basic
    parser.add_argument('--name', required=True)
    parser.add_argument('--dataset', default='busi')
    parser.add_argument('--data_dir', default='../data')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--input_h', type=int, default=256)
    parser.add_argument('--input_w', type=int, default=256)
    parser.add_argument('--dataseed', type=int, default=2981)
    parser.add_argument('--num_workers', type=int, default=4)

    # Architecture
    parser.add_argument('--grid_sizes', type=str, default='3,5,10')
    parser.add_argument('--fusion_type', default='softmax')
    parser.add_argument('--use_stag', action='store_true',
                        help='enable STAG in the segmentation decoder')
    parser.add_argument('--stag_stages', default='dec', type=str,
                        choices=['all', 'dec', 'dec1', 'dec2', 'dec12', 'none'])
    parser.add_argument('--stag_alpha', default=0.5, type=float)
    parser.add_argument('--stag_norm', default='minmax', type=str,
                        choices=['minmax', 'sigmoid', 'none'])

    # USG specific
    parser.add_argument('--mode', default='joint',
                        choices=['seg_only', 'gen_only', 'joint'])
    parser.add_argument('--lambda_type', default='dynamic',
                        choices=['fixed', 'dynamic'])
    parser.add_argument('--lambda_val', type=float, default=0.5,
                        help='Fixed lambda for gen loss')
    parser.add_argument('--lambda_0', type=float, default=1.0,
                        help='Initial lambda for dynamic schedule')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Gamma for dynamic lambda: lambda_0 * exp(-gamma * L_seg/L_gen)')
    parser.add_argument('--T', type=int, default=1000,
                        help='Diffusion timesteps')
    parser.add_argument('--gen_size', type=int, default=64,
                        help='Image size of the generation branch')
    parser.add_argument('--gen_view', default='ukan', choices=['ukan', 'resize'],
                        help='ukan: images_64 preparation of Diffusion_UKAN_plus/tools for BUSI, '
                             'GlaS and CVC (whole image for other datasets); resize: the '
                             'segmentation input resized to gen_size')

    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--kan_lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--kan_weight_decay', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-5)

    return parser.parse_args()


def main():
    args = parse_args()
    seed_torch(args.dataseed)

    config = vars(args)

    output_path = os.path.join(args.output_dir, args.name)
    os.makedirs(output_path, exist_ok=True)

    # Save config
    with open(os.path.join(output_path, 'config.yml'), 'w') as f:
        yaml.dump(config, f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(config).to(device)
    ddpm = SimpleDDPM(T=args.T, device=device)

    # Optimizer with dual learning rates
    kan_param_ids = set()
    for name, module in model.named_modules():
        if 'KANLinear' in type(module).__name__:
            for p in module.parameters():
                kan_param_ids.add(id(p))

    kan_params = [p for p in model.parameters() if id(p) in kan_param_ids]
    other_params = [p for p in model.parameters() if id(p) not in kan_param_ids]

    optimizer = torch.optim.Adam([
        {'params': kan_params, 'lr': args.kan_lr, 'weight_decay': args.kan_weight_decay},
        {'params': other_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
    ])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    seg_criterion = BCEDiceLoss()

    # Dataset
    dataset_name = args.dataset
    img_ext = '.png'
    mask_ext_map = {'busi': '_mask.png'}
    mask_ext = mask_ext_map.get(dataset_name, '.png')

    img_ids = sorted(glob(os.path.join(args.data_dir, dataset_name, 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    train_ids = split_ids(img_ids, args.data_dir, dataset_name, 'train', args.dataseed)
    val_ids = split_ids(img_ids, args.data_dir, dataset_name, 'val', args.dataseed)

    train_transform = Compose([
        RandomRotate90(),
        geometric.transforms.HorizontalFlip(p=0.5),
        Resize(args.input_h, args.input_w),
        transforms.Normalize(),
    ])
    val_transform = Compose([
        Resize(args.input_h, args.input_w),
        transforms.Normalize(),
    ])

    gen_view = 'resize' if args.gen_view == 'resize' else UKAN_GEN_VIEWS.get(dataset_name)
    train_ds = USGDataset(
        gen_view=gen_view, gen_size=args.gen_size,
        img_ids=train_ids,
        img_dir=os.path.join(args.data_dir, dataset_name, 'images'),
        mask_dir=os.path.join(args.data_dir, dataset_name, 'masks'),
        img_ext=img_ext, mask_ext=mask_ext,
        num_classes=1, transform=train_transform)
    val_ds = Dataset(
        img_ids=val_ids,
        img_dir=os.path.join(args.data_dir, dataset_name, 'images'),
        mask_dir=os.path.join(args.data_dir, dataset_name, 'masks'),
        img_ext=img_ext, mask_ext=mask_ext,
        num_classes=1, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # Logging
    log_path = os.path.join(output_path, 'log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'lr', 'train_loss', 'seg_loss', 'gen_loss',
                         'lambda_gen', 'val_iou', 'val_dice'])

    best_iou = 0
    running_seg_loss = 1.0
    running_gen_loss = 1.0

    for epoch in range(args.epochs):
        model.train()
        loss_meter = AverageMeter()
        seg_loss_meter = AverageMeter()
        gen_loss_meter = AverageMeter()
        lambda_meter = AverageMeter()

        pbar = tqdm(train_loader, desc='Epoch {}/{}'.format(epoch + 1, args.epochs))
        for inp, mask, x_0, _ in pbar:
            inp = inp.to(device)
            mask = mask.to(device)
            x_0 = x_0.to(device)

            # Compute dynamic lambda
            if args.lambda_type == 'dynamic':
                ratio = running_seg_loss / max(running_gen_loss, 1e-8)
                lam = args.lambda_0 * math.exp(-args.gamma * ratio)
            else:
                lam = args.lambda_val

            loss = 0.0
            seg_l = torch.tensor(0.0)
            gen_l = torch.tensor(0.0)

            if args.mode in ['seg_only', 'joint']:
                seg_out = model(inp, mode='seg')
                seg_l = seg_criterion(seg_out, mask)
                loss = loss + seg_l

            if args.mode in ['gen_only', 'joint']:
                # Noise-corrupted version of the same images for the generation branch
                t = torch.randint(0, args.T, (inp.shape[0],), device=device)
                x_t, noise = ddpm.q_sample(x_0, t)
                # In joint training the shared encoder keeps the BatchNorm statistics
                # of the segmentation inputs, which are used at evaluation time.
                with frozen_bn_stats(model.seg_model) if args.mode == 'joint' else nullcontext():
                    noise_pred = model(x_t, mode='gen', t=t)
                gen_l = F.mse_loss(noise_pred, noise)

                if args.mode == 'joint':
                    loss = loss + lam * gen_l
                else:
                    loss = gen_l

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update running losses for dynamic lambda
            running_seg_loss = 0.9 * running_seg_loss + 0.1 * seg_l.item()
            running_gen_loss = 0.9 * running_gen_loss + 0.1 * gen_l.item()

            loss_meter.update(loss.item(), inp.size(0))
            seg_loss_meter.update(seg_l.item(), inp.size(0))
            gen_loss_meter.update(gen_l.item(), inp.size(0))
            lambda_meter.update(lam, inp.size(0))

            pbar.set_postfix(loss='{:.4f}'.format(loss_meter.avg),
                             seg='{:.4f}'.format(seg_loss_meter.avg),
                             gen='{:.4f}'.format(gen_loss_meter.avg),
                             lam='{:.4f}'.format(lambda_meter.avg))

        lr = optimizer.param_groups[1]['lr']
        scheduler.step()

        # Validation (segmentation metrics); not defined for generation-only training
        val_iou, val_dice = 0.0, 0.0
        if args.mode != 'gen_only':
            val_iou, val_dice = seg_metrics(model, val_loader, device)

        print('Epoch {}: loss={:.4f} seg={:.4f} gen={:.4f} lam={:.4f} | val_iou={:.4f} val_dice={:.4f}'.format(
            epoch + 1, loss_meter.avg, seg_loss_meter.avg, gen_loss_meter.avg,
            lambda_meter.avg, val_iou, val_dice))

        # Log
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, lr, loss_meter.avg,
                             seg_loss_meter.avg, gen_loss_meter.avg,
                             lambda_meter.avg, val_iou, val_dice])

        # Save the best segmentation model; generation-only keeps the latest epoch
        if args.mode == 'gen_only' or val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(),
                       os.path.join(output_path, 'model.pth'))

    print('Best val IoU: {:.4f}'.format(best_iou))

    # Separate test split (e.g. official ISIC 2018): evaluate the best validation checkpoint
    test_ids = split_ids(img_ids, args.data_dir, dataset_name, 'test', args.dataseed)
    if args.mode != 'gen_only' and test_ids != val_ids:
        test_ds = Dataset(
            img_ids=test_ids,
            img_dir=os.path.join(args.data_dir, dataset_name, 'images'),
            mask_dir=os.path.join(args.data_dir, dataset_name, 'masks'),
            img_ext=img_ext, mask_ext=mask_ext,
            num_classes=1, transform=val_transform)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)
        model.load_state_dict(torch.load(os.path.join(output_path, 'model.pth'),
                                         map_location=device))
        test_iou, test_dice = seg_metrics(model, test_loader, device)
        with open(os.path.join(output_path, 'test.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iou', 'dice'])
            writer.writerow([test_iou, test_dice])
        print('Test ({} images): IoU {:.4f} - Dice {:.4f}'.format(len(test_ids), test_iou, test_dice))


if __name__ == '__main__':
    main()
