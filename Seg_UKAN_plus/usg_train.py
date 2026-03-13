"""
Unified Seg-Gen (USG) Training Script — Table VII.

Shared MS-KAN encoder with:
  - Segmentation decoder head (from UKANPP)
  - Generation decoder head (lightweight DDPM denoiser)
  - Dynamic loss balancing: lambda(t) = lambda_0 * exp(-gamma * L_seg / L_gen)

Modes:
  - seg_only:  standard segmentation (reuse UKANPP, same as Table I)
  - gen_only:  generation only using encoder features
  - joint:     unified framework with dynamic or fixed lambda

Usage:
    python usg_train.py --name usg_joint_dynamic_seed2981 --dataset busi \
        --mode joint --lambda_type dynamic --lambda_0 1.0 --gamma 0.5 \
        --grid_sizes 3,5,10 --dataseed 2981
"""
import argparse
import os
import random
import csv
import yaml
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from glob import glob
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from albumentations import RandomRotate90, Resize, Flip

import archs
from dataset import Dataset
from losses import BCEDiceLoss
from metrics import iou_score
from utils import AverageMeter, str2bool


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---- Lightweight Generation Decoder ----

class GenDecoder(nn.Module):
    """Lightweight generation decoder that takes encoder bottleneck features
    and predicts noise (for DDPM training).

    Bottleneck is at 1/32 resolution. Needs 5 upsamplings to reach full res.
    """
    def __init__(self, embed_dim=256, img_channels=3, T=1000):
        super().__init__()
        ch = embed_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(ch, ch * 4),
            nn.SiLU(),
            nn.Linear(ch * 4, ch),
        )

        # Register sinusoidal position encoding for timesteps
        self.register_buffer('time_freqs',
                             self._sinusoidal_embedding(T, ch))

        # Upsampling path: bottleneck (ch, H/32, W/32) -> (img_channels, H, W)
        # 5 up-blocks: 1/32 -> 1/16 -> 1/8 -> 1/4 -> 1/2 -> 1
        self.up1 = self._up_block(ch, ch)          # 1/32 -> 1/16
        self.up2 = self._up_block(ch, ch // 2)     # 1/16 -> 1/8
        self.up3 = self._up_block(ch // 2, ch // 4)  # 1/8 -> 1/4
        self.up4 = self._up_block(ch // 4, ch // 8)  # 1/4 -> 1/2
        self.up5 = self._up_block(ch // 8, ch // 8)  # 1/2 -> 1

        # Final conv to image channels
        self.final = nn.Sequential(
            nn.Conv2d(ch // 8, ch // 8, 3, padding=1),
            nn.GroupNorm(8, ch // 8),
            nn.SiLU(),
            nn.Conv2d(ch // 8, img_channels, 1),
        )

    def _sinusoidal_embedding(self, T, dim):
        pos = torch.arange(T).float()
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim).float() * -emb)
        emb = pos[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (T, dim)

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, encoder_feat, noisy_img, t):
        """
        Args:
            encoder_feat: (B, C, H/32, W/32) bottleneck features from shared encoder
            noisy_img: (B, 3, H, W) noisy image at timestep t
            t: (B,) integer timesteps

        Returns:
            noise_pred: (B, 3, H, W) predicted noise
        """
        # Time conditioning
        t_emb = self.time_freqs[t]  # (B, ch)
        t_emb = self.time_embed(t_emb)  # (B, ch)
        t_emb = t_emb[:, :, None, None]  # (B, ch, 1, 1)

        # Add time embedding to encoder features
        x = encoder_feat + t_emb

        # Upsample: 1/32 -> full resolution
        x = self.up1(x)   # -> 1/16
        x = self.up2(x)   # -> 1/8
        x = self.up3(x)   # -> 1/4
        x = self.up4(x)   # -> 1/2
        x = self.up5(x)   # -> 1

        return self.final(x)


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

        out = F.relu(F.interpolate(m.decoder1(bottleneck), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t4)
        _, _, H2, W2 = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in m.dblock1:
            out = blk(out, H2, W2)
        out = m.dnorm3(out)
        out = out.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(m.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t3)
        _, _, H3, W3 = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in m.dblock2:
            out = blk(out, H3, W3)
        out = m.dnorm4(out)
        out = out.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(m.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(m.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(m.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        return m.final(out)

    def decode_gen(self, bottleneck, noisy_img, t):
        """Run generation decoder."""
        return self.gen_decoder(bottleneck, noisy_img, t)

    def forward(self, x, mode='seg', noisy_img=None, t=None):
        bottleneck, skips = self.encode(x)
        if mode == 'seg':
            return self.decode_seg(bottleneck, skips)
        elif mode == 'gen':
            return self.decode_gen(bottleneck, noisy_img, t)
        elif mode == 'both':
            seg_out = self.decode_seg(bottleneck, skips)
            gen_out = self.decode_gen(bottleneck, noisy_img, t)
            return seg_out, gen_out


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

    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--kan_lr', type=float, default=1e-2)

    return parser.parse_args()


def main():
    args = parse_args()
    seed_torch(args.dataseed)

    config = vars(args)
    grid_sizes = [int(g) for g in args.grid_sizes.split(',')]

    output_path = os.path.join(args.output_dir, args.name)
    os.makedirs(output_path, exist_ok=True)

    # Save config
    with open(os.path.join(output_path, 'config.yml'), 'w') as f:
        yaml.dump(config, f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    seg_model = archs.UKANPP(
        num_classes=1, input_channels=3, deep_supervision=False,
        embed_dims=[128, 160, 256], grid_sizes=grid_sizes,
        fusion_type=args.fusion_type)

    embed_dims = [128, 160, 256]
    # Bottleneck is embed_dims[2] = 256 channels at 1/16 resolution
    gen_decoder = GenDecoder(embed_dim=embed_dims[2], img_channels=3, T=args.T)

    model = USGModel(seg_model, gen_decoder).to(device)
    ddpm = SimpleDDPM(T=args.T, device=device)

    # Optimizer with dual learning rates
    kan_params, other_params = [], []
    for name, param in model.named_parameters():
        if 'KANLinear' in type(param).__name__ or 'kan' in name.lower():
            kan_params.append(param)
        else:
            other_params.append(param)

    # Simpler grouping: check module type
    kan_param_ids = set()
    for name, module in model.named_modules():
        if 'KANLinear' in type(module).__name__:
            for p in module.parameters():
                kan_param_ids.add(id(p))

    kan_params = [p for p in model.parameters() if id(p) in kan_param_ids]
    other_params = [p for p in model.parameters() if id(p) not in kan_param_ids]

    optimizer = torch.optim.Adam([
        {'params': kan_params, 'lr': args.kan_lr},
        {'params': other_params, 'lr': args.lr},
    ])

    seg_criterion = BCEDiceLoss()

    # Dataset
    dataset_name = args.dataset
    img_ext = '.png'
    mask_ext_map = {'busi': '_mask.png'}
    mask_ext = mask_ext_map.get(dataset_name, '.png')

    img_ids = sorted(glob(os.path.join(args.data_dir, dataset_name, 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    train_ids, val_ids = train_test_split(img_ids, test_size=0.2,
                                          random_state=args.dataseed)

    train_transform = Compose([
        RandomRotate90(),
        Flip(),
        Resize(args.input_h, args.input_w),
        OneOf([
            transforms.HueSaturationValue(),
            transforms.RandomBrightnessContrast(),
        ], p=0.5),
        transforms.Normalize(),
    ])
    val_transform = Compose([
        Resize(args.input_h, args.input_w),
        transforms.Normalize(),
    ])

    train_ds = Dataset(
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
        for inp, mask, _ in pbar:
            inp = inp.to(device)
            mask = mask.to(device)

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
                # Create noisy version of input images for diffusion
                x_0 = inp * 2 - 1  # normalize to [-1, 1] (images are [0,1] after Normalize)
                t = torch.randint(0, args.T, (inp.shape[0],), device=device)
                x_t, noise = ddpm.q_sample(x_0, t)

                # Get encoder features (shared)
                bottleneck, skips = model.encode(inp)
                noise_pred = model.decode_gen(bottleneck.detach() if args.mode == 'gen_only' else bottleneck,
                                              x_t, t)
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

        # Validation (segmentation metrics)
        model.eval()
        iou_meter = AverageMeter()
        dice_meter = AverageMeter()

        with torch.no_grad():
            for inp, mask, _ in val_loader:
                inp = inp.to(device)
                mask = mask.to(device)
                seg_out = model(inp, mode='seg')
                iou, dice, _ = iou_score(seg_out, mask)
                iou_meter.update(iou, inp.size(0))
                dice_meter.update(dice, inp.size(0))

        print('Epoch {}: loss={:.4f} seg={:.4f} gen={:.4f} lam={:.4f} | val_iou={:.4f} val_dice={:.4f}'.format(
            epoch + 1, loss_meter.avg, seg_loss_meter.avg, gen_loss_meter.avg,
            lambda_meter.avg, iou_meter.avg, dice_meter.avg))

        # Log
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, args.lr, loss_meter.avg,
                              seg_loss_meter.avg, gen_loss_meter.avg,
                              lambda_meter.avg, iou_meter.avg, dice_meter.avg])

        # Save best model
        if iou_meter.avg > best_iou:
            best_iou = iou_meter.avg
            torch.save(model.state_dict(),
                       os.path.join(output_path, 'model.pth'))

    print('Best val IoU: {:.4f}'.format(best_iou))


if __name__ == '__main__':
    main()
