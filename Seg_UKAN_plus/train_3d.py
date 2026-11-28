"""Training script for 3D U-KAN++ on volumetric medical image segmentation."""

import argparse
import os
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter

from archs_3d import UKANPP3D
from dataset_3d import get_dataset
from sliding_window import sliding_window_inference
from utils import AverageMeter


# Dataset configs
DATASET_CONFIGS = {
    'synapse': {'num_classes': 9, 'input_channels': 1, 'crop_size': (96, 96, 96)},
    'acdc':    {'num_classes': 4, 'input_channels': 1, 'crop_size': (16, 96, 96)},
    'brats2021': {'num_classes': 4, 'input_channels': 4, 'crop_size': (96, 96, 96)},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='experiment name')
    parser.add_argument('--dataset', default='synapse', choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--data_dir', required=True, help='path to dataset root')
    parser.add_argument('--output_dir', default='outputs_3d')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('--dataseed', default=2981, type=int)

    # Model
    parser.add_argument('--embed_dims', default='64,128,256', type=str)
    parser.add_argument('--grid_sizes', default='3,5,10', type=str)
    parser.add_argument('--fusion_type', default='softmax', choices=['softmax', 'concat', 'sum'])
    parser.add_argument('--no_kan', action='store_true')
    parser.add_argument('--use_stag', action='store_true', help='enable STAG in the decoder')
    parser.add_argument('--stag_stages', default='dec', type=str,
                        choices=['all', 'dec', 'dec1', 'dec2', 'dec12', 'none'])
    parser.add_argument('--stag_alpha', default=0.5, type=float)
    parser.add_argument('--stag_norm', default='minmax', type=str,
                        choices=['minmax', 'sigmoid', 'none'])

    # Crop size override
    parser.add_argument('--crop_size', default=None, type=str,
                        help='D,H,W crop size (e.g. "96,96,96")')

    # Optimizer
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam'])
    parser.add_argument('--momentum', default=0.99, type=float)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--kan_lr', default=1e-2, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float)

    # Scheduler
    parser.add_argument('--scheduler', default='poly',
                        choices=['poly', 'CosineAnnealingLR', 'ReduceLROnPlateau'])
    parser.add_argument('--poly_power', default=0.9, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)

    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--early_stopping', default=-1, type=int)

    # Epoch length and validation
    parser.add_argument('--iters_per_epoch', default=0, type=int,
                        help='training iterations per epoch (0: one pass over the training set)')
    parser.add_argument('--val_interval', default=1, type=int, help='validate every N epochs')
    parser.add_argument('--sw_overlap', default=0.5, type=float,
                        help='overlap of sliding-window validation (window = crop size)')
    parser.add_argument('--sw_batch_size', default=2, type=int)

    return parser.parse_args()


def dice_score(pred, target, num_classes, ignore_bg=True):
    """Compute per-class Dice score."""
    dices = []
    start = 1 if ignore_bg else 0
    for c in range(start, num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        if union == 0:
            dices.append(1.0)
        else:
            dices.append((2.0 * inter / union).item())
    return dices


class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss for multi-class segmentation."""
    def __init__(self, num_classes, weight_ce=1.0, weight_dice=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, pred, target):
        # pred: (B, C, D, H, W), target: (B, D, H, W)
        ce_loss = self.ce(pred, target)

        # Soft dice
        pred_soft = torch.softmax(pred, dim=1)
        dice_loss = 0
        count = 0
        for c in range(1, self.num_classes):  # skip background
            p = pred_soft[:, c]
            t = (target == c).float()
            inter = (p * t).sum()
            union = p.sum() + t.sum()
            if union > 0:
                dice_loss += 1.0 - 2.0 * inter / union
                count += 1
        if count > 0:
            dice_loss = dice_loss / count

        return self.weight_ce * ce_loss + self.weight_dice * dice_loss


def train_epoch(train_loader, model, criterion, optimizer, num_classes):
    model.train()
    meters = {'loss': AverageMeter(), 'dice': AverageMeter()}

    pbar = tqdm(total=len(train_loader), desc='Train')
    for image, label, _ in train_loader:
        image = image.cuda()
        label = label.cuda()

        output = model(image)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1)
        batch_dice = np.mean(dice_score(pred, label, num_classes))

        meters['loss'].update(loss.item(), image.size(0))
        meters['dice'].update(batch_dice, image.size(0))

        pbar.set_postfix(loss=f"{meters['loss'].avg:.4f}", dice=f"{meters['dice'].avg:.4f}")
        pbar.update(1)
    pbar.close()
    return OrderedDict(loss=meters['loss'].avg, dice=meters['dice'].avg)


def validate_epoch(val_loader, model, criterion, num_classes, window, overlap, sw_batch_size):
    model.eval()
    meters = {'loss': AverageMeter(), 'dice': AverageMeter()}
    per_class_dice = [AverageMeter() for _ in range(num_classes - 1)]

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader), desc='Val')
        for image, label, _ in val_loader:
            image = image.cuda()
            label = label.cuda()

            output = sliding_window_inference(image, model, window, overlap, sw_batch_size)
            loss = criterion(output, label)

            pred = output.argmax(dim=1)
            class_dices = dice_score(pred, label, num_classes)
            batch_dice = np.mean(class_dices)

            meters['loss'].update(loss.item(), image.size(0))
            meters['dice'].update(batch_dice, image.size(0))
            for i, d in enumerate(class_dices):
                per_class_dice[i].update(d, image.size(0))

            pbar.set_postfix(loss=f"{meters['loss'].avg:.4f}", dice=f"{meters['dice'].avg:.4f}")
            pbar.update(1)
        pbar.close()

    result = OrderedDict(loss=meters['loss'].avg, dice=meters['dice'].avg)
    for i, m in enumerate(per_class_dice):
        result[f'dice_c{i+1}'] = m.avg
    return result


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    config = vars(parse_args())
    seed_torch(config['dataseed'])

    ds_cfg = DATASET_CONFIGS[config['dataset']]
    num_classes = ds_cfg['num_classes']
    input_channels = ds_cfg['input_channels']

    crop_size = tuple(int(x) for x in config['crop_size'].split(',')) if config['crop_size'] \
        else ds_cfg['crop_size']

    exp_name = config['name'] or f"{config['dataset']}_ukanpp3d_seed{config['dataseed']}"
    out_path = os.path.join(config['output_dir'], exp_name)
    os.makedirs(out_path, exist_ok=True)

    writer = SummaryWriter(out_path)

    print('-' * 40)
    for k, v in config.items():
        print(f'{k}: {v}')
    print(f'crop_size: {crop_size}')
    print(f'num_classes: {num_classes}, input_channels: {input_channels}')
    print('-' * 40)

    with open(os.path.join(out_path, 'config.yml'), 'w') as f:
        yaml.dump(config, f)

    # Model
    embed_dims = [int(x) for x in config['embed_dims'].split(',')]
    grid_sizes = [int(x) for x in config['grid_sizes'].split(',')]

    model = UKANPP3D(
        num_classes=num_classes, input_channels=input_channels,
        embed_dims=embed_dims, grid_sizes=grid_sizes,
        fusion_type=config['fusion_type'], no_kan=config['no_kan'],
        use_stag=config['use_stag'], stag_stages=config['stag_stages'],
        stag_alpha=config['stag_alpha'], stag_norm=config['stag_norm'],
    ).cuda()

    # Param groups: higher LR for KAN layers
    param_groups = []
    for name, param in model.named_parameters():
        if 'layer' in name.lower() and 'fc' in name.lower():
            param_groups.append({'params': param, 'lr': config['kan_lr'],
                                 'weight_decay': config['kan_weight_decay']})
        else:
            param_groups.append({'params': param, 'lr': config['lr'],
                                 'weight_decay': config['weight_decay']})

    if config['optimizer'] == 'SGD':
        optimizer = optim.SGD(param_groups, lr=config['lr'], momentum=config['momentum'])
    else:
        optimizer = optim.Adam(param_groups)
    criterion = DiceCELoss(num_classes).cuda()

    if config['scheduler'] == 'poly':
        scheduler = lr_scheduler.LambdaLR(
            optimizer, lambda epoch: (1 - epoch / config['epochs']) ** config['poly_power'])
    elif config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'],
                                                    eta_min=config['min_lr'])
    else:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20,
                                                    min_lr=config['min_lr'])

    # Data
    train_dataset = get_dataset(config['dataset'], config['data_dir'],
                                split='train', crop_size=crop_size, augment=True)
    val_split = 'test' if config['dataset'] == 'synapse' else 'val'
    val_dataset = get_dataset(config['dataset'], config['data_dir'],
                              split=val_split, crop_size=crop_size, augment=False)

    sampler = None
    if config['iters_per_epoch'] > 0:
        sampler = torch.utils.data.RandomSampler(
            train_dataset, replacement=True,
            num_samples=config['iters_per_epoch'] * config['batch_size'])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=sampler is None, sampler=sampler,
        num_workers=config['num_workers'], drop_last=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=config['num_workers'], pin_memory=True)

    # Training loop
    log_cols = ['epoch', 'lr', 'loss', 'dice', 'val_loss', 'val_dice']
    log = OrderedDict([(c, []) for c in log_cols])

    best_dice = 0
    trigger = 0

    for epoch in range(config['epochs']):
        print(f'\nEpoch [{epoch}/{config["epochs"]}]')

        train_log = train_epoch(train_loader, model, criterion, optimizer, num_classes)
        validate = (epoch + 1) % config['val_interval'] == 0 or epoch + 1 == config['epochs']
        val_log = validate_epoch(val_loader, model, criterion, num_classes, crop_size,
                                 config['sw_overlap'], config['sw_batch_size']) if validate else None

        if config['scheduler'] == 'ReduceLROnPlateau':
            if val_log is not None:
                scheduler.step(val_log['loss'])
        else:
            scheduler.step()

        val_loss = val_log['loss'] if val_log is not None else float('nan')
        val_dice = val_log['dice'] if val_log is not None else float('nan')
        print(f"loss {train_log['loss']:.4f} - dice {train_log['dice']:.4f} "
              f"- val_loss {val_loss:.4f} - val_dice {val_dice:.4f}")

        log['epoch'].append(epoch)
        log['lr'].append(optimizer.param_groups[0]['lr'])
        log['loss'].append(train_log['loss'])
        log['dice'].append(train_log['dice'])
        log['val_loss'].append(val_loss)
        log['val_dice'].append(val_dice)

        pd.DataFrame(log).to_csv(os.path.join(out_path, 'log.csv'), index=False)

        writer.add_scalar('train/loss', train_log['loss'], epoch)
        writer.add_scalar('train/dice', train_log['dice'], epoch)
        if val_log is not None:
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/dice', val_dice, epoch)

        trigger += 1
        if val_log is not None and val_log['dice'] > best_dice:
            torch.save(model.state_dict(), os.path.join(out_path, 'model.pth'))
            best_dice = val_log['dice']
            print(f"=> saved best model (Dice: {best_dice:.4f})")
            # Save per-class dice
            per_class = {k: v for k, v in val_log.items() if k.startswith('dice_c')}
            if per_class:
                print(f"   Per-class: {per_class}")
            trigger = 0

        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()

    writer.close()
    print(f'\nBest val Dice: {best_dice:.4f}')


if __name__ == '__main__':
    main()
