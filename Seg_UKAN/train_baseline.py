"""Train baseline models (U-Net, Att-UNet, UNet++, TransUNet, U-NeXt) on any dataset.
Uses segmentation_models_pytorch for standard architectures.
Reuses the same dataset/training loop as train.py for fair comparison.
"""
import argparse, os, random, csv, yaml, shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from collections import OrderedDict
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from albumentations.augmentations import transforms
from albumentations.augmentations import geometric
from albumentations.core.composition import Compose, OneOf
from albumentations import RandomRotate90, Resize

import segmentation_models_pytorch as smp

import losses
from dataset import Dataset
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


ARCH_MAP = {
    'unet':    lambda: smp.Unet(encoder_name='resnet34', encoder_weights='imagenet',
                                 in_channels=3, classes=1),
    'attunet': lambda: smp.Unet(encoder_name='resnet34', encoder_weights='imagenet',
                                 in_channels=3, classes=1,
                                 decoder_attention_type='scse'),
    'unetpp':  lambda: smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights='imagenet',
                                         in_channels=3, classes=1),
    'transunet': lambda: smp.Unet(encoder_name='mit_b2', encoder_weights='imagenet',
                                    in_channels=3, classes=1),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--arch', required=True, choices=list(ARCH_MAP.keys()))
    parser.add_argument('--dataset', default='isic2018')
    parser.add_argument('--data_dir', default='../data')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--input_h', type=int, default=256)
    parser.add_argument('--input_w', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--dataseed', type=int, default=2981)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--input_channels', type=int, default=3)
    return parser.parse_args()


def train_epoch(train_loader, model, criterion, optimizer):
    avg_loss = AverageMeter()
    avg_iou = AverageMeter()
    model.train()
    pbar = tqdm(total=len(train_loader))
    for inp, target, _ in train_loader:
        inp, target = inp.cuda(), target.cuda()
        output = model(inp)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iou, dice, _ = iou_score(output, target)
        avg_loss.update(loss.item(), inp.size(0))
        avg_iou.update(iou, inp.size(0))
        pbar.set_postfix(loss=avg_loss.avg, iou=avg_iou.avg)
        pbar.update(1)
    pbar.close()
    return avg_loss.avg, avg_iou.avg


def validate(val_loader, model, criterion):
    avg_loss = AverageMeter()
    avg_iou = AverageMeter()
    avg_dice = AverageMeter()
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for inp, target, _ in val_loader:
            inp, target = inp.cuda(), target.cuda()
            output = model(inp)
            loss = criterion(output, target)
            iou, dice, _ = iou_score(output, target)
            avg_loss.update(loss.item(), inp.size(0))
            avg_iou.update(iou, inp.size(0))
            avg_dice.update(dice, inp.size(0))
            pbar.set_postfix(loss=avg_loss.avg, iou=avg_iou.avg, dice=avg_dice.avg)
            pbar.update(1)
        pbar.close()
    return avg_loss.avg, avg_iou.avg, avg_dice.avg


def main():
    args = parse_args()
    config = vars(args)
    seed_torch(config['dataseed'])

    exp_dir = os.path.join(config['output_dir'], config['name'])
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f)

    # Model
    model = ARCH_MAP[config['arch']]().cuda()
    print(f"Model: {config['arch']}, params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Loss
    criterion = losses.BCEDiceLoss().cuda()

    # Optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])

    # Dataset
    dataset_name = config['dataset']
    img_ext = '.png'
    mask_ext = '_mask.png' if dataset_name == 'busi' else '.png'

    img_ids = sorted(glob(os.path.join(config['data_dir'], dataset_name, 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    train_ids, val_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])

    train_transform = Compose([RandomRotate90(), geometric.transforms.Flip(),
                               Resize(config['input_h'], config['input_w']), transforms.Normalize()])
    val_transform = Compose([Resize(config['input_h'], config['input_w']), transforms.Normalize()])

    train_dataset = Dataset(img_ids=train_ids,
                            img_dir=os.path.join(config['data_dir'], dataset_name, 'images'),
                            mask_dir=os.path.join(config['data_dir'], dataset_name, 'masks'),
                            img_ext=img_ext, mask_ext=mask_ext,
                            num_classes=config['num_classes'], transform=train_transform)
    val_dataset = Dataset(img_ids=val_ids,
                          img_dir=os.path.join(config['data_dir'], dataset_name, 'images'),
                          mask_dir=os.path.join(config['data_dir'], dataset_name, 'masks'),
                          img_ext=img_ext, mask_ext=mask_ext,
                          num_classes=config['num_classes'], transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'],
                                                shuffle=True, num_workers=config['num_workers'], drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'],
                                              shuffle=False, num_workers=config['num_workers'], drop_last=False)

    # Training
    best_iou = 0
    log_rows = []
    for epoch in range(config['epochs']):
        train_loss, train_iou = train_epoch(train_loader, model, criterion, optimizer)
        val_loss, val_iou, val_dice = validate(val_loader, model, criterion)
        scheduler.step()

        print(f"Epoch {epoch}/{config['epochs']} - loss {train_loss:.4f} - iou {train_iou:.4f} "
              f"- val_loss {val_loss:.4f} - val_iou {val_iou:.4f} - val_dice {val_dice:.4f}")

        log_rows.append([epoch, optimizer.param_groups[0]['lr'], train_loss, train_iou,
                         val_loss, val_iou, val_dice])

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(exp_dir, 'model.pth'))
            print(f"=> saved best model (iou: {best_iou:.4f})")

    # Save log
    df = pd.DataFrame(log_rows, columns=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou', 'val_dice'])
    df.to_csv(os.path.join(exp_dir, 'log.csv'), index=False)
    print(f"Best IoU: {best_iou:.4f}")


if __name__ == '__main__':
    main()
