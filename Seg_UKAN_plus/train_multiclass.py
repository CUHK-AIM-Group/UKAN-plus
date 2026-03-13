"""
2D multi-class segmentation training for Synapse and ACDC.

Uses UKANPP with CE+Dice loss for multi-class segmentation on 2D slices
(same approach as TransUNet, Swin-UNet, etc.)

Usage:
    python train_multiclass.py --name synapse_ukanpp_seed2981 \
        --dataset synapse --data_dir ../data/synapse \
        --num_classes 9 --img_size 224 --epochs 150 --batch_size 24
"""
import argparse
import os
import random
import csv
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import archs
from utils import AverageMeter
from dataset_synapse_acdc import SynapseSliceDataset, ACDCSliceDataset


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class DiceCELoss(nn.Module):
    """Combined Dice + CrossEntropy loss for multi-class segmentation."""
    def __init__(self, num_classes, dice_weight=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        """
        pred: (B, num_classes, H, W)
        target: (B, H, W) long
        """
        ce_loss = self.ce(pred, target)

        # Dice loss
        pred_soft = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()

        dice_loss = 0.0
        for c in range(1, self.num_classes):  # skip background
            pred_c = pred_soft[:, c]
            target_c = target_one_hot[:, c]
            inter = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice_loss += 1 - (2 * inter + 1) / (union + 1)
        dice_loss /= (self.num_classes - 1)

        return (1 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss


def compute_dice_per_class(pred, target, num_classes):
    """Compute per-class Dice scores.

    pred: (B, num_classes, H, W)
    target: (B, H, W) long

    Returns: dict of class_idx -> dice_score
    """
    pred_labels = pred.argmax(dim=1)  # (B, H, W)
    dices = {}
    for c in range(1, num_classes):
        pred_c = (pred_labels == c).float()
        target_c = (target == c).float()
        inter = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        if union == 0:
            dices[c] = 1.0 if inter == 0 else 0.0
        else:
            dices[c] = (2 * inter / union).item()
    return dices


def evaluate_synapse_volumes(model, data_dir, num_classes, img_size, device):
    """Evaluate on Synapse test volumes with sliding window on 2D slices."""
    import h5py

    test_dir = 'test_vol_h5' if os.path.isdir(
        os.path.join(data_dir, 'test_vol_h5')) else 'test_vol'
    list_file = os.path.join(data_dir, 'lists', 'test.txt')
    with open(list_file, 'r') as f:
        test_list = [l.strip() for l in f if l.strip()]

    all_dices = {c: [] for c in range(1, num_classes)}

    model.eval()
    with torch.no_grad():
        for name in test_list:
            filepath = os.path.join(data_dir, test_dir, name + '.npy.h5')
            data = h5py.File(filepath, 'r')
            image = np.array(data['image']).astype(np.float32)  # (D, H, W)
            label = np.array(data['label']).astype(np.long)     # (D, H, W)

            pred_vol = np.zeros_like(label)
            for d in range(image.shape[0]):
                sl = image[d]
                # Resize
                import cv2
                sl_resized = cv2.resize(sl, (img_size, img_size),
                                        interpolation=cv2.INTER_LINEAR)
                # To tensor
                inp = torch.from_numpy(
                    np.stack([sl_resized]*3, axis=0)[np.newaxis]
                ).float().to(device)
                out = model(inp)  # (1, C, H, W)
                pred = out.argmax(dim=1).squeeze().cpu().numpy()
                # Resize back
                pred = cv2.resize(pred.astype(np.float32),
                                  (label.shape[2], label.shape[1]),
                                  interpolation=cv2.INTER_NEAREST).astype(np.long)
                pred_vol[d] = pred

            # Per-class Dice on this volume
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

    # Average across volumes
    organ_names = {1:'aorta', 2:'gallbladder', 3:'left_kidney', 4:'right_kidney',
                   5:'liver', 6:'pancreas', 7:'spleen', 8:'stomach'}
    mean_dice = 0
    for c in range(1, num_classes):
        d = np.mean(all_dices[c])
        name = organ_names.get(c, 'class_{}'.format(c))
        print('  {}: {:.4f}'.format(name, d))
        mean_dice += d
    mean_dice /= (num_classes - 1)
    print('  Mean Dice: {:.4f}'.format(mean_dice))
    return mean_dice


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--dataset', default='synapse', choices=['synapse', 'acdc'])
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--kan_lr', type=float, default=1e-2)
    parser.add_argument('--dataseed', type=int, default=2981)
    parser.add_argument('--num_workers', type=int, default=4)

    # Architecture
    parser.add_argument('--arch', default='UKANPP')
    parser.add_argument('--grid_sizes', type=str, default='3,5,10')
    parser.add_argument('--fusion_type', default='softmax')
    parser.add_argument('--no_kan', action='store_true')

    # Eval
    parser.add_argument('--eval_interval', type=int, default=10)

    return parser.parse_args()


def main():
    args = parse_args()
    seed_torch(args.dataseed)
    config = vars(args)

    output_path = os.path.join(args.output_dir, args.name)
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, 'config.yml'), 'w') as f:
        yaml.dump(config, f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    grid_sizes = [int(g) for g in args.grid_sizes.split(',')]
    embed_dims = [128, 160, 256]

    if args.arch == 'UKANPP':
        model = archs.UKANPP(
            num_classes=args.num_classes, input_channels=3,
            deep_supervision=False, embed_dims=embed_dims,
            grid_sizes=grid_sizes, fusion_type=args.fusion_type,
            no_kan=args.no_kan)
    else:
        model = archs.UKAN(
            num_classes=args.num_classes, input_channels=3,
            deep_supervision=False, embed_dims=embed_dims,
            no_kan=args.no_kan)

    model = model.to(device)

    # Optimizer with dual LR
    from kan import KANLinear
    kan_param_ids = set()
    for name, module in model.named_modules():
        if isinstance(module, KANLinear):
            for p in module.parameters():
                kan_param_ids.add(id(p))

    kan_params = [p for p in model.parameters() if id(p) in kan_param_ids]
    other_params = [p for p in model.parameters() if id(p) not in kan_param_ids]

    optimizer = torch.optim.AdamW([
        {'params': kan_params, 'lr': args.kan_lr, 'weight_decay': 0.01},
        {'params': other_params, 'lr': args.lr, 'weight_decay': 0.01},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    criterion = DiceCELoss(num_classes=args.num_classes)

    # Dataset
    if args.dataset == 'synapse':
        train_ds = SynapseSliceDataset(args.data_dir, split='train',
                                        img_size=args.img_size, augment=True)
        # No separate val set for Synapse; eval on test volumes periodically
        val_ds = None
    elif args.dataset == 'acdc':
        train_ds = ACDCSliceDataset(args.data_dir, split='train',
                                     img_size=args.img_size, augment=True)
        val_ds = ACDCSliceDataset(args.data_dir, split='valid',
                                   img_size=args.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              drop_last=True)
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)

    # Logging
    log_path = os.path.join(output_path, 'log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'lr', 'train_loss', 'val_mean_dice'])

    best_dice = 0

    for epoch in range(args.epochs):
        model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(train_loader, desc='Epoch {}/{}'.format(epoch + 1, args.epochs))
        for image, label, _ in pbar:
            image = image.to(device)
            label = label.to(device)

            pred = model(image)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_meter.update(loss.item(), image.size(0))
            pbar.set_postfix(loss='{:.4f}'.format(loss_meter.avg))

        scheduler.step()

        # Evaluate
        val_dice = 0.0
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            if args.dataset == 'synapse':
                print('Evaluating on Synapse test volumes...')
                val_dice = evaluate_synapse_volumes(
                    model, args.data_dir, args.num_classes,
                    args.img_size, device)
            elif val_ds is not None:
                # ACDC validation
                model.eval()
                dice_meter = AverageMeter()
                with torch.no_grad():
                    for image, label, _ in val_loader:
                        image = image.to(device)
                        label = label.to(device)
                        pred = model(image)
                        dices = compute_dice_per_class(pred, label, args.num_classes)
                        mean_d = np.mean(list(dices.values()))
                        dice_meter.update(mean_d, image.size(0))
                val_dice = dice_meter.avg
                print('Epoch {}: val_mean_dice={:.4f}'.format(epoch + 1, val_dice))

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(),
                           os.path.join(output_path, 'model.pth'))
                print('New best: {:.4f}'.format(best_dice))

        # Log
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, scheduler.get_last_lr()[0],
                              loss_meter.avg, val_dice])

    print('Training complete. Best Dice: {:.4f}'.format(best_dice))


if __name__ == '__main__':
    main()
