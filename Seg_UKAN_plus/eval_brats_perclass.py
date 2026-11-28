"""Quick BraTS per-class eval: ET, TC, WT dice from saved checkpoints."""
import os, sys, yaml, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_3d import get_dataset
from archs_3d import UKANPP3D
from utils import AverageMeter

def dice_single(pred, gt):
    smooth = 1e-5
    inter = (pred & gt).sum()
    return (2.0 * inter + smooth) / (pred.sum() + gt.sum() + smooth)

def eval_brats(exp_dir, data_dir, device='cuda'):
    config_path = os.path.join(exp_dir, 'config.yml')
    ckpt_path = os.path.join(exp_dir, 'model.pth')
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    embed_dims = [int(x) for x in config['embed_dims'].split(',')]
    grid_sizes = [int(x) for x in config['grid_sizes'].split(',')]

    model = UKANPP3D(
        num_classes=4, input_channels=4, embed_dims=embed_dims,
        grid_sizes=grid_sizes, fusion_type=config.get('fusion_type', 'softmax'),
        no_kan=config.get('no_kan', False),
        use_stag=config.get('use_stag', False), stag_stages=config.get('stag_stages', 'dec'),
        stag_alpha=config.get('stag_alpha', 0.5), stag_norm=config.get('stag_norm', 'minmax'))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device).eval()

    val_dataset = get_dataset('brats2021', data_dir, split='val',
                              crop_size=(96, 96, 96), augment=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # BraTS classes: 0=bg, 1=NCR/NET, 2=ED, 3=ET
    # Regions: ET=class3, TC=class1+class3, WT=class1+class2+class3
    et_dice, tc_dice, wt_dice = AverageMeter(), AverageMeter(), AverageMeter()

    with torch.no_grad():
        for image, mask, _name in val_loader:
            image, mask = image.to(device), mask.to(device)
            output = model(image)
            pred = output.argmax(dim=1).cpu().numpy()[0]
            gt = mask.cpu().numpy()[0]

            # ET: enhancing tumor (class 3)
            et_d = dice_single(pred == 3, gt == 3)
            # TC: tumor core (class 1 + class 3)
            tc_d = dice_single((pred == 1) | (pred == 3), (gt == 1) | (gt == 3))
            # WT: whole tumor (class 1 + class 2 + class 3)
            wt_d = dice_single(pred >= 1, gt >= 1)

            et_dice.update(et_d, 1)
            tc_dice.update(tc_d, 1)
            wt_dice.update(wt_d, 1)

    mean_d = (et_dice.avg + tc_dice.avg + wt_dice.avg) / 3
    return et_dice.avg, tc_dice.avg, wt_dice.avg, mean_d

if __name__ == '__main__':
    import csv
    data_dir = sys.argv[1]
    exp_dirs = sys.argv[2:]
    device = 'cuda'

    for exp_dir in exp_dirs:
        name = os.path.basename(exp_dir)
        print(f"Evaluating {name}...")
        et, tc, wt, mean = eval_brats(exp_dir, data_dir, device)
        print(f"  ET={et*100:.2f}, TC={tc*100:.2f}, WT={wt*100:.2f}, Mean={mean*100:.2f}")

        # Save to csv
        csv_path = os.path.join(exp_dir, 'eval_perclass.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['ET_dice', 'TC_dice', 'WT_dice', 'Mean_dice', 'experiment'])
            w.writerow([et, tc, wt, mean, name])
        print(f"  Saved to {csv_path}")
