"""
Table XII: Statistical Testing — paired t-test between U-KAN and U-KAN++.

Compares 3-seed results for IoU and Dice across datasets.

Usage:
    python stat_test.py --output_dir outputs
"""
import os
import sys
import csv
import numpy as np
from scipy import stats


def load_best_metric(output_dir, name, metric='val_iou'):
    """Load the best val metric from a run's log.csv."""
    log_path = os.path.join(output_dir, name, 'log.csv')
    if not os.path.exists(log_path):
        return None
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    best_row = max(rows, key=lambda r: float(r.get('val_iou', 0)))
    return float(best_row.get(metric, 0))


def paired_t_test(scores_a, scores_b):
    """Paired t-test. Returns (t_stat, p_value)."""
    if len(scores_a) != len(scores_b) or len(scores_a) < 2:
        return float('nan'), float('nan')
    t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
    return t_stat, p_val


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'outputs'
    seeds = [2981, 6142, 1187]
    datasets = ['busi', 'glas', 'cvc', 'isic2018']

    # Table XII: U-KAN baseline vs U-KAN++ comparison
    print("=" * 70)
    print("Table XII: Paired t-test (U-KAN baseline vs U-KAN++)")
    print("=" * 70)
    print("{:<12} {:>8} {:>8} {:>10} {:>8} {:>8} {:>10}".format(
        'Dataset', 'IoU_base', 'IoU_pp', 'p(IoU)', 'Dice_base', 'Dice_pp', 'p(Dice)'))
    print("-" * 70)

    for ds in datasets:
        # U-KAN++ results (from Table I runs)
        pp_ious, pp_dices = [], []
        for seed in seeds:
            iou = load_best_metric(output_dir, '{}_ukanpp_seed{}'.format(ds, seed), 'val_iou')
            dice = load_best_metric(output_dir, '{}_ukanpp_seed{}'.format(ds, seed), 'val_dice')
            if iou is not None:
                pp_ious.append(iou)
                pp_dices.append(dice)

        # U-KAN baseline results (from Table VIII progressive runs)
        base_ious, base_dices = [], []
        for seed in seeds:
            iou = load_best_metric(output_dir, 'progressive_ukan_baseline_seed{}'.format(seed), 'val_iou')
            dice = load_best_metric(output_dir, 'progressive_ukan_baseline_seed{}'.format(seed), 'val_dice')
            if iou is not None:
                base_ious.append(iou)
                base_dices.append(dice)

        if len(pp_ious) >= 2 and len(base_ious) >= 2:
            _, p_iou = paired_t_test(base_ious, pp_ious)
            _, p_dice = paired_t_test(base_dices, pp_dices)
            print("{:<12} {:>8.4f} {:>8.4f} {:>10.6f} {:>8.4f} {:>8.4f} {:>10.6f}".format(
                ds,
                np.mean(base_ious), np.mean(pp_ious), p_iou,
                np.mean(base_dices), np.mean(pp_dices), p_dice))
        elif len(pp_ious) >= 2:
            print("{:<12} {:>8} {:>8.4f} {:>10} {:>8} {:>8.4f} {:>10}".format(
                ds, '--', np.mean(pp_ious), '--', '--', np.mean(pp_dices), '--'))
        else:
            print("{:<12} {:>8} {:>8} {:>10} {:>8} {:>8} {:>10}".format(
                ds, '--', '--', '--', '--', '--', '--'))

    # Also compare ablation configs vs full U-KAN++
    print("\n" + "=" * 70)
    print("MS-KAN Ablation: pairwise p-values vs default [3,5,10] softmax (BUSI)")
    print("=" * 70)

    default_ious = []
    for seed in seeds:
        v = load_best_metric(output_dir, 'ablation_g3_5_10_softmax_seed{}'.format(seed), 'val_iou')
        if v is not None:
            default_ious.append(v)

    ablation_configs = [
        ("3", "softmax", "G=3"),
        ("5", "softmax", "G=5"),
        ("10", "softmax", "G=10"),
        ("3_5", "softmax", "G={3,5}"),
        ("3_5_10", "concat", "G={3,5,10} concat"),
        ("3_5_10", "sum", "G={3,5,10} sum"),
    ]

    print("{:<25} {:>8} {:>10}".format('Config', 'IoU', 'p-value'))
    print("-" * 45)

    for grid_tag, fusion, label in ablation_configs:
        ious = []
        for seed in seeds:
            v = load_best_metric(output_dir, 'ablation_g{}_{}_seed{}'.format(grid_tag, fusion, seed), 'val_iou')
            if v is not None:
                ious.append(v)

        if len(ious) >= 2 and len(default_ious) >= 2:
            _, p = paired_t_test(ious, default_ious)
            print("{:<25} {:>8.4f} {:>10.6f}".format(label, np.mean(ious), p))
        elif len(ious) >= 2:
            print("{:<25} {:>8.4f} {:>10}".format(label, np.mean(ious), '--'))
        else:
            print("{:<25} {:>8} {:>10}".format(label, '--', '--'))

    if len(default_ious) >= 2:
        print("{:<25} {:>8.4f} {:>10}".format('G={3,5,10} softmax*', np.mean(default_ious), 'ref'))


if __name__ == '__main__':
    main()
