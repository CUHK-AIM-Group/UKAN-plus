"""
Statistical Testing — paired comparison between U-KAN and U-KAN++ (Dice).

Runs are paired by seed within each dataset. For every dataset the script
reports mean±SD of both models, the mean paired difference with its 95%
t-interval and the paired t-test p-value, then applies the Holm correction
across all datasets in the file.

Input CSV columns (one row per dataset and seed):
    dataset,seed,ukan_dice,ukanpp_dice

Usage:
    python stat_test.py --csv dice_per_seed.csv
"""
import argparse
import csv
from collections import OrderedDict

import numpy as np
from scipy import stats


def load_pairs(csv_path):
    """Return {dataset: [(seed, ukan_dice, ukanpp_dice), ...]} in file order."""
    pairs = OrderedDict()
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            pairs.setdefault(row['dataset'], []).append(
                (int(row['seed']), float(row['ukan_dice']), float(row['ukanpp_dice'])))
    return pairs


def paired_summary(base, new):
    """Mean paired difference, its 95% t-interval, and the paired t-test p-value."""
    diff = new - base
    n = len(diff)
    mean_diff = diff.mean()
    half = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)
    p = stats.ttest_rel(new, base).pvalue
    return mean_diff, (mean_diff - half, mean_diff + half), p


def holm(pvals):
    """Holm step-down adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(np.argsort(pvals)):
        running = max(running, (m - rank) * pvals[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True,
                        help='per-seed Dice with columns dataset,seed,ukan_dice,ukanpp_dice')
    args = parser.parse_args()

    rows = []
    for ds, entries in load_pairs(args.csv).items():
        entries.sort()
        base = np.array([e[1] for e in entries])
        new = np.array([e[2] for e in entries])
        if len(base) < 2:
            raise ValueError('%s: at least two paired seeds are required' % ds)
        mean_diff, ci, p = paired_summary(base, new)
        rows.append((ds, base, new, mean_diff, ci, p))

    adjusted = holm([r[5] for r in rows])

    print("{:<12} {:>15} {:>15} {:>7} {:>18} {:>8} {:>8}".format(
        'Dataset', 'U-KAN', 'U-KAN++', 'Delta', '95% CI', 'p', 'Holm p'))
    print("-" * 89)
    for (ds, base, new, mean_diff, ci, p), p_holm in zip(rows, adjusted):
        print("{:<12} {:>8.2f}±{:<6.2f} {:>8.2f}±{:<6.2f} {:>+7.2f} [{:>+7.2f}, {:>+7.2f}] {:>8.4f} {:>8.4f}".format(
            ds, base.mean(), base.std(ddof=1), new.mean(), new.std(ddof=1),
            mean_diff, ci[0], ci[1], p, p_holm))


if __name__ == '__main__':
    main()
