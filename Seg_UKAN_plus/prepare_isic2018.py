"""
Prepare ISIC 2018 Task 1 with its official train / validation / test split.

Reads the official challenge release (https://challenge.isic-archive.com/data/#2018):
    <isic_root>/ISIC2018_Task1-2_Training_Input/ISIC_*.jpg
    <isic_root>/ISIC2018_Task1_Training_GroundTruth/ISIC_*_segmentation.png
    and the Validation and Test folders with the same naming,
and writes the layout read by train.py, train_baseline.py and usg_train.py:
    <out_dir>/images/<id>.png
    <out_dir>/masks/0/<id>.png
    <out_dir>/splits/{train,val,test}.txt
With splits/ present, the training scripts select the checkpoint on 'val' and write
the metrics on 'test' to test.csv (see dataset.split_ids).

Usage:
    python prepare_isic2018.py --isic_root /path/to/ISIC2018 --out_dir /path/to/data_root/isic2018
"""
import argparse
import os
from glob import glob

import cv2
from tqdm import tqdm

SPLITS = {
    'train': ('ISIC2018_Task1-2_Training_Input', 'ISIC2018_Task1_Training_GroundTruth'),
    'val': ('ISIC2018_Task1-2_Validation_Input', 'ISIC2018_Task1_Validation_GroundTruth'),
    'test': ('ISIC2018_Task1-2_Test_Input', 'ISIC2018_Task1_Test_GroundTruth'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--isic_root', required=True,
                        help='folder containing the six official ISIC 2018 Task 1 folders')
    parser.add_argument('--out_dir', required=True,
                        help='output dataset folder, e.g. <data_root>/isic2018')
    parser.add_argument('--size', type=int, default=0,
                        help='resize images and masks to size x size to save disk space and loading '
                             'time; 0 keeps the original resolution (the training scripts resize on the fly)')
    args = parser.parse_args()

    for sub in ('images', os.path.join('masks', '0'), 'splits'):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    for split, (img_folder, gt_folder) in SPLITS.items():
        paths = sorted(glob(os.path.join(args.isic_root, img_folder, 'ISIC_*.jpg')))
        if not paths:
            raise FileNotFoundError('no ISIC_*.jpg in ' + os.path.join(args.isic_root, img_folder))
        ids = []
        for path in tqdm(paths, desc=split):
            img_id = os.path.splitext(os.path.basename(path))[0]
            img = cv2.imread(path)
            mask = cv2.imread(os.path.join(args.isic_root, gt_folder, img_id + '_segmentation.png'),
                              cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError('missing ground truth of %s in %s' % (img_id, gt_folder))
            if args.size:
                img = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (args.size, args.size), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(args.out_dir, 'images', img_id + '.png'), img)
            cv2.imwrite(os.path.join(args.out_dir, 'masks', '0', img_id + '.png'), mask)
            ids.append(img_id)
        with open(os.path.join(args.out_dir, 'splits', split + '.txt'), 'w') as f:
            f.write('\n'.join(ids) + '\n')
        print('%s: %d images' % (split, len(ids)))


if __name__ == '__main__':
    main()
