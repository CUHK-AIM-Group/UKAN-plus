import os

import cv2
import numpy as np
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split


def split_ids(img_ids, data_dir, dataset, split, seed):
    """Image ids of the 'train', 'val' or 'test' split of a 2D dataset.

    If <data_dir>/<dataset>/splits/ exists, <split>.txt in it lists the ids, one file
    name without extension per line (e.g. the official ISIC 2018 split written by
    prepare_isic2018.py). Otherwise img_ids are split 80/20 at random with `seed`, and
    'val' and 'test' are the same 20%.
    """
    split_dir = os.path.join(data_dir, dataset, 'splits')
    if os.path.isdir(split_dir):
        with open(os.path.join(split_dir, split + '.txt')) as f:
            return [line.strip() for line in f if line.strip()]
    train_ids, val_ids = train_test_split(img_ids, test_size=0.2, random_state=seed)
    return train_ids if split == 'train' else val_ids


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        for i in range(self.num_classes):

            # print(os.path.join(self.mask_dir, str(i),
            #             img_id + self.mask_ext))

            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        if mask.max()<1:
            mask[mask>0] = 1.0

        return img, mask, {'img_id': img_id}
