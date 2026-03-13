"""2D slice datasets for Synapse and ACDC (TransUNet-style training)."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class SynapseSliceDataset(Dataset):
    """Synapse multi-organ 2D slice dataset (TransUNet format).

    Loads .npz slices from train_npz/ with keys 'image' and 'label'.
    For testing, loads .npy.h5 volumes and evaluates per-volume.

    Args:
        data_dir: path to synapse/ directory
        split: 'train' or 'test'
        img_size: resize to (img_size, img_size)
        num_classes: number of segmentation classes (9 for Synapse: bg + 8 organs)
    """
    def __init__(self, data_dir, split='train', img_size=224, num_classes=9,
                 augment=True):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.num_classes = num_classes
        self.augment = augment and (split == 'train')

        list_file = os.path.join(data_dir, 'lists', '{}.txt'.format(split))
        with open(list_file, 'r') as f:
            self.sample_list = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx]

        if self.split == 'train':
            data = np.load(os.path.join(self.data_dir, 'train_npz',
                                        name + '.npz'))
            image, label = data['image'].astype(np.float32), data['label'].astype(np.float32)
        else:
            import h5py
            test_dir = 'test_vol_h5' if os.path.isdir(
                os.path.join(self.data_dir, 'test_vol_h5')) else 'test_vol'
            filepath = os.path.join(self.data_dir, test_dir, name + '.npy.h5')
            data = h5py.File(filepath, 'r')
            image = np.array(data['image']).astype(np.float32)
            label = np.array(data['label']).astype(np.float32)
            # For test volumes, return the full volume for sliding window eval
            return image, label, name

        # Resize
        if image.shape[0] != self.img_size or image.shape[1] != self.img_size:
            image = cv2.resize(image, (self.img_size, self.img_size),
                               interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.img_size, self.img_size),
                               interpolation=cv2.INTER_NEAREST)

        # Augmentation
        if self.augment:
            if np.random.random() > 0.5:
                image = np.flip(image, axis=1).copy()
                label = np.flip(label, axis=1).copy()
            if np.random.random() > 0.5:
                image = np.flip(image, axis=0).copy()
                label = np.flip(label, axis=0).copy()
            k = np.random.randint(0, 4)
            image = np.rot90(image, k).copy()
            label = np.rot90(label, k).copy()

        # Expand to 3-channel (grayscale -> RGB-like)
        image = np.stack([image, image, image], axis=0)  # (3, H, W)
        image = torch.from_numpy(image)
        label = torch.from_numpy(label.astype(np.long))  # (H, W)
        return image, label, name


class ACDCSliceDataset(Dataset):
    """ACDC cardiac 2D slice dataset (MT-UNet/Zenodo format).

    Loads .npz slices with keys 'img' and 'label'.

    Args:
        data_dir: path to acdc/ directory
        split: 'train', 'valid', or 'test'
        img_size: resize to (img_size, img_size)
        num_classes: 4 (bg, RV, MYO, LV)
    """
    def __init__(self, data_dir, split='train', img_size=224, num_classes=4,
                 augment=True):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.num_classes = num_classes
        self.augment = augment and (split == 'train')

        split_dir = os.path.join(data_dir, split)
        self.sample_list = sorted([f for f in os.listdir(split_dir)
                                   if f.endswith('.npz')])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx]
        data = np.load(os.path.join(self.data_dir, self.split, name))
        image = data['img'].astype(np.float32)
        label = data['label'].astype(np.float32)

        # Resize
        if image.shape[0] != self.img_size or image.shape[1] != self.img_size:
            image = cv2.resize(image, (self.img_size, self.img_size),
                               interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.img_size, self.img_size),
                               interpolation=cv2.INTER_NEAREST)

        # Augmentation
        if self.augment:
            if np.random.random() > 0.5:
                image = np.flip(image, axis=1).copy()
                label = np.flip(label, axis=1).copy()
            if np.random.random() > 0.5:
                image = np.flip(image, axis=0).copy()
                label = np.flip(label, axis=0).copy()
            k = np.random.randint(0, 4)
            image = np.rot90(image, k).copy()
            label = np.rot90(label, k).copy()

        # Expand to 3-channel
        image = np.stack([image, image, image], axis=0)  # (3, H, W)
        image = torch.from_numpy(image)
        label = torch.from_numpy(label.astype(np.long))  # (H, W)
        return image, label, name
