"""3D dataset loaders for Synapse, ACDC, and BraTS 2021."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


class RandomCrop3D:
    def __init__(self, crop_size):
        self.crop_size = crop_size  # (D, H, W)

    def __call__(self, image, label):
        d, h, w = image.shape
        cd, ch, cw = self.crop_size
        # Pad if volume is smaller than crop size
        pad_d = max(cd - d, 0)
        pad_h = max(ch - h, 0)
        pad_w = max(cw - w, 0)
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            label = np.pad(label, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            d, h, w = image.shape

        sd = np.random.randint(0, d - cd + 1)
        sh = np.random.randint(0, h - ch + 1)
        sw = np.random.randint(0, w - cw + 1)
        return image[sd:sd+cd, sh:sh+ch, sw:sw+cw], label[sd:sd+cd, sh:sh+ch, sw:sw+cw]


class RandomFlip3D:
    def __call__(self, image, label):
        for axis in range(3):
            if np.random.random() > 0.5:
                image = np.flip(image, axis=axis).copy()
                label = np.flip(label, axis=axis).copy()
        return image, label


class RandomRotate3D:
    def __call__(self, image, label):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k, axes=(1, 2)).copy()
        label = np.rot90(label, k, axes=(1, 2)).copy()
        return image, label


class SynapseDataset(Dataset):
    """Synapse multi-organ dataset (TransUNet format).

    Expected structure:
        data_dir/
        ├── train_npz/    # .npz files with 'image' and 'label' keys (2D slices)
        ├── test_vol/     # .npy.h5 files with 'image' and 'label' keys (3D volumes)
        └── lists/
            ├── train.txt
            └── test.txt

    Training slices listed as caseXXXX_sliceYYY are stacked back into one
    volume per case, so each training sample is a 3D crop of a whole case.
    """
    def __init__(self, data_dir, split='train', crop_size=(64, 128, 128),
                 num_classes=9, augment=True):
        self.data_dir = data_dir
        self.split = split
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.augment = augment and (split == 'train')

        list_file = os.path.join(data_dir, 'lists', f'{split}.txt')
        with open(list_file, 'r') as f:
            names = [line.strip() for line in f if line.strip()]

        self.case_slices = {}
        if split == 'train':
            for name in names:
                case = name.rsplit('_slice', 1)[0]
                self.case_slices.setdefault(case, []).append(name)
            for case, slices in self.case_slices.items():
                if len(slices) > 1:
                    slices.sort(key=lambda n: int(n.rsplit('_slice', 1)[1]))
            self.sample_list = list(self.case_slices)
        else:
            self.sample_list = names

        self.cropper = RandomCrop3D(crop_size)
        self.flipper = RandomFlip3D()
        self.rotator = RandomRotate3D()

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx]

        if self.split == 'train':
            # TransUNet format: 2D slices as .npz, stacked along depth
            slices = [np.load(os.path.join(self.data_dir, 'train_npz', s + '.npz'))
                      for s in self.case_slices[name]]
            image = np.stack([s['image'] for s in slices]) if slices[0]['image'].ndim == 2 \
                else slices[0]['image']
            label = np.stack([s['label'] for s in slices]) if slices[0]['label'].ndim == 2 \
                else slices[0]['label']
        else:
            # Test volumes as .npy.h5
            # Support both directory names
            test_dir = 'test_vol_h5' if os.path.isdir(os.path.join(self.data_dir, 'test_vol_h5')) else 'test_vol'
            filepath = os.path.join(self.data_dir, test_dir, name + '.npy.h5')
            data = h5py.File(filepath, 'r')
            image, label = np.array(data['image']), np.array(data['label'])

        # Normalize
        image = (image - image.mean()) / (image.std() + 1e-8)

        if self.augment:
            image, label = self.cropper(image, label)
            image, label = self.flipper(image, label)
            image, label = self.rotator(image, label)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # (1, D, H, W)
        label = torch.from_numpy(label.astype(np.int64))                 # (D, H, W)
        return image, label, name


class ACDCDataset(Dataset):
    """ACDC cardiac dataset.

    Expected structure:
        data_dir/
        ├── train/   # .h5 files with 'image' and 'label'
        ├── val/
        └── test/
    """
    def __init__(self, data_dir, split='train', crop_size=(16, 128, 128),
                 num_classes=4, augment=True):
        self.data_dir = data_dir
        self.split = split
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.augment = augment and (split == 'train')

        split_dir = os.path.join(data_dir, split)
        self.sample_list = sorted([f for f in os.listdir(split_dir) if f.endswith('.h5')])

        self.cropper = RandomCrop3D(crop_size)
        self.flipper = RandomFlip3D()

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx]
        filepath = os.path.join(self.data_dir, self.split, name)
        data = h5py.File(filepath, 'r')
        image = np.array(data['image'])   # (D, H, W)
        label = np.array(data['label'])   # (D, H, W)

        image = (image - image.mean()) / (image.std() + 1e-8)

        if self.augment:
            image, label = self.cropper(image, label)
            image, label = self.flipper(image, label)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.int64))
        return image, label, name


class BraTSDataset(Dataset):
    """BraTS 2021 dataset.

    Supports two formats:
    1. Preprocessed .npz (fast): data_dir/{split}_npz/*.npz with 'image' and 'label'
    2. Raw NIfTI (slow fallback): data_dir/{split}/BraTS2021_*/*.nii.gz

    With full_volume=True (evaluation only), whole volumes are returned
    without cropping, for sliding-window inference.
    """
    def __init__(self, data_dir, split='train', crop_size=(96, 96, 96),
                 num_classes=4, augment=True, full_volume=False):
        self.data_dir = data_dir
        self.split = split
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.augment = augment and (split == 'train')
        self.full_volume = full_volume and not self.augment

        # Prefer preprocessed npz if available
        npz_dir = os.path.join(data_dir, f'{split}_npz')
        if os.path.isdir(npz_dir) and len(os.listdir(npz_dir)) > 0:
            self.use_npz = True
            self.sample_list = sorted([f[:-4] for f in os.listdir(npz_dir)
                                       if f.endswith('.npz')])
        else:
            self.use_npz = False
            split_dir = os.path.join(data_dir, split)
            self.sample_list = sorted([d for d in os.listdir(split_dir)
                                       if os.path.isdir(os.path.join(split_dir, d))])

        self.cropper = RandomCrop3D(crop_size)
        self.flipper = RandomFlip3D()

    def __len__(self):
        return len(self.sample_list)

    def _load_npz(self, name):
        npz_dir = os.path.join(self.data_dir, f'{self.split}_npz')
        data = np.load(os.path.join(npz_dir, f'{name}.npz'))
        image = data['image'].astype(np.float32)  # (4, D, H, W), already remapped
        label = data['label'].astype(np.int64)
        # Per-modality normalization
        for i in range(image.shape[0]):
            m, s = image[i].mean(), image[i].std()
            image[i] = (image[i] - m) / (s + 1e-8)
        return image, label

    def _load_nifti(self, name):
        import nibabel as nib
        case_dir = os.path.join(self.data_dir, self.split, name)
        modalities = []
        for mod in ['t1', 't1ce', 't2', 'flair']:
            path = os.path.join(case_dir, f'{name}_{mod}.nii.gz')
            vol = nib.load(path).get_fdata().astype(np.float32)
            vol = (vol - vol.mean()) / (vol.std() + 1e-8)
            modalities.append(vol)
        image = np.stack(modalities, axis=0)
        seg_path = os.path.join(case_dir, f'{name}_seg.nii.gz')
        label = nib.load(seg_path).get_fdata().astype(np.int64)
        label[label == 4] = 3
        return image, label

    def __getitem__(self, idx):
        name = self.sample_list[idx]

        if self.use_npz:
            image, label = self._load_npz(name)
        else:
            image, label = self._load_nifti(name)

        # Crop to crop_size (random for train, center for val) unless full_volume
        if not self.full_volume:
            d, h, w = image.shape[1:]
            cd, ch, cw = self.crop_size
            pad_d, pad_h, pad_w = max(cd-d, 0), max(ch-h, 0), max(cw-w, 0)
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                image = np.pad(image, ((0,0),(0,pad_d),(0,pad_h),(0,pad_w)), mode='constant')
                label = np.pad(label, ((0,pad_d),(0,pad_h),(0,pad_w)), mode='constant')
                d, h, w = image.shape[1:]

            if self.augment:
                sd = np.random.randint(0, d - cd + 1)
                sh = np.random.randint(0, h - ch + 1)
                sw = np.random.randint(0, w - cw + 1)
            else:
                sd = (d - cd) // 2
                sh = (h - ch) // 2
                sw = (w - cw) // 2
            image = image[:, sd:sd+cd, sh:sh+ch, sw:sw+cw]
            label = label[sd:sd+cd, sh:sh+ch, sw:sw+cw]

        if self.augment:
            for axis in range(3):
                if np.random.random() > 0.5:
                    image = np.flip(image, axis=axis+1).copy()
                    label = np.flip(label, axis=axis).copy()

        image = torch.from_numpy(image.copy().astype(np.float32))
        label = torch.from_numpy(label.copy().astype(np.int64))
        return image, label, name


def get_dataset(dataset_name, data_dir, split='train', crop_size=None, augment=True,
                full_volume=False):
    """Factory function to get dataset by name.

    Synapse test and ACDC val/test volumes are always returned whole;
    full_volume only changes BraTS evaluation, which otherwise uses a center crop.
    """
    if dataset_name == 'synapse':
        crop_size = crop_size or (64, 128, 128)
        return SynapseDataset(data_dir, split=split, crop_size=crop_size,
                              num_classes=9, augment=augment)
    elif dataset_name == 'acdc':
        crop_size = crop_size or (16, 128, 128)
        return ACDCDataset(data_dir, split=split, crop_size=crop_size,
                           num_classes=4, augment=augment)
    elif dataset_name == 'brats2021':
        crop_size = crop_size or (96, 96, 96)
        return BraTSDataset(data_dir, split=split, crop_size=crop_size,
                            num_classes=4, augment=augment, full_volume=full_volume)
    else:
        raise ValueError(f"Unknown 3D dataset: {dataset_name}")
