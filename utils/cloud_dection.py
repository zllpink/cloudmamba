import os
import cv2
import torch
import numpy as np
import tifffile
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

import random
import functools
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class RandomFlipOrRotate(object):
    funcs = [TF.hflip,
             TF.vflip,
             functools.partial(TF.rotate, angle=90),
             functools.partial(TF.rotate, angle=180),
             functools.partial(TF.rotate, angle=270)]
    def __call__(self, imgs: list):
        rand = random.randint(0, 4)
        for i in range(len(imgs)):
            img = imgs[i]
            imgs[i] = self.funcs[rand](img)
        return imgs


class ImageDataset(Dataset):
    """WHUS2 binary cloud detection dataset.

    Directory structure (applies to both train and test):
        <root>/<mode>/image/*.tif   -- 4-band uint16 TIF images (H, W, 4), value range ~[0, 4500]
        <root>/<mode>/gt/*.tif      -- single-band binary labels (0=clear, 255=cloud)

    If the val directory is empty, samples are automatically split from train
    at val_ratio proportion with a fixed random seed for reproducibility.
    """

    _MEAN    = [0.5, 0.5, 0.5, 0.5]
    _STD     = [0.5, 0.5, 0.5, 0.5]
    _MAX_VAL = 4500.0   # empirical upper bound for WHUS2 value range

    def __init__(self, args, mode="train", normalization=True, val_ratio=0.15, seed=42):
        self.args = args
        self.mode = mode
        self.normalization = normalization
        self.RandomFlipOrRotate = RandomFlipOrRotate()

        img_dir = os.path.join(args.root, mode, args.cloudy)
        lbl_dir = os.path.join(args.root, mode, args.label)

        # if val dir is empty, auto-split from train
        if mode == "val" and (not os.path.exists(img_dir) or len(os.listdir(img_dir)) == 0):
            all_imgs, all_lbls = self._get_path_pairs(
                os.path.join(args.root, "train", args.cloudy),
                os.path.join(args.root, "train", args.label),
            )
            rng = random.Random(seed)
            indices = list(range(len(all_imgs)))
            rng.shuffle(indices)
            split = int(len(indices) * (1 - val_ratio))
            val_idx = indices[split:]
            self.image_paths = [all_imgs[i] for i in val_idx]
            self.label_paths  = [all_lbls[i]  for i in val_idx]
            print(f"[ImageDataset] val dir empty -> split {len(self.image_paths)} samples from train")
        elif mode == "train" and os.path.exists(os.path.join(args.root, "val", args.cloudy)) \
                and len(os.listdir(os.path.join(args.root, "val", args.cloudy))) == 0:
            # when val is empty, train only uses first (1-val_ratio) portion to avoid data leakage
            all_imgs, all_lbls = self._get_path_pairs(img_dir, lbl_dir)
            rng = random.Random(seed)
            indices = list(range(len(all_imgs)))
            rng.shuffle(indices)
            split = int(len(indices) * (1 - val_ratio))
            train_idx = indices[:split]
            self.image_paths = [all_imgs[i] for i in train_idx]
            self.label_paths  = [all_lbls[i]  for i in train_idx]
            print(f"[ImageDataset] train subset: {len(self.image_paths)} samples (val_ratio={val_ratio})")
        else:
            self.image_paths, self.label_paths = self._get_path_pairs(img_dir, lbl_dir)

        self.length = len(self.image_paths)

    def _get_path_pairs(self, img_dir: str, lbl_dir: str):
        img_names = sorted(os.listdir(img_dir))
        img_names = [n for n in img_names if n.lower().endswith(self.args.file_suffix)]

        lbl_map = {}
        for name in os.listdir(lbl_dir):
            stem = os.path.splitext(name)[0]
            lbl_map[stem] = os.path.join(lbl_dir, name)

        image_paths, label_paths = [], []
        for img_name in img_names:
            stem = os.path.splitext(img_name)[0]
            if stem in lbl_map:
                image_paths.append(os.path.join(img_dir, img_name))
                label_paths.append(lbl_map[stem])

        assert len(image_paths) > 0, \
            f"No matched image-label pairs found in {img_dir} / {lbl_dir}"
        return image_paths, label_paths

    def __getitem__(self, index):
        i = index % self.length

        # read 4-band uint16 TIF, original layout (H, W, 4) -> convert to (4, H, W)
        arr = tifffile.imread(self.image_paths[i])
        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            arr = arr.transpose(2, 0, 1)          # (H, W, C) -> (C, H, W)
        image = torch.from_numpy(
            arr.astype(np.float32) / self._MAX_VAL
        ).clamp(0.0, 1.0)  # [4, H, W], range [0, 1]

        # read label (single-band TIF, values 0/255)
        lbl = tifffile.imread(self.label_paths[i])
        if lbl.ndim == 3:
            lbl = lbl[0] if lbl.shape[0] == 1 else lbl[:, :, 0]
        label = torch.from_numpy(
            (lbl >= 128).astype(np.int64)
        ).unsqueeze(0)  # [1, H, W]

        # data augmentation
        if self.mode == 'train':
            image, label = self.RandomFlipOrRotate([image, label])

        # normalization (4 channels)
        if self.normalization:
            image = TF.normalize(image, self._MEAN, self._STD)

        label = label.squeeze(0)   # [H, W]
        return image, label

    def __len__(self):
        return self.length


def send_to_device(tensor, device):
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)


class ForeverDataIterator:
    """A data iterator that will never stop producing data."""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)
