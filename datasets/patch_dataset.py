import random
from torch.utils.data import Dataset
import numpy as np
import os

def crop(img, patch_size, position_h, position_w):
    patch = img[position_h:position_h + patch_size, position_w:position_w + patch_size]
    return patch

def aug(img, flip_h, flip_w, transpose):
    if flip_h:
        img = img[::-1, :]
    if flip_w:
        img = img[:, ::-1]
    if transpose:
        img = img.T
    return img

def unaug(img, flip_h, flip_w, transpose):
    if transpose:
        img = img.T
    if flip_w:
        img = img[:, ::-1]
    if flip_h:
        img = img[::-1, :]
    return img

class TrainDataset(Dataset):
    def __init__(self, path, patch_size=96, mode='train'):
        content = open(os.path.join(path, 'burst_raw/competition_train_input.0.2.bin'), 'rb').read()
        input = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))
        content = open(os.path.join(path, 'burst_raw/competition_train_gt.0.2.bin'), 'rb').read()
        gt = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))

        self.patch_size = patch_size
        self.mode = mode
        if self.mode == 'train':
            self.input = input[:7168]
            self.gt = gt[:7168]
        elif self.mode == 'validate_models':
            self.input = input[7168:]
            self.gt = gt[7168:]
        elif self.mode == 'full':
            self.input = input
            self.gt = gt

    def __getitem__(self, idx):
        flip_h = random.random() > 0.5
        flip_w = random.random() > 0.5
        transpose = random.random() > 0.5
        idx = idx % len(self.input)

        input = self.input[idx]
        gt = self.gt[idx]
        H, W = input.shape
        if H == self.patch_size:
            position_h, position_w = 0, 0
        else:
            position_h = np.random.randint(0, H - self.patch_size)
            position_w = np.random.randint(0, W - self.patch_size)

        input = crop(input, self.patch_size, position_h, position_w)
        gt = crop(gt, self.patch_size, position_h, position_w)

        input = aug(input, flip_h, flip_w, transpose)
        gt = aug(gt, flip_h, flip_w, transpose)

        input = np.float32(input[None, :, :]) * np.float32(1 / 65536)
        gt = np.float32(gt[None, :, :]) * np.float32(1 / 65536)

        return input, gt

    def __len__(self):
        if self.mode == 'validate_models':
            return len(self.input)
        else:
            return 8000000