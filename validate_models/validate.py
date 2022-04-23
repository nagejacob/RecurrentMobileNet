import sys
sys.path.append('..')
import argparse
from datasets.patch_dataset import aug, unaug
import importlib
import numpy as np
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.count_params import count_params

class Validate(nn.Module):
    def __init__(self, path):
        super(Validate, self).__init__()
        content = open(os.path.join(path, 'burst_raw/competition_train_input.0.2.bin'), 'rb').read()
        input = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))
        content = open(os.path.join(path, 'burst_raw/competition_train_gt.0.2.bin'), 'rb').read()
        gt = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))

        self.input = input[7168:]
        self.gt = gt[7168:]

    def forward(self, model, with_psnr=False):
        output = np.empty_like(self.gt)
        for i in tqdm(range(0, len(self.gt))):
            inp = self.input[i, :, :]
            oup = np.zeros_like(inp, dtype=np.float32)
            for j in range(8):
                flip_h = (j % 2 == 1)
                flip_w = (j % 4 > 1)
                transpose = (j > 3)
                aug_inp = aug(inp, flip_h, flip_w, transpose)
                aug_inp = torch.from_numpy(np.float32(aug_inp[None, None, :, :]) * np.float32(1 / 65536))
                aug_inp = aug_inp.cuda()
                model.eval()
                with torch.no_grad():
                    pred = model(aug_inp)
                pred = pred.detach().cpu().numpy()
                pred = pred[0, 0, :, :]
                unaug_pred = unaug(pred, flip_h, flip_w, transpose)
                oup += unaug_pred[:, :]
            oup = oup / 8
            oup = (oup * 65536).clip(0, 65535).astype('uint16')
            output[i] = oup

        validate_gt = np.float32(self.gt)
        validate_output = np.float32(output)
        means = validate_gt.mean(axis=(1, 2))
        weight = (1 / means) ** 0.5
        diff = np.abs(validate_output - validate_gt).mean(axis=(1, 2))
        diff = diff * weight
        score = diff.mean()
        score = np.log10(100 / score) * 5

        if with_psnr:
            diff = np.mean((validate_output - validate_gt) ** 2, axis=(1, 2))
            psnr = 20 * np.log10(65535 / np.sqrt(diff))
            psnr = np.mean(psnr)
            return score, psnr
        else:
            return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate the denoiser")

    parser.add_argument("--model", type=str, default='models.recurrent_mobilenet')
    parser.add_argument("--load_from", type=str, default='/mnt/disk10T/Documents/codes/RecurrentMobileNet/model_pytorch.pth')
    parser.add_argument("--dataset_dir", type=str, default='/mnt/disk10T/Documents/datasets/MegCup')
    argspar = parser.parse_args()
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))

    validate = Validate(argspar.dataset_dir)
    model = importlib.import_module(argspar.model).Predictor().cuda()
    assert count_params(model) < 100000
    model.load_state_dict(torch.load(argspar.load_from))
    model.eval()

    print(validate(model, with_psnr=True))