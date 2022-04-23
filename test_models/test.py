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

class TestEnsemble(nn.Module):
    def __init__(self, path, out_path):
        super(TestEnsemble, self).__init__()
        content = open(os.path.join(path, 'burst_raw/competition_test_input.0.2.bin'), 'rb').read()
        input = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))

        self.input = input
        self.out_path = out_path

    def forward(self, model):
        fout = open(self.out_path, 'wb')
        for i in tqdm(range(0, len(self.input))):
            inp = self.input[i, :, :]
            oup = np.zeros_like(inp, dtype=np.float32)
            for j in range(8):
                flip_h = (j % 2 == 1)
                flip_w = (j % 4 > 1)
                transpose = (j > 3)
                aug_inp = aug(inp, flip_h, flip_w, transpose)
                aug_inp = torch.from_numpy(np.float32(aug_inp[None, None, :, :]) * np.float32(1 / 65536))
                aug_inp = aug_inp.cuda()
                with torch.no_grad():
                    pred = model(aug_inp)
                pred = pred.detach().cpu().numpy()
                pred = pred[0, 0, :, :]
                unaug_pred = unaug(pred, flip_h, flip_w, transpose)
                oup += unaug_pred[:, :]
            oup = oup / 8
            oup = (oup * 65536).clip(0, 65535).astype('uint16')
            fout.write(oup.tobytes())

        fout.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test the denoiser")

    parser.add_argument("--model", type=str, default='models.recurrent_mobilenet')
    parser.add_argument("--load_from", type=str, default='/mnt/disk10T/Documents/codes/RecurrentMobileNet/model_pytorch.pth')
    parser.add_argument("--dataset_dir", type=str, default='/mnt/disk10T/Documents/datasets/MegCup')
    parser.add_argument("--output", type=str, default='result.bin')
    argspar = parser.parse_args()
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))

    model = importlib.import_module(argspar.model).Predictor().cuda()
    assert count_params(model) < 100000
    model.load_state_dict(torch.load(argspar.load_from))
    model.eval()

    test = TestEnsemble(argspar.dataset_dir, argspar.output)
    test(model)