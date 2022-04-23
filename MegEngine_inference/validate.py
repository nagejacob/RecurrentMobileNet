import sys
sys.path.append('..')
import argparse
import megengine as mge
import megengine.module as M
import numpy as np
import os
from recurrent_mobilenet import Predictor
from tqdm import tqdm
from MegEngine_inference.utils import aug, unaug

class ValidateEnsemble(M.Module):
    def __init__(self, path):
        super(ValidateEnsemble, self).__init__()
        content = open(os.path.join(path, 'burst_raw/competition_train_input.0.2.bin'), 'rb').read()
        input = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))
        content = open(os.path.join(path, 'burst_raw/competition_train_gt.0.2.bin'), 'rb').read()
        gt = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))

        self.input = input[7168:]
        self.gt = gt[7168:]

    def forward(self, model):
        output = np.empty_like(self.gt)
        for i in tqdm(range(0, len(self.gt))):
            inp = self.input[i, :, :]
            oup = np.zeros_like(inp, dtype=np.float32)
            for j in range(8):
                flip_h = (j % 2 == 1)
                flip_w = (j % 4 > 1)
                transpose = (j > 3)
                aug_inp = aug(inp, flip_h, flip_w, transpose)
                aug_inp = mge.tensor(np.float32(aug_inp[None, None, :, :]) * np.float32(1 / 65536))
                pred = model(aug_inp)
                pred = pred.numpy()
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
        return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate the denoiser")
    parser.add_argument("--path", type=str, default='/hdd/Documents/datasets/MegCup')
    parser.add_argument("--model_file", type=str, default='model_megengine.pth')
    argspar = parser.parse_args()

    model = Predictor()
    model.load_state_dict(mge.load(argspar.model_file))
    assert model.unroll == 48

    validate = ValidateEnsemble(argspar.path)
    print(validate(model))