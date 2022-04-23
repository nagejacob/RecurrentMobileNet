import sys
sys.path.append('..')
import argparse
import megengine as mge
import megengine.module as M
from recurrent_mobilenet import Predictor
import numpy as np
import os
from tqdm import tqdm
from MegEngine_inference.utils import aug, unaug

class TestEnsemble(M.Module):
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
                aug_inp = mge.tensor(np.float32(aug_inp[None, None, :, :]) * np.float32(1 / 65536))
                pred = model(aug_inp)
                pred = pred.numpy()
                pred = pred[0, 0, :, :]
                unaug_pred = unaug(pred, flip_h, flip_w, transpose)
                oup += unaug_pred[:, :]
            oup = oup / 8
            oup = (oup * 65536).clip(0, 65535).astype('uint16')
            fout.write(oup.tobytes())

        fout.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the denoiser")
    parser.add_argument("--path", type=str, default='/hdd/Documents/datasets/MegCup')
    parser.add_argument("--model_file", type=str, default='model_megengine.pth')
    argspar = parser.parse_args()

    model = Predictor()
    model.load_state_dict(mge.load(argspar.model_file))
    assert model.unroll == 48

    test = TestEnsemble(argspar.path, 'result.bin')
    test(model)