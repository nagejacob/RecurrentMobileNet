import sys
sys.path.append('..')
import importlib
import os
import megengine as mge
import torch

pytorch_path = '/hdd/Documents/codes/RecurrentMobileNet/model_pytorch.pth'
megengine_path = '/hdd/Documents/codes/RecurrentMobileNet/MegEngine_inference/model_megengine.pth'

if __name__ == '__main__':
    state_dict = torch.load(pytorch_path, map_location=torch.device('cpu'))
    pytorch_weights = {k: v.numpy() for k, v in state_dict.items()}

    megengine_weights = {}
    print('pytorch params')
    for k, v in pytorch_weights.items():
        print(k, v.shape)
        # for group convs
        if 'conv.2.weight' in k:
            v = v.reshape(v.shape[0], 1, 1, v.shape[2], v.shape[2])
        megengine_weights[k] = v

    model = importlib.import_module('MegEngine_inference.recurrent_mobilenet').Predictor()
    print('megengine params')
    for k, v in model.state_dict().items():
        print(k, v.shape)

    model.load_state_dict(megengine_weights, strict=False)
    fout = open(megengine_path, 'wb')
    mge.save(model.state_dict(), fout)
    fout.close()