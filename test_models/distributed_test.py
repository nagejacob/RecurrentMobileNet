import sys
sys.path.append('..')
import argparse
from datasets.patch_dataset import aug, unaug
import importlib
import numpy as np
import os
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

def main(rank, world_size, args, output):
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)
    content = open(os.path.join(args.dataset_dir, 'burst_raw/competition_test_input.0.2.bin'), 'rb').read()
    input = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))

    net = importlib.import_module(args.model).Predictor().to(torch.device('cuda:%d' % rank))
    state_dict = torch.load(args.load_from, map_location=torch.device('cuda:%d' % rank))
    net.load_state_dict(state_dict)
    model = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank], output_device=rank)
    model.eval()

    if rank == 0:
        pbar = tqdm(total=len(input) // world_size)

    for i in range(0, len(input)):
        if i % world_size != rank:
            continue

        inp = input[i, :, :]
        oup = np.zeros_like(inp, dtype=np.float32)
        for j in range(8):
            flip_h = (j % 2 == 1)
            flip_w = (j % 4 > 1)
            transpose = (j > 3)
            aug_inp = aug(inp, flip_h, flip_w, transpose)
            aug_inp = torch.from_numpy(np.float32(aug_inp[None, None, :, :]) * np.float32(1 / 65536))
            aug_inp = aug_inp.to(torch.device('cuda:%d' % rank))
            model.eval()
            with torch.no_grad():
                pred = model(aug_inp)
            pred = pred.detach().cpu().numpy()
            pred = pred[0, 0, :, :]
            unaug_pred = unaug(pred, flip_h, flip_w, transpose)
            oup += unaug_pred[:, :]
        oup = oup / 8
        oup = (oup * 65536).clip(0, 65535)
        output[i, :, :] = torch.from_numpy(oup)

        if rank == 0:
            pbar.update(1)

    if rank == 0:
        pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test the denoiser")

    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--port", type=str, default='12300')
    parser.add_argument("--model", type=str, default='models.recurrent_mobilenet')
    parser.add_argument("--load_from", type=str, default='/mnt/disk10T/Documents/codes/RecurrentMobileNet/model_pytorch.pth')
    parser.add_argument("--dataset_dir", type=str, default='/mnt/disk10T/Documents/datasets/MegCup')
    argspar = parser.parse_args()
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))

    content = open(os.path.join(argspar.dataset_dir, 'burst_raw/competition_test_input.0.2.bin'), 'rb').read()
    input = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))
    output = torch.empty([len(input), 256, 256], dtype=torch.float32)
    output.share_memory_()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = argspar.port
    mp.spawn(main,
             args=(argspar.world_size, argspar, output),
             nprocs=argspar.world_size,
             join=True)

    validate_output = output.numpy()
    fout = open('result.bin', 'wb')
    for i in range(len(input)):
        oup = validate_output[i].astype('uint16')
        fout.write(oup.tobytes())