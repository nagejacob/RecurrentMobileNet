import sys
sys.path.append('..')
import argparse
import importlib
import os
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.io import date_time, log

def main(rank, world_size, args):
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)

    trainset = importlib.import_module(args.dataset).TrainDataset(path=args.path, patch_size=args.patch_size, mode='full')
    trainsampler = DistributedSampler(dataset=trainset, shuffle=True)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, sampler=trainsampler, num_workers=4, drop_last=True)
    if rank == 0:
        validate = importlib.import_module(args.validate_models).Validate(path=args.path)
    net = importlib.import_module(args.model).Predictor().to(torch.device('cuda:%d' % rank))
    model = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank], output_device=rank)
    optimizer = torch.optim.Adam(model.module.parameters(), lr=2e-4)

    iter = 0
    while True:
        for input, gt in trainloader:
            for g in optimizer.param_groups:
                g['lr'] = 2e-4 * (args.num_iters - iter) / args.num_iters

            input = input.to(torch.device('cuda:%d' % rank))
            gt = gt.to(torch.device('cuda:%d' % rank))
            weight = (1 / torch.mean(gt, dim=[1, 2, 3], keepdim=True)) ** 0.5 / 10

            output = model(input)
            loss = F.l1_loss(output, gt, reduction='none') * weight
            loss = torch.mean(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter += 1
            if rank == 0 and iter % args.print_every == 0:
                log(args.log_file, '%s, iter: %d, loss: %f\n' % (date_time(), iter, loss.item()))

            if rank == 0 and iter % args.val_every == 0:
                score, psnr = validate(model, with_psnr=True)
                log(args.log_file, '%s, iter: %d, score: %f, psnr: %f\n' % (date_time(), iter, score, psnr))
            if rank == 0 and iter % args.save_every == 0:
                fout = open(os.path.join(args.log_dir, 'iter_%d.pth' % iter), 'wb')
                torch.save(model.module.state_dict(), fout)
                fout.close()
            if iter > args.num_iters:
                if rank == 0:
                    fout = open(os.path.join(args.log_dir, 'model.pth'), 'wb')
                    torch.save(model.module.state_dict(), fout)
                    fout.close()
                torch.distributed.destroy_process_group()
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the denoiser")

    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--port", type=str, default='12300')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dataset", type=str, default='datasets.patch_dataset')
    parser.add_argument("--validate_models", type=str, default='validate_models.validate')
    parser.add_argument("--model", type=str, default='models.recurrent_mobilenet')
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--print_every", type=int, default=10000)
    parser.add_argument("--save_every", type=int, default=100000)
    parser.add_argument("--val_every", type=int, default=10000)
    parser.add_argument("--num_iters", type=int, default=4000000)
    parser.add_argument("--path", type=str, default='/mnt/disk10T/Documents/datasets/MegCup')
    parser.add_argument("--log_dir", type=str, default="/mnt/disk10T/Documents/codes/RecurrentMobileNet/logs/recurrent_mobilenet")

    argspar = parser.parse_args()
    argspar.log_file = os.path.join(argspar.log_dir, 'log.out')
    argspar.batch_size = argspar.batch_size // argspar.world_size

    if not os.path.exists(argspar.log_dir):
        os.makedirs(argspar.log_dir)
    log(argspar.log_file, "\n### Training the denoiser ###\n")
    log(argspar.log_file, "> Parameters:\n")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        log(argspar.log_file, '\t{}: {}\n'.format(p, v))

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = argspar.port
    mp.spawn(main,
             args=(argspar.world_size, argspar),
             nprocs=argspar.world_size,
             join=True)