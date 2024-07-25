from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
from test_dataloader import test_dataLoader as lsn
from test_dataloader import test_trainLoader as DA
from model import DenseLiDAR
from Submodules.loss.total_loss import total_loss
from tqdm import tqdm
import torch.multiprocessing as mp

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='deepCompletion')
parser.add_argument('--datapath', default='', help='datapath')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
parser.add_argument('--checkpoint', type=int, default=5, help='number of epochs to make a checkpoint')
parser.add_argument('--batch_size', type=int, default=1, help='batch size to train')
parser.add_argument('--gpu_nums', type=int, default=1, help='number of GPUs to train')  # gpu_nums를 world_size로 사용
parser.add_argument('--loadmodel', default='', help='load model')
parser.add_argument('--savemodel', default='my', help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--master_addr', type=str, default='localhost', help='master address for distributed training')
parser.add_argument('--master_port', type=str, default='12355', help='master port for distributed training')
args = parser.parse_args()

def setup(rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size, args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    setup(rank, world_size, args.master_addr, args.master_port)
    
    torch.cuda.set_device(rank)
    
    train_left_img, train_sparse, train_depth, train_pseudo, train_dense = lsn.dataloader(args.datapath, mode='train')
    val_left_img, val_sparse, val_depth, val_pseudo, val_dense = lsn.dataloader(args.datapath, mode='val')

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        DA.myImageFloder(train_left_img, train_sparse, train_depth, train_pseudo, train_dense, True),
        num_replicas=world_size,
        rank=rank
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        DA.myImageFloder(val_left_img, val_sparse, val_depth, val_pseudo, val_dense, True),
        num_replicas=world_size,
        rank=rank
    )

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_left_img, train_sparse, train_depth, train_pseudo, train_dense, True),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=train_sampler
    )

    ValImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(val_left_img, val_sparse, val_depth, val_pseudo, val_dense, True),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=val_sampler
    )

    model = DenseLiDAR(args.batch_size).to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7, last_epoch=-1)

    def save_model(model, optimizer, epoch, path):
        os.makedirs(os.path.dirname('checkpoint/'), exist_ok=True)
        if rank == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, path)
            print(f'Checkpoint saved at: {path}\n')

    def train(inputl, gt1, sparse, pseudo, dense, params):
        device = 'cuda'
        model.train()
        inputl = Variable(torch.FloatTensor(inputl))
        gt1 = Variable(torch.FloatTensor(gt1))
        sparse = Variable(torch.FloatTensor(sparse))
        pseudo = Variable(torch.FloatTensor(pseudo))
        dense = Variable(torch.FloatTensor(dense))
        params = Variable(torch.FloatTensor(params))
        if args.cuda:
            inputl, gt1, sparse, pseudo, dense, params = inputl.cuda(), gt1.cuda(), sparse.cuda(), pseudo.cuda(), dense.cuda(), params.cuda()
        optimizer.zero_grad()

        dense_depth = model(inputl, sparse, pseudo, device)

        t_loss, s_loss, d_loss = total_loss(dense, gt1, dense_depth)
        t_loss.backward()
        optimizer.step()

        return t_loss, s_loss, d_loss

    def validate(inputl, gt1, sparse, pseudo, dense, params):
        device = 'cuda'
        model.eval()
        with torch.no_grad():
            inputl = Variable(torch.FloatTensor(inputl))
            gt1 = Variable(torch.FloatTensor(gt1))
            sparse = Variable(torch.FloatTensor(sparse))
            pseudo = Variable(torch.FloatTensor(pseudo))
            dense = Variable(torch.FloatTensor(dense))
            params = Variable(torch.FloatTensor(params))
            if args.cuda:
                inputl, gt1, sparse, pseudo, dense, params = inputl.cuda(), gt1.cuda(), sparse.cuda(), pseudo.cuda(), dense.cuda(), params.cuda()

            dense_depth = model(inputl, sparse, pseudo, device)
            t_loss, s_loss, d_loss = total_loss(dense, gt1, dense_depth)

        return t_loss, s_loss, d_loss

    torch.cuda.empty_cache()
    start_full_time = time.time()

    for epoch in range(1, args.epochs + 1):
        total_train_loss = 0
        total_val_loss = 0

        TrainImgLoader.sampler.set_epoch(epoch)

        ## training ##
        for batch_idx, (imgL_crop, input_crop1, sparse2, pseudo_crop, dense_crop, params) in tqdm(
                enumerate(TrainImgLoader), total=len(TrainImgLoader), desc=f"Epoch {epoch}"):  # rawimage, gtlidar,rawlidar,pseudo_depth,gt_depth,param

            loss, loss1, loss2 = train(imgL_crop, input_crop1, sparse2, pseudo_crop, dense_crop, params)
            total_train_loss += loss

        print('epoch %d total training loss = %.10f' % (epoch, total_train_loss / len(TrainImgLoader)))

        ## validation ##
        for batch_idx, (imgL_crop, input_crop1, sparse2, pseudo_crop, dense_crop, params) in tqdm(
                enumerate(ValImgLoader), total=len(ValImgLoader),
                desc=f"Epoch {epoch}"):  # rawimage, gtlidar,rawlidar,pseudo_depth,gt_depth,param

            loss, loss1, loss2 = validate(imgL_crop, input_crop1, sparse2, pseudo_crop, dense_crop, params)
            total_val_loss += loss

        print('epoch %d total validation loss = %.10f' % (epoch, total_val_loss / len(ValImgLoader)))

        if epoch % args.checkpoint == 0 and rank == 0:
            save_path = f'checkpoint/epoch-{epoch}_loss-{total_val_loss / len(ValImgLoader):.3f}.tar'
            save_model(model, optimizer, epoch, save_path)

    print('full finetune time = %.2f HR' % ((time.time() - start_full_time) / 3600))

    cleanup()

if __name__ == '__main__':
    world_size = args.gpu_nums
    mp.spawn(main_worker,
             args=(world_size, args),
             nprocs=world_size,
             join=True)
