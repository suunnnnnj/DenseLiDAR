import argparse
import os
import torch
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import Variable
from torch.optim import AdamW
from dataloader import dataLoader as lsn
from dataloader import trainLoader as DA
from model import DenseLiDAR
from submodules.loss.total_loss import total_loss
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='depth completion')
parser.add_argument('--data_path', default='', help='data path')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
parser.add_argument('--checkpoint', type=int, default=5, help='number of epochs to make a checkpoint')
parser.add_argument('--batch_size', type=int, default=1, help='batch size to train')
parser.add_argument('--gpu_nums', type=int, default=1, help='number of GPUs to train')
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


def save_model(model, optimizer, epoch, path, rank):
    os.makedirs(os.path.dirname('checkpoint/'), exist_ok=True)

    if rank == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, path)

        print(f'Checkpoint saved at: {path}\n')


def train(image, gt, sparse, pseudo_depth_map, pseudo_gt_map, model, optimizer, device, args):
    model.train()

    image = Variable(torch.FloatTensor(image))
    gt = Variable(torch.FloatTensor(gt))
    sparse = Variable(torch.FloatTensor(sparse))
    pseudo_depth_map = Variable(torch.FloatTensor(pseudo_depth_map))
    pseudo_gt_map = Variable(torch.FloatTensor(pseudo_gt_map))

    if args.cuda:
        image, gt, sparse, pseudo_depth_map, pseudo_gt_map = image.cuda(), gt.cuda(), sparse.cuda(), pseudo_depth_map.cuda(), pseudo_gt_map.cuda()

    optimizer.zero_grad()

    dense_depth = model(image, sparse, pseudo_depth_map, device)

    t_loss, s_loss, d_loss = total_loss(pseudo_gt_map, gt, dense_depth)
    t_loss.backward()

    optimizer.step()

    return t_loss, s_loss, d_loss

def validate(image, gt, sparse, pseudo_depth_map, pseudo_gt_map, model, device, args):
    model.eval()
    with torch.no_grad():
        image = Variable(torch.FloatTensor(image))
        gt = Variable(torch.FloatTensor(gt))
        sparse = Variable(torch.FloatTensor(sparse))
        pseudo_depth_map = Variable(torch.FloatTensor(pseudo_depth_map))
        pseudo_gt_map = Variable(torch.FloatTensor(pseudo_gt_map))

        if args.cuda:
            image, gt, sparse, pseudo_depth_map, pseudo_gt_map = image.cuda(), gt.cuda(), sparse.cuda(), pseudo_depth_map.cuda(), pseudo_gt_map.cuda()

        dense_depth = model(image, sparse, pseudo_depth_map, device)

        t_loss, s_loss, d_loss = total_loss(pseudo_gt_map, gt, dense_depth)

    return t_loss, s_loss, d_loss

def main_worker(rank, world_size, args):
    batch_size = int(args.batch_size / args.gpu_nums)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    setup(rank, world_size, args.master_addr, args.master_port)
    torch.cuda.set_device(rank)

    train_image, train_sparse, train_gt, train_pseudo_depth_map, train_pseudo_gt_map = lsn.dataloader(args.data_path, mode='train')
    val_image, val_sparse, val_gt, val_pseudo_depth_map, val_pseudo_gt_map = lsn.dataloader(args.data_path, mode='val')

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        DA.myImageFloder(train_image, train_sparse, train_gt, train_pseudo_depth_map, train_pseudo_gt_map, True),
        num_replicas=world_size,
        rank=rank
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        DA.myImageFloder(val_image, val_sparse, val_gt, val_pseudo_depth_map, val_pseudo_gt_map, True),
        num_replicas=world_size,
        rank=rank
    )

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_image, train_sparse, train_gt, train_pseudo_depth_map, train_pseudo_gt_map, True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )

    ValImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(val_image, val_sparse, val_gt, val_pseudo_depth_map, val_pseudo_gt_map, True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=True
    )

    model = DenseLiDAR(batch_size).to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7, last_epoch=-1)

    torch.cuda.empty_cache()

    start_full_time = time.time()

    for epoch in range(1, args.epochs + 1):
        total_train_loss = 0
        total_train_s_loss = 0
        total_train_d_loss = 0

        total_val_loss = 0
        total_val_s_loss = 0
        total_val_d_loss =0

        TrainImgLoader.sampler.set_epoch(epoch)

        ## training ##
        for batch_idx, (image, gt, sparse, pseudo_depth_map, pseudo_gt_map) in tqdm(
                enumerate(TrainImgLoader), total=len(TrainImgLoader), desc=f"Epoch {epoch}"):
            train_loss, train_s_loss, train_d_loss = train(image, gt, sparse, pseudo_depth_map, pseudo_gt_map, model, optimizer, rank, args)
            total_train_loss += train_loss
            total_train_s_loss += train_s_loss
            total_train_d_loss += train_d_loss

        print('epoch %d total training loss = %.10f' % (epoch, total_train_loss / len(TrainImgLoader)))
        print('epoch %d structural training loss = %.10f, depth training loss = %.10f' % (epoch, total_train_s_loss / len(TrainImgLoader), total_train_d_loss / len(TrainImgLoader)))
        print()

        ## validation ##
        for batch_idx, (image, gt, sparse, pseudo_depth_map, pseudo_gt_map) in tqdm(
                enumerate(ValImgLoader), total=len(ValImgLoader), desc=f"Epoch {epoch}"):
            val_loss, val_s_loss, val_d_loss = validate(image, gt, sparse, pseudo_depth_map, pseudo_gt_map, model, rank, args)
            total_val_loss += val_loss
            total_val_s_loss += val_s_loss
            total_val_d_loss += val_d_loss

        print('epoch %d total validation loss = %.10f' % (epoch, total_val_loss / len(ValImgLoader)))
        print('epoch %d structural validation loss = %.10f, depth validation loss = %.10f' % (epoch, total_val_s_loss / len(ValImgLoader), total_val_d_loss / len(ValImgLoader)))
        print()

        scheduler.step()

        if epoch % args.checkpoint == 0 and rank == 0:
            save_path = f'checkpoint/epoch-{epoch}_loss-{total_val_loss / len(ValImgLoader):.3f}.tar'
            save_model(model, optimizer, epoch, save_path, rank)

    print('full finetune time = %.2f HR' % ((time.time() - start_full_time) / 3600))
    cleanup()


if __name__ == '__main__':
    world_size = args.gpu_nums
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
