import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import argparse
import numpy as np
from PIL import Image
from skimage import transform
from Submodules.loss.total_loss import total_loss
from Submodules.custom_ip import interpolate_depth_map
from dataloader.dataLoader import KITTIDepthDataset, ToTensor
from model import DenseLiDAR

parser = argparse.ArgumentParser(description='deepCompletion')
parser.add_argument('--datapath', default='', help='datapath')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=1, help='number of batch size to train')
parser.add_argument('--gpu_nums', type=int, default=1, help='number of gpus to train')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--world_size', type=int, default=1, help='number of processes for DDP')
args = parser.parse_args()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.manual_seed(args.seed)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Dataset과 DataLoader 설정
    root_dir = 'sample/'

    train_transform = transforms.Compose([
        ToTensor()
    ])

    train_dataset = KITTIDepthDataset(root_dir=root_dir, mode='train', transform=train_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=8, pin_memory=True, sampler=train_sampler)

    val_dataset = KITTIDepthDataset(root_dir=root_dir, mode='val', transform=train_transform)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_loader = DataLoader(val_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=8, pin_memory=True, sampler=val_sampler)

    model = DenseLiDAR(int(args.batch_size / args.gpu_nums)).to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    best_optimizer_state = None

    num_epochs = args.epochs
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} [Training]")):
            annotated_image = data['annotated_image'].to(device)
            velodyne_image = data['velodyne_image'].to(device)
            raw_image = data['raw_image'].to(device)
            targets = annotated_image.to(device)

            pseudo_gt_map = interpolate_depth_map(targets)

            optimizer.zero_grad()

            dense_pseudo_depth = model(raw_image, velodyne_image)
            dense_pseudo_depth = dense_pseudo_depth.to(device)
            dense_target = pseudo_gt_map.clone().detach().to(device)

            loss = total_loss(dense_target, targets, dense_pseudo_depth)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1} training loss: {running_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader, desc=f"Epoch {epoch + 1} [Validation]")):
                annotated_image = data['annotated_image'].to(device)
                velodyne_image = data['velodyne_image'].to(device)
                raw_image = data['raw_image'].to(device)
                targets = annotated_image.to(device)

                pseudo_gt_map = interpolate_depth_map(targets)
                dense_pseudo_depth = model(raw_image, velodyne_image)
                dense_pseudo_depth = dense_pseudo_depth.to(device)
                dense_target = pseudo_gt_map.clone().detach().to(device)

                loss = total_loss(dense_target, targets, dense_pseudo_depth)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} validation loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
            best_optimizer_state = optimizer.state_dict()

    if rank == 0:
        save_path = 'best_model.tar'
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': best_optimizer_state,
        }, save_path)
        print(f'Best model saved at epoch {best_epoch} with validation loss: {best_val_loss:.4f}')
        print('Training Finished')

    cleanup()

def main():
    world_size = args.world_size
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()