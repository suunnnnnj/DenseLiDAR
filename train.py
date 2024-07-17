import argparse

import torch
import argparse
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from Submodules.loss.total_loss import total_loss
from Submodules.custom_ip import interpolate_depth_map
from Submodules.morphology import morphology_torch
from dataloader.dataLoader import KITTIDepthDataset, ToTensor
from model import DenseLiDAR

parser = argparse.ArgumentParser(description='deepCpmpletion')
parser.add_argument('--datapath', default='', help='datapath')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=1, help='number of batch size to train')
parser.add_argument('--gpu_nums', type=int, default=1, help='number of gpu to train')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--morph', default='morphology', metavar='S', help='random seed (default: 1)') #
args = parser.parse_args()
batch_size = int(args.batch_size / args.gpu_nums)

def select_morph(opt):
    if opt == 'morphology':
        f = morphology_torch
    elif opt == 'interpolate':
        f = interpolate_depth_map
    else:
        print("Please type correct function")
    return f

def train(model, train_loader, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} [Training]")):
        annotated_image = data['annotated_image'].to(device)
        velodyne_image = data['velodyne_image'].to(device)
        raw_image = data['raw_image'].to(device)
        targets = annotated_image
        morph = select_morph(args.morph)
        pseudo_gt_map = morph(targets, device)

        optimizer.zero_grad()

        dense_pseudo_depth = model(raw_image, velodyne_image, device)
        dense_pseudo_depth = dense_pseudo_depth.to(device)  # (B, H, W) -> (B, 1, H, W)
        dense_target = pseudo_gt_map.clone().detach().to(device)  # GPU로 이동

        loss = total_loss(dense_target, targets, dense_pseudo_depth)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"\nEpoch {epoch + 1} training loss: {running_loss / len(train_loader):.4f}")

def validate(model, val_loader, epoch, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader, desc=f"Epoch {epoch + 1} [Validation]")):
            annotated_image = data['annotated_image'].to(device)
            velodyne_image = data['velodyne_image'].to(device)
            raw_image = data['raw_image'].to(device)
            targets = annotated_image
            morph = select_morph(args.morph)
            
            pseudo_gt_map = morph(targets, device)
            
            dense_pseudo_depth = model(raw_image, velodyne_image, device)
            
            dense_pseudo_depth = dense_pseudo_depth.to(device)  # (B, H, W) -> (B, 1, H, W)

            dense_target = pseudo_gt_map.clone().detach()

            loss = total_loss(dense_target, annotated_image, dense_pseudo_depth)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"\nEpoch {epoch + 1} validation loss: {avg_val_loss:.4f}")
    return avg_val_loss

def save_model(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def main():
    # Paths
    root_dir = 'sample/'

    # Dataset과 DataLoader 설정
    train_transform = transforms.Compose([
        ToTensor()
    ])

    train_dataset = KITTIDepthDataset(root_dir=root_dir, mode='train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    val_dataset = KITTIDepthDataset(root_dir=root_dir, mode='val', transform=train_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # define model, loss function, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DenseLiDAR(bs=batch_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    best_optimizer_state = None

    num_epochs = 100
    for epoch in range(num_epochs):
        train(model, train_loader, optimizer, epoch, device)
        val_loss = validate(model, val_loader, epoch, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
            best_optimizer_state = optimizer.state_dict()

    # 훈련이 모두 끝난 후 최상의 모델을 저장
    save_path = 'best_model.tar'
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_model_state,
        'optimizer_state_dict': best_optimizer_state,
    }, save_path)

    print(f'Best model saved at epoch {best_epoch} with validation loss: {best_val_loss:.4f}')
    print('Training Finished')

if __name__ == '__main__':
    main()
