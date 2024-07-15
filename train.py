import torch
import cv2
import os
import numpy as np
from torch import optim
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from Submodules.DCU.submodels.depthCompletionNew_blockN import depthCompletionNew_blockN, maskFt
from Submodules.DCU.submodels.total_loss import total_loss
from Submodules.DCU.submodels.L1_Structural_loss import l_structural 
from Submodules.DCU.submodels.L2_depth_loss import L2_depth_loss
from Submodules.data_rectification import rectify_depth
from Submodules.ip_basic.depth_completion import ip_basic
from denselidar import tensor_transform
from dataloader.dataLoader import KITTIDepthDataset, ToTensor
from model import DenseLiDAR

def train(model, train_loader, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        annotated_image = data['annotated_image'].to(device)
        velodyne_image = data['velodyne_image'].to(device)
        raw_image = data['raw_image'].to(device)
        targets = annotated_image

        # targets를 numpy 배열로 변환
        targets_np = targets.cpu().numpy()

        # 3채널 이미지를 그레이스케일로 변환
        targets_np_gray = cv2.cvtColor(targets_np[0].transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
        # np.float32로 변환
        print("train pseudo gt map")
        pseudo_gt_map = ip_basic(np.float32(targets_np_gray / 256.0))

        optimizer.zero_grad()

        dense_pseudo_depth, pseudo_depth_map = model(raw_image, velodyne_image, device) 
        dense_pseudo_depth = dense_pseudo_depth.to(device)  # (B, H, W) -> (B, 1, H, W)
        dense_target = pseudo_gt_map.clone().detach().to(device)  # GPU로 이동
        print(f"VType: {dense_pseudo_depth.dtype}, VShape: {dense_pseudo_depth.shape}")
        print(f"VType: {dense_target.dtype}, VShape: {dense_target.shape}")

        loss = total_loss(dense_target, targets, dense_pseudo_depth)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch + 1} training loss: {running_loss / len(train_loader):.4f}")
    


def validate(model, val_loader, epoch, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            annotated_image = data['annotated_image'].to(device)
            velodyne_image = data['velodyne_image'].to(device)
            raw_image = data['raw_image'].to(device)

            # targets를 numpy 배열로 변환
            targets_np = annotated_image.cpu().numpy()

            # 3채널 이미지를 그레이스케일로 변환
            targets_np_gray = cv2.cvtColor(targets_np[0].transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
            # np.float32로 변환
            print("val pseudo gt map")
            pseudo_gt_map = ip_basic(np.float32(targets_np_gray / 256.0))
            print("val pseudo depth map")
            pseudo_depth_map = ip_basic(velodyne_image.cpu().numpy().squeeze())
            
            dense_pseudo_depth = model(raw_image, velodyne_image, device)
            
            dense_pseudo_depth = dense_pseudo_depth.unsqueeze(1).to(device)  # (B, H, W) -> (B, 1, H, W)
            
            dense_target = torch.tensor(pseudo_gt_map).to(device)  # GPU로 이동
            
            loss = total_loss(dense_target, annotated_image, dense_pseudo_depth)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch + 1} validation loss: {avg_val_loss:.4f}")
    return avg_val_loss

def save_model(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def main():
    # Paths
    root_dir = 'datasets/'

    # Dataset과 DataLoader 설정
    train_transform = transforms.Compose([
        ToTensor()
    ])

    train_dataset = KITTIDepthDataset(root_dir=root_dir, mode='train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)

    val_dataset = KITTIDepthDataset(root_dir=root_dir, mode='val', transform=train_transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # define model, loss function, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DenseLiDAR(bs=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    best_optimizer_state = None

    num_epochs = 20
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
