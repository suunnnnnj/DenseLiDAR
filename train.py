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
from Submodules.data_rectification import rectify_depth
from Submodules.ip_basic.depth_completion import ip_basic
from denselidar import tensor_transform
from dataloader.dataLoader import KITTIDepthDataset, ToTensor
from model import DenseLiDAR


def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        annotated_image = data['annotated_image'].cuda()
        velodyne_image = data['velodyne_image'].cuda()
        raw_image = data['raw_image'].cuda()
        targets = annotated_image

        optimizer.zero_grad()

        dense_pseudo_depth, residual_depth_prediction = model(raw_image, velodyne_image,
                                                              "pseudo depth map")  # sparse_path_placeholder는 실제 경로로 교체해야 합니다.
        dense_pseudo_depth = dense_pseudo_depth.unsqueeze(1).cuda()  # (B, H, W) -> (B, 1, H, W)
        residual_depth_prediction = residual_depth_prediction.unsqueeze(1).cuda()  # (B, H, W) -> (B, 1, H, W)
        # dense_target = pseudo gt map
        loss = criterion(dense_target, targets, dense_pseudo_depth)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.4f}")
            running_loss = 0.0


# validation 수정 예정
def validate(model, val_loader, criterion, epoch):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            annotated_image = data['annotated_image'].cuda()
            velodyne_image = data['velodyne_image'].cuda()
            raw_image = data['raw_image'].cuda()

            dense_pseudo_depth, residual_depth_prediction = model(raw_image, velodyne_image, "pseudo depth map")

            dense_pseudo_depth = dense_pseudo_depth.unsqueeze(1).cuda()  # (B, H, W) -> (B, 1, H, W)
            residual_depth_prediction = residual_depth_prediction.unsqueeze(1).cuda()  # (B, H, W) -> (B, 1, H, W)

            # dense_target = pseudo gt map

            loss = criterion(dense_target, annotated_image, residual_depth_prediction)
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
    root_dir = 'path_to_kitti_dataset'

    # Dataset과 DataLoader 설정
    train_transform = transforms.Compose([
        ToTensor()
    ])

    train_dataset = KITTIDepthDataset(root_dir=root_dir, mode='train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    val_dataset = KITTIDepthDataset(root_dir=root_dir, mode='val', transform=train_transform)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # define model, loss function, optimizer
    model = DenseLiDAR(bs=4).cuda()
    criterion = total_loss()  # loss 수정 예정
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # optimizer 수정 예정

    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    best_optimizer_state = None

    num_epochs = 20
    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, epoch)
        val_loss = validate(model, val_loader, criterion, epoch)

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