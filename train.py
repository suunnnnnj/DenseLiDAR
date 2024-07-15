import torch
import cv2
import os
import numpy as np
from torch.nn import Module

from Submodules.DCU.submodels.depthCompletionNew_blockN import depthCompletionNew_blockN, maskFt
from Submodules.DCU.submodels.total_loss import total_loss
from Submodules.data_rectification import rectify_depth
from Submodules.ip_basic.depth_completion import ip_basic
from denselidar import tensor_transform
from dataloader.dataLoader import KITTIDepthDataset, ToTensor

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        annotated_image = data['annotated_image'].cuda()
        velodyne_image = data['velodyne_image'].cuda()
        raw_image = data['raw_image'].cuda()
        targets = annotated_image

        optimizer.zero_grad()

        dense_pseudo_depth, residual_depth_prediction = model(raw_image, velodyne_image, "pseudo depth map")  # sparse_path_placeholder는 실제 경로로 교체해야 합니다.
        dense_pseudo_depth = torch.tensor(dense_pseudo_depth).unsqueeze(0).unsqueeze(0).cuda()  # (H, W) -> (1, 1, H, W)
        residual_depth_prediction =  torch.tensor(residual_depth_prediction).unsqueeze(0).unsqueeze(0).cuda() 
        #dense_target = pseudo gt map
        loss = criterion(dense_target, targets, dense_pseudo_depth)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.4f}")
            running_loss = 0.0

    return loss

#validation 수정 예정
def validate(model, val_loader, criterion, epoch):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            annotated_image = data['annotated_image'].cuda()
            velodyne_image = data['velodyne_image'].cuda()
            raw_image = data['raw_image'].cuda()

            inputs = torch.cat((raw_image, velodyne_image), dim=1)
            targets = annotated_image

            outputs = model(inputs, velodyne_image, "sparse_path_placeholder")  # sparse_path_placeholder는 실제 경로로 교체해야 합니다.
            outputs = torch.tensor(outputs).unsqueeze(0).unsqueeze(0).cuda()  # (H, W) -> (1, 1, H, W)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    print(f"Epoch {epoch + 1} validation loss: {val_loss / len(val_loader):.4f}")

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
    criterion = total_loss() #loss 수정 예정
    optimizer = optim.Adam(model.parameters(), lr=0.001) #optimizer 수정 예정

    
    num_epochs = 20
    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, epoch)
        validate(model, val_loader, criterion, epoch)

    print('Training Finished')

if __name__ == '__main__':
    main()