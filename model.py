# 리팩토링 및 변수 이름 변경 필요.
import torch
import cv2
import os
import numpy as np
from torch.nn import Module

from Submodules.DCU.submodels.depthCompletionNew_blockN import depthCompletionNew_blockN, maskFt
from Submodules.data_rectification import rectify_depth
from Submodules.ip_basic.depth_completion import ip_basic
from denselidar import tensor_transform
from sample_dataloader.dataLoader import dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DenseLiDAR(Module):
    def __init__(self,bs):
        super().__init__()

        self.bs = bs
        self.processing = ip_basic
        self.rectification = rectify_depth
        self.DCU = depthCompletionNew_blockN(bs)
        self.mask = maskFt()

    def forward(self, image, lidar, sparse_path):
        pseudo_depth_map = self.processing(sparse_path)
        rectified_depth = self.rectification(lidar, pseudo_depth_map)

        pseudo_depth = torch.tensor(pseudo_depth_map).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        output_normal, output_concat = self.DCU(image.to(device), torch.tensor(pseudo_depth).to(device), torch.tensor(rectified_depth).to(device))
        output_concat2_processed = self.mask(output_concat)
        multiplied_output = output_normal * output_concat2_processed
        multiplied_output_np = multiplied_output.squeeze().detach().cpu().numpy()
        multiplied_output_np = cv2.normalize(multiplied_output_np, None, 0, 255, cv2.NORM_MINMAX)
        return multiplied_output_np


if __name__ == "__main__":
    # 테스트 입력 데이터 생성
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Specify your data path here
    images, lidars, gt = dataloader(current_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DenseLiDAR(bs=1).to(device)

    left_image_path = images[0]
    sparse_depth_path = lidars[0]
    output_path = 'result/sample1.png'  # Output image
    gt_path = gt[0]

    sparse_depth_np = cv2.imread(sparse_depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    sparse_depth = torch.tensor(sparse_depth_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    left_image_np = cv2.imread(left_image_path).astype(np.float32)
    left_image = torch.tensor(left_image_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    # 모델 forward pass
    with torch.no_grad():
        output = model(left_image, sparse_depth, sparse_depth_path)

    import matplotlib.pyplot as plt
    plt.imshow(output, 'gray')
    plt.show()