import torch
import cv2
import os
import numpy as np
from torch.nn import Module

from Submodules.DCU.submodels.depthCompletionNew_blockN import depthCompletionNew_blockN, maskFt
from Submodules.data_rectification import rectify_depth
from Submodules.ip_basic.depth_completion import ip_basic
from Submodules.tensor_transform import tensor_transform, multiply_output
from sample_dataloader.dataLoader import dataloader

class DenseLiDAR(Module):
    def __init__(self,bs):
        super().__init__()
        self.bs = bs
        self.processing = ip_basic
        self.rectification = rectify_depth
        self.DCU = depthCompletionNew_blockN(bs)
        self.mask = maskFt()
        self.mul = multiply_output

    def forward(self, image, sparse, sparse_path):
        pseudo_depth_map = self.processing(sparse_path)
        rectified_depth = self.rectification(sparse, pseudo_depth_map)
        output_normal, output_concat = self.DCU(image.to(device), torch.tensor(pseudo_depth_map).to(device), torch.tensor(rectified_depth).to(device))
        output_concat2_processed = self.mask(output_concat)
        result = self.mul(output_concat2_processed)
        return result

def set_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Specify your data path here
    images, lidars, gt = dataloader(current_dir)

    image_path = images[0]
    sparse_path = lidars[0]
    output_path = 'result/sample1.png'  # Output image
    sparse, image = tensor_transform(sparse_path, image_path)

    return sparse, image, sparse_path

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DenseLiDAR(bs=1).to(device)

    sparse, image, sparse_path = set_data()

    # 모델 forward pass
    with torch.no_grad():
        output = model(image, sparse, sparse_path)

    # 확인을 위한 결과 시각화
    import matplotlib.pyplot as plt
    plt.imshow(output, 'gray')
    plt.show()