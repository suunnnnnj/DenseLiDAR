import torch
import os
from torch.nn import Module

from Submodules.DCU.submodels.depthCompletionNew_blockN import depthCompletionNew_blockN
from Submodules.data_rectification import rectify_depth
from Submodules.tensor_transform import tensor_transform
from sample_dataloader.dataLoader import dataloader
from Submodules.custom_ip import interpolate_depth_map

class DenseLiDAR(Module):
    def __init__(self, bs):
        super().__init__()
        self.bs = bs
        self.processing = interpolate_depth_map
        self.rectification = rectify_depth
        self.DCU = depthCompletionNew_blockN(bs)

    def forward(self, image, sparse, device):
        # sparse = torch.tensor(sparse).to(device).squeeze()
        pseudo_depth_map = self.processing(sparse)
        rectified_depth = self.rectification(sparse.to(device), pseudo_depth_map.to(device))
        dense, attention = self.DCU(image.to(device), pseudo_depth_map.to(device), rectified_depth.to(device))

        residual = dense - sparse
        final_dense_depth = pseudo_depth_map.to(device) + residual

        return final_dense_depth

def set_data(device):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Specify your data path here
    images, lidars, gt = dataloader(current_dir)

    image_path = images[0]
    sparse_path = lidars[0]
    output_path = 'result/sample1.png'  # Output image
    sparse, image = tensor_transform(sparse_path, image_path)
    image = image.to(device)
    sparse = sparse.to(device)

    return sparse, image, sparse_path

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DenseLiDAR(bs=1).to(device)

    sparse, image, sparse_path = set_data(device)

    # 모델 forward pass
    with torch.no_grad():
        final_dense_depth = model(image, sparse, sparse_path, device)

    # 확인을 위한 결과 시각화
    import matplotlib.pyplot as plt
    plt.imshow(final_dense_depth.cpu().squeeze(), 'gray')
    plt.show()
