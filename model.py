import torch
import os
from torch.nn import Module

from Submodules.DCU.submodels.depthCompletionNew_blockN import depthCompletionNew_blockN
from Submodules.data_rectification import rectify_depth
from Submodules.tensor_transform import tensor_transform
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
