from torch.nn import Module

from Submodules.DCU import depthCompletionNew_blockN
from Submodules.data_rectification import rectify_depth
from Submodules.custom_ip import interpolate_depth_map

class DenseLiDAR(Module):
    def __init__(self, bs):
        super().__init__()
        self.bs = bs
        self.processing = interpolate_depth_map
        self.rectification = rectify_depth
        self.DCU = depthCompletionNew_blockN(bs)

    def forward(self, image, sparse):
        # sparse = torch.tensor(sparse).to(device).squeeze()
        pseudo_depth_map = self.processing(sparse)
        rectified_depth = self.rectification(sparse, pseudo_depth_map)
        dense, attention = self.DCU(image, pseudo_depth_map, rectified_depth)

        residual = dense - sparse
        final_dense_depth = pseudo_depth_map + residual

        return final_dense_depth