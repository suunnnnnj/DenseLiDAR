from torch.nn import Module
from Submodules.DCU import depthCompletionNew_blockN
from Submodules.data_rectification import rectify_depth

class DenseLiDAR(Module):
    def __init__(self, bs):
        super().__init__()
        self.bs = bs
        self.rectification = rectify_depth
        self.DCU = depthCompletionNew_blockN(bs)

    def forward(self, image, sparse, pseudo_depth_map, device):
        rectified_depth = self.rectification(sparse, pseudo_depth_map)
        normal2, concat2 = self.DCU(image, pseudo_depth_map, rectified_depth)
        
        residual = normal2 - sparse

        final_dense_depth = pseudo_depth_map + residual

        return final_dense_depth