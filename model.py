from torch.nn import Module
from Submodules.DCU import depthCompletionNew_blockN
from Submodules.data_rectification import rectify_depth
from Submodules.custom_ip import interpolate_depth_map
from Submodules.morphology import morphology_torch
from torchvision.transforms import InterpolationMode, transforms

class DenseLiDAR(Module):
    def __init__(self, bs):
        super().__init__()
        self.bs = bs
        
        self.processing = morphology_torch
        self.rectification = rectify_depth
        self.DCU = depthCompletionNew_blockN(bs)

    def forward(self, image, sparse, device):
        # input: image: 256 / sparse: 1216
        
        pseudo_depth_map = self.processing(sparse, device)
        # pseudo_depth: 1216
        rectified_depth = self.rectification(sparse, pseudo_depth_map)
        
        BICUBIC = InterpolationMode.BICUBIC
        resize_transform = transforms.Resize((256, 512), antialias=True, interpolation=BICUBIC)
        pseudo_depth_map = resize_transform(pseudo_depth_map)
        sparse = resize_transform(sparse)
        
        normal2, attention = self.DCU(image, pseudo_depth_map, rectified_depth)
        residual = normal2 - sparse

        final_dense_depth = pseudo_depth_map + residual
        
        return final_dense_depth

