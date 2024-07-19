import torch
from torchvision.transforms import InterpolationMode, transforms

def rectify_depth(sparse_depth, pseudo_depth, threshold=1.0):
    difference = torch.abs(sparse_depth - pseudo_depth)
    rectified_depth = torch.where(difference > threshold, torch.tensor(0.0, device=sparse_depth.device), sparse_depth)
    
    BICUBIC = InterpolationMode.BICUBIC
    resize_transform = transforms.Resize((256, 512), antialias=True, interpolation=BICUBIC)
    rectified_depth = resize_transform(rectified_depth)
        
    return rectified_depth
