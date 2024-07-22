import torch
import torch.nn.functional as F

def normalize_hw(tensor):
    """
    Normalize the H and W dimensions of the input tensor.
    
    Args:
    tensor (torch.Tensor): Input tensor of shape (B, C, H, W)
    
    Returns:
    torch.Tensor: Normalized tensor with the same shape (B, C, H, W)
    """
    tensor = tensor.float()
    B, C, H, W = tensor.size()
    tensor = tensor.view(B, C, -1)      # Reshape to (B, C, H*W)
    tensor = F.normalize(tensor, p=2, dim=2)  # Normalize along the H*W dimension
    tensor = tensor.view(B, C, H, W)    # Reshape back to (B, C, H, W)
    return tensor