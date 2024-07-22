def normalize_hw(tensor):
    """
    Normalize the pixel values of the input tensor to be between 0 and 1 using min-max normalization,
    without altering the batch and channel dimensions.
    
    Args:
    tensor (torch.Tensor): Input tensor of shape (B, C, H, W)
    
    Returns:
    torch.Tensor: Normalized tensor with the same shape (B, C, H, W)
    """
    tensor = tensor.float()
    min_val = tensor.amin(dim=(2, 3), keepdim=True)
    max_val = tensor.amax(dim=(2, 3), keepdim=True)
    tensor = (tensor - min_val) / (max_val - min_val)
    return tensor
