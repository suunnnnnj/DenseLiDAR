"""
    Calculate the L2 depth loss using PyTorch tensors

    Input parameters:
    gt (torch.Tensor): Tensor of ground truth
    pseudo_depth_map (torch.Tensor): Tensor of pseudo depth map
    dense_depth_residual (torch.Tensor): Tensor of dense depth residual

    Returns:
    torch.Tensor: The L2 depth loss

"""

import torch

def L2_depth_loss(gt, dense_pseudo_depth):
    # Ensure the inputs are tensors
    if not isinstance(gt, torch.Tensor):
        gt = torch.tensor(gt)
    if not isinstance(dense_pseudo_depth, torch.Tensor):
        dense_pseudo_depth = torch.tensor(dense_pseudo_depth)

    n = gt.size(0)
    loss = torch.sum((gt - dense_pseudo_depth) ** 2) / n
    return loss