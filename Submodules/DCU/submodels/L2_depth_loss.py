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

def L2_depth_loss(gt, pseudo_depth_map, dense_depth_residual):
    # Ensure the inputs are tensors
    if not isinstance(gt, torch.Tensor):
        gt = torch.tensor(gt)
    if not isinstance(pseudo_depth_map, torch.Tensor):
        pseudo_depth_map = torch.tensor(pseudo_depth_map)
    if not isinstance(dense_depth_residual, torch.Tensor):
        dense_depth_residual = torch.tensor(dense_depth_residual)

    n = gt.size(0)
    loss = torch.sum((gt - pseudo_depth_map - dense_depth_residual) ** 2) / n
    return loss

# Example usage:
gt = torch.tensor([1.0, 2.0, 3.0])
pseudo_depth_map= torch.tensor([0.9, 1.8, 2.7])
dense_depth_residual = torch.tensor([0.1, 0.2, 0.3])

loss = L2_depth_loss(gt, pseudo_depth_map, dense_depth_residual)
print(f"L2 depth loss: {loss.item()}")