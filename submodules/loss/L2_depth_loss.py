import torch

def L2_depth_loss(gt_lidar, dense_depth):
    if not isinstance(gt_lidar, torch.Tensor):
        gt_lidar = torch.tensor(gt_lidar)
    if not isinstance(dense_depth, torch.Tensor):
        dense_depth = torch.tensor(dense_depth)
    
    if gt_lidar.shape[1] > 1:
        gt_lidar = torch.mean(gt_lidar, dim=1, keepdim=True)
    
    mask = gt_lidar > 0

    gt_lidar_masked = gt_lidar[mask]
    dense_depth_masked = dense_depth[mask]

    n = gt_lidar_masked.size(0)

    loss = torch.sum((gt_lidar_masked - dense_depth_masked) ** 2) / n
    return loss
