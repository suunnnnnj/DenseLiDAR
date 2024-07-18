import torch

def L2_depth_loss(lidar_data, dense_depth):
    if not isinstance(lidar_data, torch.Tensor):
        lidar_data = torch.tensor(lidar_data)
    if not isinstance(dense_depth, torch.Tensor):
        dense_depth = torch.tensor(dense_depth)
    
    if lidar_data.shape[1] > 1:
        lidar_data = torch.mean(lidar_data, dim=1, keepdim=True)

    mask = lidar_data > 0

    lidar_data_masked = lidar_data[mask]
    dense_depth_masked = dense_depth[mask]

    n = lidar_data_masked.size(0)

    loss = torch.sum((lidar_data_masked - dense_depth_masked) ** 2) / n
    return loss
