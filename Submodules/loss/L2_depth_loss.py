import torch

def L2_depth_loss(gt, dense_depth):
    if not isinstance(gt, torch.Tensor):
        gt = torch.tensor(gt)
    if not isinstance(dense_depth, torch.Tensor):
        dense_depth = torch.tensor(dense_depth)

    n = gt.size(0)
    loss = torch.sum((gt - dense_depth) ** 2) / n
    return loss