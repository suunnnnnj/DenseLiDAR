import torch

def L2_depth_loss(gt, dense_depth):
    if not isinstance(gt, torch.Tensor):
        gt = torch.tensor(gt)
    if not isinstance(dense_depth, torch.Tensor):
        dense_depth = torch.tensor(dense_depth)
    
    if gt.shape[1] > 1:
        gt = torch.mean(gt, dim=1, keepdim=True)

    mask = gt > 0

    gt_masked = gt[mask]
    dense_depth_masked = dense_depth[mask]

    n = gt_masked.size(0)

    loss = torch.sum((gt_masked - dense_depth_masked) ** 2) / n
    return loss
