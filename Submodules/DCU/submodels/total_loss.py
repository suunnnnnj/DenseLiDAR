import torch
from L1_structural_loss import l_structural
from L2_depth_loss import L2_depth_loss

def total_loss(pseudo_gt_map, gt, pseudo_depth_map, dense_depth_residual):
    Dense_depth = pseudo_depth_map + dense_depth_residual
    structural_loss = l_structural(pseudo_gt_map, Dense_depth)
    depth_loss = L2_depth_loss(gt, pseudo_depth_map, dense_depth_residual)
    return structural_loss + depth_loss

loss = total_loss(D, D_pred, gt, pseudo_depth_map, dense_depth_residual)
print(f"Total loss: {loss.item()}")
