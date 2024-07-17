from Submodules.loss.L1_Structural_loss import l_structural
from Submodules.loss.L2_depth_loss import L2_depth_loss

def total_loss(pseudo_gt_map, gt, dense_depth):
    structural_loss = l_structural(pseudo_gt_map, dense_depth)
    depth_loss = L2_depth_loss(gt, dense_depth)
    return structural_loss + depth_loss

