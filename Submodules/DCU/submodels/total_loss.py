import torch
from Submodules.DCU.submodels.L2_depth_loss import L2_depth_loss
from Submodules.DCU.submodels.L1_Structural_loss import l_structural

def total_loss(pseudo_gt_map, gt, dense_pseudo_depth):
    structural_loss = l_structural(pseudo_gt_map, dense_pseudo_depth)
    depth_loss = L2_depth_loss(gt, pseudo_depth_map, dense_pseudo_depth)
    return structural_loss + depth_loss

