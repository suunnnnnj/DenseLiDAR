import torch
from L1_structural_loss import l_structural
from L2_depth_loss import L2_depth_loss

def total_loss(pseudo_gt_map, gt, dense_pseudo_depth):
    structural_loss = l_structural(pseudo_gt_map, dense_pseudo_depth)
    depth_loss = L2_depth_loss(gt, pseudo_depth_map, dense_pseudo_depth)
    return structural_loss + depth_loss

