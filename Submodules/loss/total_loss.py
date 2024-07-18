from Submodules.loss.L1_Structural_loss import l_structural
from Submodules.loss.L2_depth_loss import L2_depth_loss

def total_loss(pseudo_gt, lidar_data, dense_depth):
    structural_loss = l_structural(pseudo_gt, dense_depth)
    depth_loss = L2_depth_loss(lidar_data, dense_depth)
    loss = structural_loss + depth_loss
    return loss, structural_loss, depth_loss

