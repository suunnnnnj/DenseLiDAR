import torch
import torch.nn.functional as F
import numpy as np

def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()

def total_loss(pred, predN, target):
    valid_mask = (target > 0.0).detach()
    pred = pred.permute(0, 2, 3, 1)
    predN = predN.permute(0, 2, 3, 1)

    pred_n = pred[valid_mask]
    predN_n = predN[valid_mask]
    target_n = target[valid_mask]

    loss3 = mse_loss(predN_n, target_n)
    loss1_function = torch.nn.MSELoss(reduction='mean')
    loss1 =  loss1_function(pred_n, target_n)

    loss = 0.5 * loss1 + 0.3 * loss3

    return loss, loss1, loss3

def mae(gt, img):
    dif = gt[np.where(gt > 0.0)] - img[np.where(gt > 0.0)]
    error = np.mean(np.fabs(dif))
    return error