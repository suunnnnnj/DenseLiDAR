import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision import transforms

#L1 Structural Loss
def gradient_x(img):
    img = F.pad(img, (0, 0, 1, 0), mode='replicate')
    gx = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gx

# HAVE SOME PROBLEM!!!! 
def gradient_y(img):
    img = F.pad(img, (1, 0, 0, 0), mode='replicate')
    gy = img[:, :, :, :-1] - img[:, :, :, 1:]
    return gy

def l_grad(D, D_pred):
    grad_x_true = gradient_x(D)
    grad_y_true = gradient_y(D)
    grad_x_pred = gradient_x(D_pred)
    grad_y_pred = gradient_y(D_pred)
    grad_loss = torch.mean(torch.abs(grad_x_true - grad_x_pred) + torch.abs(grad_y_true - grad_y_pred))
    return grad_loss

def ssim(img1, img2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, 3, 1, 0)
    mu2 = F.avg_pool2d(img2, 3, 1, 0)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 0) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 0) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 0) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def l_ssim(D, D_pred):
    ssim_loss = 1 - ssim(D, D_pred)
    return ssim_loss

def l_structural(D, D_pred):
    lambda_grad = 1.0
    lambda_ssim = 1.0
    
    grad_loss = l_grad(D, D_pred)
    ssim_loss = l_ssim(D, D_pred)
    
    structural_loss = lambda_grad * grad_loss + lambda_ssim * ssim_loss
    return structural_loss
