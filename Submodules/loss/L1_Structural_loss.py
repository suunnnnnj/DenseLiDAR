import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchvision.transforms.functional import rgb_to_grayscale

def gradient_x(img):
    if img.dim() == 4:
        img = F.pad(img, (0, 0, 1, 0), mode='replicate')
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
    else:
        raise ValueError("Expected 4D tensor as input for gradient_x")
    return gx

def gradient_y(img):
    if img.dim() == 4:
        img = F.pad(img, (1, 0, 0, 0), mode='replicate')
        gy = img[:, :, :, :-1] - img[:, :, :, 1:]
    else:
        raise ValueError("Expected 4D tensor as input for gradient_y")
    return gy

def l_grad(pseudo_gt_map, dense_depth):
    grad_x_true = gradient_x(pseudo_gt_map)
    grad_y_true = gradient_y(pseudo_gt_map)
    grad_x_pred = gradient_x(dense_depth)
    grad_y_pred = gradient_y(dense_depth)
    grad_loss = torch.mean(torch.abs(grad_x_true - grad_x_pred) + torch.abs(grad_y_true - grad_y_pred))
    return grad_loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel=1, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=1) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        if img1.size(1) != 1:
            img1 = rgb_to_grayscale(img1)
        if img2.size(1) != 1:
            img2 = rgb_to_grayscale(img2)

        window = self.window
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        return _ssim(img1, img2, window, self.window_size, self.channel, self.size_average)

def ssim(img1, img2, window_size=11, size_average=True):
    if img1.size(1) != 1:
        img1 = rgb_to_grayscale(img1)
    if img2.size(1) != 1:
        img2 = rgb_to_grayscale(img2)

    window = create_window(window_size, 1)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, 1, size_average)

def l_ssim(pseudo_gt_map, dense_depth):
    ssim_loss = 1 - ssim(pseudo_gt_map, dense_depth)
    return ssim_loss

def l_structural(pseudo_gt_map, dense_depth):
    lambda_grad = 1.0
    lambda_ssim = 0.5

    grad_loss = l_grad(pseudo_gt_map, dense_depth)
    ssim_loss = l_ssim(pseudo_gt_map, dense_depth)

    structural_loss = lambda_grad * grad_loss + lambda_ssim * ssim_loss
    return structural_loss
