import torch
import torch.nn.functional as F
from typing import List, Optional

def median_blur(image_tensor, kernel_size):
    pad_size = kernel_size // 2
    padded_image = F.pad(image_tensor, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

    
    N, C, H, W = padded_image.shape
    unfolded = F.unfold(padded_image, kernel_size=(kernel_size, kernel_size))
    unfolded = unfolded.view(N, C, kernel_size * kernel_size, -1)
    
    median_values, _ = unfolded.median(dim=2)
    
    median_image = median_values.view(N, C, H - 2 * pad_size, W - 2 * pad_size)
    return median_image


def _neight2channels_like_kernel(kernel: torch.Tensor) -> torch.Tensor:
    h, w = kernel.size()
    kernel = torch.eye(h * w, dtype=kernel.dtype, device=kernel.device)
    return kernel.view(h * w, 1, h, w)

def dilation(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    device,

    origin: Optional[List[int]] = None,
    border_type: str = "geodesic",
    border_value: float = 0.0,
    max_val: float = 1e4,

) -> torch.Tensor:

    # origin
    se_h, se_w = kernel.shape
    if origin is None:
        origin = [se_h // 2, se_w // 2]

    # padding
    pad_e: List[int] = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]
    if border_type == "geodesic":
        border_value = -max_val
        border_type = "constant"
    output: torch.Tensor = F.pad(tensor, pad_e, mode=border_type, value=border_value).to(device)

    # computation
    neighborhood = torch.zeros_like(kernel).to(device)
    neighborhood[kernel == 0] = -max_val
    B, C, H, W = tensor.size()
    h_pad, w_pad = output.shape[-2:]
    reshape_kernel = _neight2channels_like_kernel(kernel).to(device).float()

    output, _ = F.conv2d(
        output.view(B * C, 1, h_pad, w_pad), reshape_kernel, padding=0, bias=neighborhood.view(-1).flip(0)
    ).max(dim=1)
    output = output.view(B, C, H, W)

    return output.view_as(tensor)

def erosion(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    device,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = "geodesic",
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = "unfold",
) -> torch.Tensor:

    # origin
    se_h, se_w = kernel.shape
    if origin is None:
        origin = [se_h // 2, se_w // 2]

    # pad
    pad_e: List[int] = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]
    if border_type == "geodesic":
        border_value = max_val
        border_type = "constant"
    output: torch.Tensor = F.pad(tensor, pad_e, mode=border_type, value=border_value)

    # computation
    if structuring_element is None:
        neighborhood = torch.zeros_like(kernel).to(device)
        neighborhood[kernel == 0] = -max_val
    else:
        neighborhood = structuring_element.clone().to(device)
        neighborhood[kernel == 0] = -max_val

    if engine == "unfold":
        output = output.unfold(2, se_h, 1).unfold(3, se_w, 1)
        output, _ = torch.min(output - neighborhood, 4)
        output, _ = torch.min(output, 4)
    elif engine == "convolution":
        B, C, H, W = tensor.size()
        Hpad, Wpad = output.shape[-2:]
        reshape_kernel = _neight2channels_like_kernel(kernel)
        output, _ = F.conv2d(
            output.view(B * C, 1, Hpad, Wpad), reshape_kernel, padding=0, bias=-neighborhood.view(-1)
        ).min(dim=1)
        output = output.view(B, C, H, W)
    else:
        raise NotImplementedError(f"engine {engine} is unknown, use 'convolution' or 'unfold'")

    return output
