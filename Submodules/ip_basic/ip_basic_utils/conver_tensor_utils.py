#cv2.mideanBluer => torch
def median_blur_torch(image_tensor, kernel_size):
    pad_size = kernel_size // 2
    padded_image = F.pad(image_tensor, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    
    N, C, H, W = padded_image.shape
    unfolded = F.unfold(padded_image, kernel_size=(kernel_size, kernel_size))
    unfolded = unfolded.view(N, C, kernel_size * kernel_size, -1)
    
    median_values, _ = unfolded.median(dim=2)
    
    median_image = median_values.view(N, C, H - 2 * pad_size, W - 2 * pad_size)
    
    return median_image

def gaussian(kernel_size, sigma):
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid([ax, ax])
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / torch.sum(kernel)

def bilateral_filter(input, kernel_size, sigma_spatial, sigma_color):
    # Create spatial Gaussian filter
    spatial_gaussian = gaussian(kernel_size, sigma_spatial).to(input.device)

    # Pad the input
    padding = kernel_size // 2
    input_padded = F.pad(input, (padding, padding, padding, padding), mode='reflect')

    # Initialize output
    output = torch.zeros_like(input)

    for i in range(input.size(1)):
        # Extract the i-th channel
        channel = input_padded[:, i:i+1, :, :]

        # Compute the color distance
        color_distance = channel - channel.mean(dim=[2, 3], keepdim=True)
        color_gaussian = torch.exp(-(color_distance**2) / (2 * sigma_color**2))

        # Apply the spatial Gaussian filter
        filtered = F.conv2d(channel * color_gaussian, spatial_gaussian.unsqueeze(0).unsqueeze(0))

        # Normalize the result
        normalization = F.conv2d(color_gaussian, spatial_gaussian.unsqueeze(0).unsqueeze(0))
        output[:, i:i+1, :, :] = filtered / normalization

    return output