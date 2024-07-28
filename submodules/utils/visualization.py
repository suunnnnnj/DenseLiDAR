import PIL
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import numpy as np

def set_image(img):
    if isinstance(img, PIL.Image.Image):
        return img

    if img.ndim == 4:
        if img.shape[1] == 3:
            grayscale_transform = transforms.Grayscale(num_output_channels=1)
            img = grayscale_transform(img)
        img = img.squeeze().squeeze().cpu().numpy()

    return img

def visualize_1(title, img):
    img = set_image(img)
    plt.title(title)
    plt.imshow(img, 'gray')
    plt.show()

def visualize_2(title, title1, title2, img1, img2):
    img1 = set_image(img1)
    img2 = set_image(img2)

    plt.suptitle(title)
    plt.subplot(2,1,1)
    plt.title(title1)
    plt.imshow(img1, 'gray')

    plt.subplot(2,1,2)
    plt.title(title2)
    plt.imshow(img2, 'gray')
    plt.show()

def visualize_tensor(tensor, title="Tensor Visualization", num_channels_to_display=3):
    tensor = tensor.detach().cpu().numpy()
    
    if tensor.ndim == 4:
        tensor = tensor[0]
    
    num_channels = tensor.shape[0]
    
    if num_channels == 1:
        tensor = tensor[0]
        plt.imshow(tensor, cmap='gray')
        plt.title(f"{title} (channel 1)")
        plt.axis('off')
        plt.show()
    elif num_channels == 3:
        tensor = np.transpose(tensor, (1, 2, 0))
        plt.imshow(tensor)
        plt.title(title)
        plt.axis('off')
        plt.show()
    else:
        fig, axes = plt.subplots(1, num_channels_to_display, figsize=(15, 5))
        for i in range(min(num_channels, num_channels_to_display)):
            axes[i].imshow(tensor[i], cmap='gray')
            axes[i].set_title(f"{title} (channel {i + 1})")
            axes[i].axis('off')
        plt.show()