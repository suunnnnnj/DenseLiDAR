import PIL
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

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