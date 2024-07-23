import torch
from torch import nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import png
from Submodules.DCU import depthCompletionNew_blockN
from Submodules.data_rectification import rectify_depth

from Submodules.utils.visualization import *

class DenseLiDAR(nn.Module):
    def __init__(self, bs, model_path=None):
        super().__init__()
        self.bs = bs
        self.rectification = rectify_depth
        self.DCU = depthCompletionNew_blockN(bs)

        # Load pretrained model weights
        if model_path is not None:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            print("Checkpoint keys:", checkpoint.keys())  # Check the structure of the checkpoint

            # Check if 'model_state_dict' directly exists in checkpoint or inside another key
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # Remove unnecessary keys
            state_dict = {k.replace('DCU.', ''): v for k, v in state_dict.items() if k.startswith('DCU.')}

            # Load the state_dict into the model
            self.DCU.load_state_dict(state_dict, strict=False)  # Set strict=False to ignore unmatched keys

            print(f"Loaded pretrained model weights from {model_path}.")

    def forward(self, image, sparse, pseudo_depth_map, device):
        rectified_depth = self.rectification(sparse, pseudo_depth_map)
        normal2, concat2 = self.DCU(image, pseudo_depth_map, rectified_depth)

        residual = normal2 - sparse

        final_dense_depth = pseudo_depth_map + residual

        return final_dense_depth
    
def normalize_non_zero_pixels(pixels):
    non_zero_mask = (pixels != 0)
    non_zero_pixels = pixels[non_zero_mask]

    if non_zero_pixels.size == 0:  # If all pixels are zero
        return pixels.astype(np.float32)
    
    normalized_pixels = (non_zero_pixels - np.min(non_zero_pixels)) / (np.max(non_zero_pixels) - np.min(non_zero_pixels))
    
    result = np.zeros_like(pixels, dtype=np.float32)
    result[non_zero_mask] = normalized_pixels

    return result

def save_depth_as_png(dense_depth, output_path):
    with open(output_path, 'wb') as f:
        depth_image = (dense_depth * 256).astype(np.uint16)

        writer = png.Writer(width=depth_image.shape[1],
                            height=depth_image.shape[0],
                            bitdepth=16,
                            greyscale=True)
        writer.write(f, depth_image)

# Paths
image_path = "demo_image.png"
sparse_depth_path = "demo_velodyne.png"
pseudo_depth_path = "demo_pseudo_depth.png"
model_path = "sample_model.tar"  # pretrained model path
output_path = "dense_depth_output.png"

resize_shape = (512, 256)

try:
    # Load the image, sparse depth map, pseudo_depth_image
    image = cv2.imread(image_path)
    sparse_depth_image = cv2.imread(sparse_depth_path, cv2.IMREAD_ANYDEPTH)
    pseudo_depth_image = cv2.imread(pseudo_depth_path, cv2.IMREAD_ANYDEPTH)
    
    # Preprocess the inputs
    image = cv2.resize(image, resize_shape, interpolation=cv2.INTER_CUBIC)
    sparse_depth_image = cv2.resize(sparse_depth_image, resize_shape, interpolation=cv2.INTER_CUBIC)
    pseudo_depth_image = cv2.resize(pseudo_depth_image, resize_shape, interpolation=cv2.INTER_CUBIC)

    image = image / 256.0
    sparse_depth_image = sparse_depth_image / 256.0
    pseudo_depth_image = pseudo_depth_image / 256.0

    image = normalize_non_zero_pixels(image)
    sparse_depth_image = normalize_non_zero_pixels(sparse_depth_image)
    pseudo_depth_image = normalize_non_zero_pixels(pseudo_depth_image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor()  # Converts the image to tensor
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    sparse_depth_tensor = transform(sparse_depth_image).unsqueeze(0).to(device)
    pseudo_depth_tensor = transform(pseudo_depth_image).unsqueeze(0).to(device)

    # Initialize and run the model
    model = DenseLiDAR(bs=image_tensor.size(0), model_path=model_path)
    model.to(device)
    model.eval()
    with torch.no_grad():
        dense_depth = model(image_tensor, sparse_depth_tensor, pseudo_depth_tensor, device)

        visualize_1("final dense depth", dense_depth)

    # Postprocess and save the results as PNG
    dense_depth_np = dense_depth.squeeze().cpu().numpy()

    save_depth_as_png(dense_depth_np, output_path)
    print(f"Dense depth map saved to {output_path}")

except Exception as e:
    print(f"An error occurred: {e}")

