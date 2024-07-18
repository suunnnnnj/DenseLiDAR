import torch
from torch import nn
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

from Submodules.DCU import depthCompletionNew_blockN
from Submodules.data_rectification import rectify_depth
from Submodules.custom_ip import interpolate_depth_map
from Submodules.morphology import morphology_torch

class DenseLiDAR(nn.Module):
    def __init__(self, bs, model_path=None):
        super().__init__()
        self.bs = bs
        self.processing = morphology_torch
        self.rectification = rectify_depth
        self.DCU = depthCompletionNew_blockN(bs)

        # Load pretrained model weights if provided
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

    def forward(self, image, sparse, device):
        pseudo_depth_map = self.processing(sparse, device)
        rectified_depth = self.rectification(sparse, pseudo_depth_map)
        dense, attention = self.DCU(image, pseudo_depth_map, rectified_depth)

        residual = dense - sparse
        final_dense_depth = pseudo_depth_map + residual

        return final_dense_depth


def save_depth_as_png(depth_array, output_path):
    depth_min = depth_array.min()
    depth_max = depth_array.max()

    depth_array_normalized = (depth_array - depth_min) / (depth_max - depth_min)
    depth_array_normalized = (depth_array_normalized * 255).astype(np.uint8)
    
    depth_image = Image.fromarray(depth_array_normalized)
    depth_image.save(output_path)

# Paths
image_path = "demo_image.png"
sparse_depth_path = "demo_velodyne.png"
model_path = "/home/mobiltech/Desktop/Test/SSDC/checkpoint/epoch-30_loss-7.34.tar"  # pretrained model path
output_path = "dense_depth_output2.png"

try:
    # Load the image and sparse depth map
    image = Image.open(image_path).convert("RGB")
    sparse_depth_image = Image.open(sparse_depth_path).convert("L")  # Load as grayscale

    # Preprocess the inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    sparse_depth_tensor = transforms.ToTensor()(sparse_depth_image).unsqueeze(0).to(device)

    # Initialize and run the model
    model = DenseLiDAR(bs=image_tensor.size(0), model_path=model_path)
    model.to(device)
    model.eval()
    with torch.no_grad():
        dense_depth = model(image_tensor, sparse_depth_tensor, device)

    # Postprocess and save the results as PNG
    dense_depth_np = dense_depth.squeeze().cpu().numpy()
    save_depth_as_png(dense_depth_np, output_path)
    print(f"Dense depth map saved to {output_path}")

except Exception as e:
    print(f"An error occurred: {e}")
