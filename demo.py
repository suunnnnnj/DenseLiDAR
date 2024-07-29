"""
Argument : model path, demo image path, demo sparse LiDAR data path, demo pseudo depth map path, output path
Example : python demo.py --model_path checkpoint/epoch-5_loss-3.273.tar --image_path demo/demo_image.png --sparse_path demo/demo_velodyne.png --pseudo_depth_map_path demo/demo_pseudo_depth.png --output_path demo/dense_depth_output.png
"""

import torch
import torchvision.transforms as transforms
import cv2
import argparse
import numpy as np
from model import DenseLiDAR
# from submodules.utils.visualization import *

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0).float()

    return image_tensor.cuda()


def load_another_data(another_data_path):
    another_data = cv2.imread(another_data_path, cv2.IMREAD_ANYDEPTH)
    another_data = another_data.astype(np.float32) / 256.0
    another_data_tensor = torch.tensor(another_data).unsqueeze(0).unsqueeze(0).float()

    return another_data_tensor.cuda()


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    return new_state_dict


def save_depth_map(dense_depth, output_path):
    dense_depth = dense_depth.squeeze().cpu().numpy()
    dense_depth = (dense_depth * 256).astype(np.uint16)
    cv2.imwrite(output_path, dense_depth)


def main(model_path, image_path, sparse_path, pseudo_depth_map_path, output_path):
    image = load_image(image_path)
    sparse = load_another_data(sparse_path)
    pseudo_depth_map = load_another_data(pseudo_depth_map_path)

    model = DenseLiDAR(bs=1).cuda()
    
    checkpoint = torch.load(model_path)
    state_dict = remove_module_prefix(checkpoint['model_state_dict'])
    model.load_state_dict(state_dict)
    model.eval()
    
    with torch.no_grad():
        final_dense_depth = model(image, sparse, pseudo_depth_map, 'cuda')

    final_dense_depth = final_dense_depth.cpu()
    save_depth_map(final_dense_depth, output_path)
    print(f"Final dense depth map saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DenseLiDAR Inference Demo')
    parser.add_argument('--model_path', type=str, default='', help='Path to the pretrained model')
    parser.add_argument('--image_path', type=str, default='demo/demo_image.png', help='Path to the image')
    parser.add_argument('--sparse_path', type=str, default='demo/demo_velodyne.png', help='Path to the sparse LiDAR data')
    parser.add_argument('--pseudo_depth_map_path', type=str, default='demo/demo_pseudo_depth.png', help='Path to the pseudo depth map')
    parser.add_argument('--output_path', type=str, default='demo/dense_depth_output.png', help='Path to save the final dense depth map')
    args = parser.parse_args()
    
    main(args.model_path, args.image_path, args.sparse_path, args.pseudo_depth_map_path, args.output_path)
