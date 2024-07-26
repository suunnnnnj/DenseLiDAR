#argument : model path ,demo image path, demo velodyne path, demo pseudo depth map path
#Ex : python3 demo.py --model_path checkpoint/epoch-5_loss-4.412.tar --left_image_path demo_image.png --sparse_depth_path demo_velodyne.png --pseudo_depth_map_path demo_pseudo_depth.png 

import torch
from torch.nn import Module
import cv2
import matplotlib.pyplot as plt
import argparse
import torchvision.transforms as transforms
import numpy as np
from model import DenseLiDAR  # 필요한 모듈을 임포트하세요

def visualize_tensor(tensor, title="Tensor Visualization", num_channels_to_display=3):
    tensor = tensor.detach().cpu().numpy()
    
    if tensor.ndim == 4:  # Batch, Channels, Height, Width
        tensor = tensor[0]  # 배치의 첫 번째 샘플 선택
    
    num_channels = tensor.shape[0]
    
    if num_channels == 1:  # 단일 채널 이미지
        tensor = tensor[0]  # 채널 차원 제거
        plt.imshow(tensor, cmap='gray')
        plt.title(f"{title} (channel 1)")
        plt.axis('off')
        plt.show()
    elif num_channels == 3:  # RGB 이미지
        tensor = np.transpose(tensor, (1, 2, 0))  # 차원을 Height, Width, Channels로 재배치
        plt.imshow(tensor)
        plt.title(title)
        plt.axis('off')
        plt.show()
    else:
        # 첫 몇 개 채널을 별도 이미지로 표시
        fig, axes = plt.subplots(1, num_channels_to_display, figsize=(15, 5))
        for i in range(min(num_channels, num_channels_to_display)):
            axes[i].imshow(tensor[i], cmap='gray')
            axes[i].set_title(f"{title} (channel {i + 1})")
            axes[i].axis('off')
        plt.show()

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0).float()
    return image_tensor.cuda()

def load_depth_map(depth_map_path):
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH)
    depth = depth_map.astype(np.float32) / 256.0
    depth_tensor = torch.tensor(depth).unsqueeze(0).unsqueeze(0).float()
    return depth_tensor.cuda()

def remove_module_prefix(state_dict):
    # 'module.' 접두사를 제거하여 state_dict 키를 수정
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def main(model_path, left_image_path, sparse_depth_path, pseudo_depth_map_path):
    # 이미지와 깊이 맵 로드
    left_image = load_image(left_image_path)
    sparse_depth = load_depth_map(sparse_depth_path)
    pseudo_depth_map = load_depth_map(pseudo_depth_map_path)

    # 모델 초기화
    model = DenseLiDAR(bs=1).cuda()  # bs 매개변수 설정
    
    # 사전 학습된 모델 로드
    checkpoint = torch.load(model_path)
    state_dict = remove_module_prefix(checkpoint['model_state_dict'])
    model.load_state_dict(state_dict)
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        final_dense_depth = model(left_image, sparse_depth, pseudo_depth_map, 'cuda')

    # 결과 시각화
    visualize_tensor(left_image, title="Left Image")
    visualize_tensor(sparse_depth, title="Sparse Depth")
    visualize_tensor(pseudo_depth_map, title="Pseudo Depth Map")
    visualize_tensor(final_dense_depth, title="Final Dense Depth Map")
    final_dense_depth = final_dense_depth.cpu()
    print(np.unique(final_dense_depth))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DenseLiDAR Inference Demo')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--left_image_path', type=str, required=True, help='Path to the left image')
    parser.add_argument('--sparse_depth_path', type=str, required=True, help='Path to the sparse depth map')
    parser.add_argument('--pseudo_depth_map_path', type=str, required=True, help='Path to the pseudo depth map')
    args = parser.parse_args()
    
    main(args.model_path, args.left_image_path, args.sparse_depth_path, args.pseudo_depth_map_path)
