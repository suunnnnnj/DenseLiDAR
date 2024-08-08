"""
사용법
command:
    - 해당 파일 run

input:
    - depthmap_path: depthmap이 있는 파일 경로
    - save_name: 저장될 파일의 이름 지정.
    - thres: 이미지 시각화 threshold.

how_to_use:
    - depthmap_path와 save_name을 지정하고 이미지에 맞게 thres 조정.

result:
    - grayscale depth image의 컬러맵 시각화 결과.

reference:
    - https://gaussian37.github.io/vision-depth-depthmap_visualization/
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from submodules.utils.get_func import get_mask

# depth image가 아닐 경우 / 3채널 이미지일 경우 에러
depthmap_path = "/home/mobiltech/Desktop/main/DenseLiDAR/demo/dense_depth_output.png"
post_path = "/home/mobiltech/Desktop/main/DenseLiDAR/post_process/post_processing_depth.png"
pseudo_path = "/home/mobiltech/Desktop/main/DenseLiDAR/demo/demo_pseudo_depth.png"
max_range = 255

def color_depthmap(depth_path, mode):
    thres = 19500 if mode == 'pred' else 65

    depthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depthmap = depthmap.astype(np.float32) * max_range / thres

    # generate 256-level color map
    cmap = plt.get_cmap("jet", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    depth_pixel_v_s, depth_pixel_u_s = np.where(depthmap > 0)

    H, W = depthmap.shape
    color_depthmap = np.zeros((H, W, 3)).astype(np.uint8)
    for depth_pixel_v, depth_pixel_u in zip(depth_pixel_v_s, depth_pixel_u_s):
        depth = depthmap[depth_pixel_v, depth_pixel_u]
        color_index = int(255 * min(depth, max_range) / max_range)
        color = cmap[color_index, :]
        cv2.circle(color_depthmap, (depth_pixel_u, depth_pixel_v), 0, color=tuple(color), thickness=-1)

    if mode == 'pseudo':
        color_depthmap[(color_depthmap == 0).all(axis=-1)] = [255, 255, 255]
    elif mode == 'pred':
        mask = get_mask(pseudo_path, max_range, thres)
        color_depthmap[mask == 1] = [255, 255, 255]
    else:
        print('Choose mode pseudo or pred')

    plt.imshow(color_depthmap)
    img = Image.fromarray(color_depthmap)
    img.save(f'{mode}_result.png', 'png')

    print(f"Visualization saved: {mode}_result.png")

color_depthmap(depthmap_path, 'pred')
color_depthmap(post_path, 'pseudo')