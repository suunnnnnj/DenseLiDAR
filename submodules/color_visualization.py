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

depthmap_path = "/home/mobiltech/Desktop/main/DenseLiDAR/post_process/post_processing_depth.png" # depth image가 아닐 경우 / 3채널 이미지일 경우 에러
# pseudo_path   = "/home/mobiltech/Desktop/main/DenseLiDAR/demo/demo_pseudo_depth.png"
save_name = 'result_basic.png'
thres = 70
max_range = 255

depthmap = cv2.imread(depthmap_path, cv2.IMREAD_UNCHANGED)
depthmap = depthmap.astype(np.float32) * max_range / thres

def get_color_depthmap(depthmap, max_range):
    # generate 256-level color map
    cmap = plt.get_cmap("jet", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    # sparse depthmap인 경우 depth가 있는 곳만 추출합니다.
    depth_pixel_v_s, depth_pixel_u_s = np.where(depthmap > 0)

    print(np.unique(depthmap))

    H, W = depthmap.shape
    color_depthmap = np.zeros((H, W, 3)).astype(np.uint8)
    for depth_pixel_v, depth_pixel_u in zip(depth_pixel_v_s, depth_pixel_u_s):
        depth = depthmap[depth_pixel_v, depth_pixel_u]
        color_index = int(255 * min(depth, max_range) / max_range)
        color = cmap[color_index, :]
        cv2.circle(color_depthmap, (depth_pixel_u, depth_pixel_v), 1, color=tuple(color), thickness=-1)

    color_depthmap[(color_depthmap == 0).all(axis=-1)] = [255, 255, 255]

    return color_depthmap

color_depthmap = get_color_depthmap(depthmap, max_range)

plt.imshow(color_depthmap)
img = Image.fromarray(color_depthmap)
img.show()
img.save('result_basic.png', 'png')