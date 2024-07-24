import skimage.transform
import numpy as np

def sparse_loader(lidar2_path):
    img2 = skimage.io.imread(lidar2_path)
    img2 = img2 * 1.0 / 256.0
    lidar2 = np.reshape(img2, [img2.shape[0], img2.shape[1], 1]).astype(np.float32)
    return lidar2

path = '/home/mobiltech/Desktop/Test/SSDC/demo_velodyne.png'
depth = sparse_loader(path)

print(depth.shape)
# 고유 픽셀 값들을 추출 및 출력
unique_pixel_values = np.unique(depth)
print("Unique depth pixel values:")
print(unique_pixel_values)