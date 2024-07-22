import cv2
from PIL import Image
import numpy as np
import sys

# 이미지 파일 경로
image_path = 'demo_velodyne.png'  # 원하는 이미지 파일 경로로 변경하세요

# OpenCV로 이미지 읽기 (그레이스케일)
opencv_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
opencv_pixels = np.array(opencv_image)  # numpy 배열로 변환

# PIL로 이미지 읽기 (그레이스케일)
pil_image = Image.open(image_path).convert('L')  # 그레이스케일로 변환
pil_pixels = np.array(pil_image)  # numpy 배열로 변환

# 정규화 함수
def normalize_non_zero_pixels(pixels):
    non_zero_mask = pixels != 0
    non_zero_pixels = pixels[non_zero_mask]
    if non_zero_pixels.size == 0:  # 0이 아닌 픽셀 값이 없으면 그대로 반환
        return pixels.astype(np.float32)
    normalized_pixels = (non_zero_pixels - np.min(non_zero_pixels)) / (np.max(non_zero_pixels) - np.min(non_zero_pixels))
    result = np.zeros_like(pixels, dtype=np.float32)
    result[non_zero_mask] = normalized_pixels
    return result

# 정규화
opencv_normalized = normalize_non_zero_pixels(opencv_pixels)
pil_normalized = normalize_non_zero_pixels(pil_pixels)

# numpy 배열 생략 없이 출력 설정
np.set_printoptions(threshold=sys.maxsize)

# 출력
print("OpenCV에서 읽은 이미지의 픽셀 값 (그레이스케일):")
print(np.unique(np.array(opencv_pixels)))

print("\nPIL에서 읽은 이미지의 픽셀 값 (그레이스케일):")
print(np.unique(np.array(pil_pixels)))

print("\nOpenCV에서 읽은 이미지의 정규화된 픽셀 값 (그레이스케일):")
print(np.unique(np.array(opencv_normalized)))

print("\nPIL에서 읽은 이미지의 정규화된 픽셀 값 (그레이스케일):")
print(np.unique(np.array(pil_normalized)))

# 두 픽셀 값 비교
if np.array_equal(opencv_normalized, pil_normalized):
    print("\n두 라이브러리에서 읽은 정규화된 픽셀 값이 같습니다.")
else:
    print("\n두 라이브러리에서 읽은 정규화된 픽셀 값이 다릅니다.")
