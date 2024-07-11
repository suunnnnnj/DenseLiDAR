# import torch
# import numpy as np
# import cv2

# # 기존 논문에 구현된 morphological processing method
# # def morphological_processing(sparse_depth):
    
# #     kernel = np.ones((5, 5), np.uint8)
# #     dilated = cv2.dilate(sparse_depth, kernel, iterations=1)
# #     return dilated

# def rectify_depth(sparse_depth, pseudo_depth, threshold=1.0):
    
#     difference = torch.abs(sparse_depth - pseudo_depth)
#     rectified_depth = torch.where(difference > threshold, torch.tensor(0.0, device=sparse_depth.device), sparse_depth)
#     return rectified_depth

# # image path
# sparse_depth_path = '/home/mobiltech/Test/raw_lidar.png' #raw_lidar
# pseudo_depth_path = '/home/mobiltech/Test/ip_basic_result.png' # ip_basic_result
# output_path = '/home/mobiltech/Test/rectification_result.png' # rectification_result
# concatenated_output_path = '/home/mobiltech/Test/concat_result.png' # concat_result

# # Transform Tensor
# sparse_depth_np = cv2.imread(sparse_depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
# pseudo_depth_np = cv2.imread(pseudo_depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

# sparse_depth = torch.tensor(sparse_depth_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
# pseudo_depth = torch.tensor(pseudo_depth_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


# # optional
# # processed_sparse_depth = morphological_processing(sparse_depth)

# rectified_depth = rectify_depth(sparse_depth, pseudo_depth, threshold=1)

# # concat
# concatenated_depth = torch.cat((pseudo_depth, rectified_depth), dim=1)  # (1, 1, H, 2W)

# # save result
# rectified_depth_np = rectified_depth.squeeze().numpy()
# concatenated_depth_np = concatenated_depth.squeeze().numpy()

# cv2.imwrite(output_path, rectified_depth_np)
# cv2.imwrite(concatenated_output_path, concatenated_depth_np)

# print(f"Rectified depth map saved to {output_path}\n")
# print(f"Concatenated depth map saved to {concatenated_output_path}")

import torch
import numpy as np
import cv2

def rectify_depth(sparse_depth, pseudo_depth, threshold=1.0):
    difference = torch.abs(sparse_depth - pseudo_depth)
    rectified_depth = torch.where(difference > threshold, torch.tensor(0.0, device=sparse_depth.device), sparse_depth)
    return rectified_depth

# 이미지 경로
sparse_depth_path = '/home/mobiltech/Test/raw_lidar.png'
pseudo_depth_path = '/home/mobiltech/Test/ip_basic_result.png'
output_path = '/home/mobiltech/Test/rectification_result.png'
concatenated_output_path = '/home/mobiltech/Test/concat_result.png'

# 이미지를 numpy 배열로 읽기
sparse_depth_np = cv2.imread(sparse_depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
pseudo_depth_np = cv2.imread(pseudo_depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

# 이미지가 올바르게 읽혔는지 확인
if sparse_depth_np is None:
    raise ValueError(f"Failed to read image at {sparse_depth_path}")
if pseudo_depth_np is None:
    raise ValueError(f"Failed to read image at {pseudo_depth_path}")

# 이미지를 텐서로 변환
sparse_depth = torch.tensor(sparse_depth_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
pseudo_depth = torch.tensor(pseudo_depth_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

# 깊이 데이터 보정
rectified_depth = rectify_depth(sparse_depth, pseudo_depth, threshold=1)

# 텐서 결합
concatenated_depth = torch.cat((pseudo_depth, rectified_depth), dim=-1)  # (1, 1, H, 2W)

# 결과를 numpy 배열로 변환
rectified_depth_np = rectified_depth.squeeze().cpu().numpy()
concatenated_depth_np = concatenated_depth.squeeze().cpu().numpy()

# 이미지를 저장할 때 적절한 형식으로 변환
rectified_depth_np_uint16 = (rectified_depth_np * 256).astype(np.uint16)
concatenated_depth_np_uint16 = (concatenated_depth_np * 256).astype(np.uint16)

# 결과 이미지 저장
cv2.imwrite(output_path, rectified_depth_np_uint16)
cv2.imwrite(concatenated_output_path, concatenated_depth_np_uint16)

print(f"Rectified depth map saved to {output_path}\n")
print(f"Concatenated depth map saved to {concatenated_output_path}")
