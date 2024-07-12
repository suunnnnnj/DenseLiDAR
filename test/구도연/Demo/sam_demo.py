"""
사용법
command:
    - 해당 파일 run

input:
    - sample_image: segment할 이미지
    - model checkpoint: sam github에서 .pth 파일 다운.

how_to_use:
    - git clone https://github.com/facebookresearch/segment-anything.git
    - segment-anything readme에서 model checkpoint 다운로드.
    - demo directory 아래에 해당 파일 붙여넣기
    - segment-anything parent directory 아래에 mdoel.pth, sample_image 붙여넣기
    - 실행

result:
    -
    -
reference:
    - https://blog.roboflow.com/how-to-use-segment-anything-model-sam/
"""

import cv2
from segment_anything import SamAutomaticMaskGenerator
import torch
from segment_anything import sam_model_registry
import supervision as sv
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_l"

sam = sam_model_registry[MODEL_TYPE](checkpoint="../sam_vit_l_0b3195.pth")
sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam)

image_bgr = cv2.imread("../sample_image.png")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
result = mask_generator.generate(image_rgb)

mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)
detections = sv.Detections.from_sam(result)
annotated_image = mask_annotator.annotate(image_bgr, detections)
print("Image Annotated!")


plt.imshow(annotated_image)
plt.show()