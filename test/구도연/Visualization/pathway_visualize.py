"""
사용법
command:
    - python test.py --loadmodel depth_completion_KITTI.tar
    - (test.py 사용법과 같음.)

input:
    - model 실행 결과 outC, outN, maskC, maskN에서 연산을 거친(test.py 내 test function에서 이미 구현)
    - predC, predN ,predMaskC, predMaskN, pred1

how_to_use:
    - DeepLiDAR project의 test.py에 pathway_visualize 함수 붙여넣기
    - test function의 pred1 = predC * predMaskC + predN * predMaskN 아래에서 함수 호출하기
    - 함수 호출: pathway_visualize(predC, predN, predMaskC, predMaskN, pred1)
    - 실행

result:
    - DeepLiDAR DCU의 Pathway별 중간 결과(depth, attention map)와 최종 출력 시각화.
"""

import cv2
import matplotlib.pyplot as plt

def pathway_visualize(predC, predN ,predMaskC, predMaskN, pred1):
    fig, axs = plt.subplots(2, 3, figsize=(30, 20))

    axs[0, 0].imshow(predC.cpu().numpy().squeeze(), cmap='gray')
    axs[0, 0].set_title('Color Pathway Depth')

    axs[0, 1].imshow(predN.cpu().numpy().squeeze(), cmap='gray')
    axs[0, 1].set_title('Normal Pathway Depth')

    axs[1, 0].imshow(predMaskC.cpu().numpy().squeeze(), cmap='gray')
    axs[1, 0].set_title('Color Pathway Attention Map')

    axs[1, 1].imshow(predMaskN.cpu().numpy().squeeze(), cmap='gray')
    axs[1, 1].set_title('Normal Pathway Attention Map')

    axs[1, 2].imshow(pred1.cpu().numpy().squeeze(), cmap='gray')
    axs[1, 2].set_title('Final Dense Depth')

    fig.delaxes(axs[0, 2])

    plt.show()

# pathway_visualize(predC, predN, predMaskC, predMaskN, pred1)