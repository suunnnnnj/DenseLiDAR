"""
사용법
command:
    - python test.py --loadmodel depth_completion_KITTI.tar
    - (test.py 사용법과 같음.)

input:
    - model 실행 결과 outC, outN, maskC, maskN에서 연산을 거친(test.py 내 test function에서 이미 구현)
    - pred1, predN, predMaskN

how_to_use:
    - DeepLiDAR project의 test.py에 difference_visualize 함수 붙여넣기
    - test function의 pred1 = predC * predMaskC + predN * predMaskN 아래에서 함수 호출하기
    - 함수 호출: difference_visualize(pred1, predN, predMaskN)
    - 실행

result:
    - DeepLiDAR DCU의 Normal Pathway의 결과(depth, attention map)와 최종 출력을 시각화하여 비교할 수 있음.
"""

import cv2
import matplotlib.pyplot as plt

def difference_visualize(pred1, predN, predMaskN):
    # pred1:     final dense depth
    # predN:     normal pathway depth
    # predMaskN: normal pathway attention map

    # 1: normal_pathway - result
    normalD_pred = cv2.subtract(predN.cpu().numpy().squeeze(), pred1.cpu().numpy().squeeze())
    normalA_pred = cv2.subtract(predMaskN.cpu().numpy().squeeze(), pred1.cpu().numpy().squeeze())

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Difference between Normal Pathway and Final Dense Depth")

    plt.subplot(2, 2, 1)
    plt.title(('Normal Pathway Depth - Final Dense Depth'))
    plt.imshow(normalD_pred, 'gray')

    plt.subplot(2, 2, 2)
    plt.title(('Normal Pathway Attention Map - Final Dense Depth'))
    plt.imshow(normalA_pred, 'gray')


    # 2: result - normal_pathway
    pred_normalD = cv2.subtract(pred1.cpu().numpy().squeeze(), predN.cpu().numpy().squeeze())
    pred_normalA = cv2.subtract(pred1.cpu().numpy().squeeze(), predMaskN.cpu().numpy().squeeze())

    plt.subplot(2, 2, 3)
    plt.title(('Final Dense Depth - Normal Pathway Depth'))
    plt.imshow(pred_normalD, 'gray')

    plt.subplot(2, 2, 4)
    plt.title(('Final Dense Depth - Normal Pathway Attention Map'))
    plt.imshow(pred_normalA, 'gray')

    plt.show()

# difference_visualize(pred1, predN, predMaskN)