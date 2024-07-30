# A Real-time Pseudo Dense Depth Guided Depth Completion Network(Non-official)
This repo is non-offical implement of paper : [DenseLiDAR: A Real-time Pseudo Dense Depth Guided Depth Completion Network](https://arxiv.org/pdf/2108.12655)
> Jiaqi Gu. et al, DenseLiDAR: A Real-Time Pseudo Dense Depth Guided Depth Completion Network. ICRA2021

We would like to thank Jiaqi Gu et al. for their groundbreaking work on DenseLiDAR. This implementation was inspired by their paper presented at ICRA 2021. Additionally, we appreciate the contributions of the open-source community and the resources provided by [PyTorch](https://pytorch.org/).
### This project was conducted as part of the Internship program at Mobiltech and Gachon University VIPlab.

- Implementation by. 구도연, 김다혜, 조재현 [[VIPlab Gachon University.](https://sites.google.com/view/vip-lab)]
- Project Supervision : 정선재 [[Mobiltech](https://www.mobiltech.io/)]

## Disclaimer

Please note that this implementation is not the official code provided by the authors of the paper. As such, the performance metrics obtained using our code may differ from those reported in the original paper. Differences in implementation details, parameter settings, and hardware can all contribute to variations in performance.


## DenseLiDAR Architecture
<img width="1173" alt="image" src="https://github.com/user-attachments/assets/4401a6aa-e52c-4239-bf95-16cd1e204443">

## Requirements
- Ubuntu 20.04 LTS
- Python 3.8
- CUDA 10.2, 11.08

### Installation
```
git clone https://github.com/suunnnnnj/DenseLiDAR.git
cd DenseLiDAR
pip install -r requirements.txt
```

### Dataset
**kitti_raw dataset download**
```
cd datasets/kitti_depth
wget https://github.com/youmi-zym/CompletionFormer/files/12575038/kitti_archives_to_download.txt
wget -i kitti_archives_to_download.txt -P kitti_raw/
cd kitti_raw
unzip "*.zip"
```

```
kitti_depth
├──data_depth_annotated
|     ├── train
|     └── val
├── data_depth_velodyne
|     ├── train
|     └── val
├── data_depth_selection
|     ├── test_depth_completion_anonymous
|     |── test_depth_prediction_anonymous
|     └── val_selection_cropped
└── kitti_raw
      ├── 2011_09_26
      ├── 2011_09_28
      ├── 2011_09_29
      ├── 2011_09_30
      └── 2011_10_03
```  


## Usage

### Training
```
python train.py --data_path [YOUR_DATASET_PATH] --epochs [EPOCHS] --checkpoint [CHECKPOINT] --batch_size [BATCH_SIZE] --gpu_nums [YOUR_GPU_NUMS]
```
**Arguments**
- `--datapath`: your dataset path | default: None
- `--epochs`: number of epochs to train | default: 40
- `--checkpoint`: number of epochs to making checkpoint | default: 5
- `--batch_size`: number of batch size to train | default: 1
- `--gpu_nums`: number of gpus to train | default: 1

**Example**
```
python train.py --data_path datasets/ --epochs 40 --batch_size 16 --gpu_nums 4
```

### Demo
```
python demo.py --model_path [YOUR_MODEL_PATH] --image_path [YOUR_IMAGE_PATH] --sparse_path [YOUR_LIDAR_POINT_PATH] --pseudo_depth_map_path [YOUR_PSEUDO_DEPTH_MAP_PATH] --output_path [YOUR_SAVE_PATH]
```
**Arguments**
- `--model_path`: your model path | default: None
- `--image_path`: your raw image path | default: demo/demo_image.png
- `--sparse_path`: your raw lidar path | default: demo/demo_velodyne.png
- `--pseudo_depth_map_path`: your pseudo depth map path | default: demo/demo_pseudo_depth.png
- `--output_path`: your save result path | default: demo/dense_depth_output.png

**Example**
```
python demo.py --model_path checkpoint/epoch-5_loss-3.273.tar --image_path demo/demo_image.png --sparse_path demo/demo_velodyne.png --pseudo_depth_map_path demo/demo_pseudo_depth.png --output_path demo/dense_depth_output.png
```
