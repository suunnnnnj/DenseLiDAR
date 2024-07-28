# Self-Supervised Depth Completion
Mobiltech-Gachon PJ for the month


## Target Architecture
<img width="1173" alt="image" src="https://github.com/user-attachments/assets/74fd3a33-5b4e-4949-be80-1177079d8825">

- For using raw image input and SAM result simultaneously.
- `dcu(dcu(guided_LiDAR + raw_image) + dcu(raw_LiDAR + guided_image)) `

## Requirements
- Ubuntu 20.04 LTS
- Python 3.8
- CUDA 10.2, 11.08

### Installation
```
git clone https://github.com/suunnnnnj/SSDC.git
cd SSDC
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
python train.py --datapath [YOUR_DATASET_PATH] --epochs [EPOCHS] --checkpoint [CHECKPOINT] --batch_size [BATCH_SIZE] --gpu_nums [YOUR_GPU_NUMS] --seed [RANDOM_SEED]
```
**Arguments**
- `--datapath`: your dataset path | default: datasets/
- `--epochs`: number of epochs to train | default: 40
- `--checkpoint`: number of epochs to making checkpoint | default: 10
- `--batch_size`: number of batch size to train | default: 64
- `--gpu_nums`: number of gpus to train | default: 1
- `--seed`: random seed (default: 1) | default: 1

example
```
python train.py --datapath kitti_dataset/ --epochs 50 --checkpoint 10 --batch_size 64 --gpu_nums 4 --seed 23
```

### Running
- Placeholder

<details>
  <summary><h3>Our Variation Samples</h3></summary>
  ## Basic DenseLiDAR Architecture
<img width="1303" alt="image" src="https://github.com/user-attachments/assets/4401a6aa-e52c-4239-bf95-16cd1e204443">

### 1. Add SAM from Basic DenseLiDAR
<img width="1294" alt="image" src="https://github.com/user-attachments/assets/7363b40e-925b-419e-b7f6-a303e3b229f9">


### 2. SAM + Depth Anything V2
- Add SAM for image guidance and Depth Anything V2 for self-supervised Learning
<img width="1139" alt="image" src="https://github.com/user-attachments/assets/e592d467-fe25-495d-8325-01cd20356708">

### 3. Raw LiDAR + SAM + Depth Anything V2
- Remove IP_Basic and rectify_depth
- Add SAM for image guidance and Depth Anything V2 for self-supervised Learning
<img width="622" alt="image" src="https://github.com/user-attachments/assets/6a6effb8-652c-4e60-b077-1d0a0ecb3374">

### 4. Simple version using Depth Anything V2 without SAM
<img width="1076" alt="image" src="https://github.com/user-attachments/assets/6368c662-5c2d-43ef-9cbd-e72a10833bc4">

### 5. Using DeepLiDAR
<img width="1078" alt="image" src="https://github.com/user-attachments/assets/39ae55a2-4c6d-48ff-836a-da804f8cd78d">


</details>


