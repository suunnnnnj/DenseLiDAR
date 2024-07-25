import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
<<<<<<< HEAD
=======
import skimage
>>>>>>> 1fcc32fd4d96882db49c888b3d1f94cf919ba4ca
import skimage.io
import cv2  # OpenCV 라이브러리 임포트
from model import DenseLiDAR
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Argument parser
parser = argparse.ArgumentParser(description='deepCompletion')
parser.add_argument('--loadmodel', default='', help='load model')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load model
model = DenseLiDAR(1)
model.cuda()

model_path = os.path.join(ROOT_DIR, args.loadmodel)

<<<<<<< HEAD
if modelpath is not None:
    checkpoint = torch.load(modelpath, map_location=torch.device('cpu'))
    print("Checkpoint keys:", checkpoint.keys())
=======
if model_path is not None:
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    print("Checkpoint keys:", checkpoint.keys())  # Check the structure of the checkpoint
>>>>>>> 1fcc32fd4d96882db49c888b3d1f94cf919ba4ca

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    state_dict = {k.replace('DCU.', ''): v for k, v in state_dict.items() if k.startswith('DCU.')}

<<<<<<< HEAD
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained model weights from {modelpath}.")
=======
    # Load the state_dict into the model
    model.load_state_dict(state_dict, strict=False)  # Set strict=False to ignore unmatched keys

    print(f"Loaded pretrained model weights from {model_path}.")
>>>>>>> 1fcc32fd4d96882db49c888b3d1f94cf919ba4ca

# Test function
def test(imgL, sparse, pseudo_depth):
    model.eval()
    if args.cuda:
        imgL = torch.FloatTensor(imgL).cuda()
        sparse = torch.FloatTensor(sparse).cuda()
        pseudo_depth = torch.FloatTensor(pseudo_depth).cuda()

<<<<<<< HEAD
    imgL = Variable(imgL)
    sparse = Variable(sparse)
    pseudo_depth = Variable(pseudo_depth)

    device = "cuda"
    with torch.no_grad():
        dense_depth = model(imgL, sparse, pseudo_depth, device)

    dense_depth = torch.squeeze(dense_depth)
=======
def test(ori_image,sparse,pseudo_depth):
        model.eval()

        if args.cuda:
           ori_image = torch.FloatTensor(ori_image).cuda()
           sparse = torch.FloatTensor(sparse).cuda()
           pseudo_depth = torch.FloatTensor(pseudo_depth).cuda()

        ori_image= Variable(ori_image)
        sparse = Variable(sparse)
        pseudo_depth = Variable(pseudo_depth)

        device = "cuda"
        with torch.no_grad():
            dense_depth = model(ori_image, sparse, pseudo_depth, device)
>>>>>>> 1fcc32fd4d96882db49c888b3d1f94cf919ba4ca

    return dense_depth.data.cpu().numpy()

<<<<<<< HEAD
def scale_crop():
    t_list = [transforms.ToTensor()]
=======

def get_transform():
    t_list = [
        transforms.ToTensor()
    ]

>>>>>>> 1fcc32fd4d96882db49c888b3d1f94cf919ba4ca
    return transforms.Compose(t_list)


# Main function
def main():
<<<<<<< HEAD
    processed = get_transform()

    pseudo_depth_fold = 'demo_pseudo_depth.png'
    left_fold = 'demo_image.png'
    lidar2_raw = 'demo_velodyne.png'

    imgL_o = skimage.io.imread(left_fold)
    imgL_o = np.reshape(imgL_o, [imgL_o.shape[0], imgL_o.shape[1], 3])
    imgL = processed(imgL_o).numpy()
    imgL = np.reshape(imgL, [1, 3, imgL_o.shape[0], imgL_o.shape[1]])

    p_depth = skimage.io.imread(pseudo_depth_fold).astype(np.float32)
    p_depth = p_depth * 1.0 / 256.0
    p_depth = np.reshape(p_depth, [imgL_o.shape[0], imgL_o.shape[1], 1])
    p_depth = processed(p_depth).numpy()
    p_depth = np.reshape(p_depth, [1, 1, imgL_o.shape[0], imgL_o.shape[1]])

    sparse = skimage.io.imread(lidar2_raw).astype(np.float32)
    sparse = sparse * 1.0 / 256.0
    sparse = np.reshape(sparse, [imgL_o.shape[0], imgL_o.shape[1], 1])
    sparse = processed(sparse).numpy()
    sparse = np.reshape(sparse, [1, 1, imgL_o.shape[0], imgL_o.shape[1]])

    output1 = "dense_depth_output.png"

    pred = test(imgL, sparse, p_depth)
    pred = np.where(pred <= 0.0, 0.9, pred)

    # 정규화 과정 추가
    pred_min = np.min(pred)
    pred_max = np.max(pred)

    # Normalize to 0-255
    pred_normalized = (pred - pred_min) / (pred_max - pred_min) * 255.0
    pred_normalized = pred_normalized.astype('uint8')

    # Apply Gaussian Blur to smooth the image
    pred_blurred = cv2.GaussianBlur(pred_normalized, (5, 5), 0)

    # Apply colormap
    pred_colormap = cv2.applyColorMap(pred_blurred, cv2.COLORMAP_JET)

    # Save the image
    cv2.imwrite(output1, pred_colormap)

if __name__ == '__main__':
    main()
=======
   processed = get_transform()

   image_path = 'demo_image.png'
   sparse_path ='demo_velodyne.png'
   pseudo_depth_path = 'demo_pseudo_depth.png'
   
   ori_image = skimage.io.imread(image_path)
   ori_image = np.reshape(ori_image, [ori_image.shape[0], ori_image.shape[1], 3])
   image = processed(ori_image).numpy()
   image = np.reshape(image, [1, 3, ori_image.shape[0], ori_image.shape[1]])

   sparse = skimage.io.imread(sparse_path).astype(np.float32)
   sparse = sparse * 1.0 / 256.0
   sparse = np.reshape(sparse, [ori_image.shape[0], ori_image.shape[1], 1])
   sparse = processed(sparse).numpy()
   sparse = np.reshape(sparse, [1, 1, ori_image.shape[0], ori_image.shape[1]])

   pseudo_depth = skimage.io.imread(pseudo_depth_path).astype(np.float32)
   pseudo_depth = pseudo_depth * 1.0 / 256.0
   pseudo_depth = np.reshape(pseudo_depth, [ori_image.shape[0], ori_image.shape[1], 1])
   pseudo_depth = processed(pseudo_depth).numpy()
   pseudo_depth = np.reshape(pseudo_depth, [1, 1, ori_image.shape[0], ori_image.shape[1]])

   output_path = "dense_depth_output.png"

   final_dense_depth = test(image, sparse, pseudo_depth)
   final_dense_depth = np.where(final_dense_depth <= 0.0, 0.9, final_dense_depth)

   final_dense_depth = final_dense_depth * 256.0
   final_dense_depth = final_dense_depth.astype('uint16')

   res_buffer = final_dense_depth.tobytes()

   output_image = Image.new("I",final_dense_depth.T.shape)
   output_image.frombytes(res_buffer,'raw',"I;16")
   output_image.save(output_path)

if __name__ == '__main__':
   main()
>>>>>>> 1fcc32fd4d96882db49c888b3d1f94cf919ba4ca
