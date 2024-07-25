from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from test_dataloader import test_dataLoader as lsn
from test_dataloader import test_trainLoader as DA
from model import DenseLiDAR
from Submodules.loss.total_loss import total_loss
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='deepCpmpletion')
parser.add_argument('--datapath', default='', help='datapath')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=1, help='number of batch size to train')
parser.add_argument('--gpu_nums', type=int, default=1, help='number of gpu to train')
parser.add_argument('--loadmodel', default= '', help='load model')
parser.add_argument('--savemodel', default='my', help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()

datapath = args.datapath
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img,all_sparse,all_depth,all_pseudo,all_dense = lsn.dataloader(datapath)

TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(all_left_img,all_sparse,all_depth,all_pseudo,all_dense, True),
        batch_size=args.batch_size , shuffle=True, num_workers=8, drop_last=True)

model = DenseLiDAR(args.batch_size)
# model = DDP(model, device_ids=[rank])

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

para_optim = []

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7, last_epoch=-1)

def train(inputl,gt1,sparse,pseudo,dense,params):
        device = 'cuda'
        model.train()
        inputl = Variable(torch.FloatTensor(inputl))
        gt1 = Variable(torch.FloatTensor(gt1))
        sparse = Variable(torch.FloatTensor(sparse))
        pseudo = Variable(torch.FloatTensor(pseudo))
        dense = Variable(torch.FloatTensor(dense))
        params = Variable(torch.FloatTensor(params))
        if args.cuda:
            inputl,gt1,sparse,pseudo, dense, params = inputl.cuda(),gt1.cuda(),sparse.cuda(),pseudo.cuda(),dense.cuda(),params.cuda()
        optimizer.zero_grad()

        dense_depth = model(inputl, sparse, pseudo, device)
        print('sparse2')
        print(sparse.shape)
        t_loss, s_loss, d_loss = total_loss(dense, gt1, dense_depth)
        t_loss.backward()
        optimizer.step()

        return t_loss, s_loss, d_loss

def main():
    start_full_time = time.time()

    for epoch in range(1, args.epochs+1):
        total_train_loss = 0

         ## training ##
        for batch_idx, (imgL_crop,input_crop1,sparse2,pseudo_crop,dense_crop,params) in tqdm(enumerate(TrainImgLoader),total=len(TrainImgLoader), desc=f"Epoch {epoch}"): #rawimage, gtlidar,rawlidar,pseudo_depth,gt_depth,param
            start_time = time.time()

            loss,loss1,loss2 = train(imgL_crop,input_crop1,sparse2,pseudo_crop,dense_crop,params)
            print('Iter %d / %d training loss = %.4f, Ploss = %.4f, n_loss = %.4f, time = %.2f' % (batch_idx, epoch, loss, loss1, loss2, time.time() - start_time))
            total_train_loss += loss

        print('epoch %d total training loss = %.10f' %(epoch, total_train_loss/len(TrainImgLoader)))

        #SAVE
        if epoch % 1 == 0:
            savefilename = args.savemodel + '.tar'
            torch.save({
	            'epoch': epoch,
	            'state_dict': model.state_dict(),
	            'train_loss': total_train_loss/len(TrainImgLoader),
	        }, savefilename)

    print('full finetune time = %.2f HR' %((time.time() - start_full_time)/3600))

if __name__ == '__main__':
   main()