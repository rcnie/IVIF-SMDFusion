# coding= utf_8
from __future__ import print_function
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import time

# Testing settings
start = time.time()
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--lambda2', type=float, default=0.2, help='weight on L1 term in objective')
opt = parser.parse_args()

ratio = str(opt.lambda2)
model_path = r'./parameter/{}/net_model_epoch_500.pth'.format('{}')
parser.add_argument('--model', type=str, default=model_path.format(ratio), help='model file to use')
parser.add_argument('--cuda', action='store_true', help='use cuda')

opt = parser.parse_args()
netG_A = torch.load(opt.model)

device = torch.device("cuda")
netG_A = netG_A.to(device)

netG_A.eval()
root_path = r"./DataSet/TrainData/MSRS/"
num = os.listdir(root_path)

for i in range(400):
      with torch.no_grad():
        imgA_path = root_path + 'ir/' + '{:04d}.png'.format(i + 1)
        imgB_path = root_path + 'vi/' + '{:04d}.png'.format(i + 1)

        imgA = Image.open(imgA_path)
        imgB = Image.open(imgB_path)

        imgA = imgA.convert('L')
        imgB = imgB.convert('L')

        imgA_V = np.asarray(imgA)
        imgB_V = np.asarray(imgB)

        imgA_V = np.atleast_3d(imgA_V).transpose(2, 0, 1).astype(np.float_)
        imgB_V = np.atleast_3d(imgB_V).transpose(2, 0, 1).astype(np.float_)

        imgA_V = imgA_V / float(255)
        imgB_V = imgB_V / float(255)

        imgA_V = torch.from_numpy(imgA_V).float()
        imgB_V = torch.from_numpy(imgB_V).float()

        channel, Row, Col = imgA_V.shape
        channel1, Row1, Col1 = imgB_V.shape

        imgA_V = imgA_V.view(1, 1, Row, Col)
        imgB_V = imgB_V.view(1, 1, Row1, Col1)

        imgA_V = imgA_V.to(device)
        imgB_V = imgB_V.to(device)

        tempA06, tempA04, tempA02, sA00, tempA14, tempA12, tempA10, sA10, tempA22, tempA21, tempA20, sA20, \
          tempB06, tempB04, tempB02, sB00, tempB14, tempB12, tempB10, sB10, tempB22, tempB21, tempB20, sB20, \
          outputA3, outputA2, outputA1, outputB3, outputB2, outputB1, outputA, outputB = netG_A(imgA_V, imgB_V)


        out_img1 = outputA.cpu()
        out_imgA = out_img1.data[0]
        out_imgA = out_imgA.squeeze()

        out_img_fuseA = out_imgA.numpy()

        output_dir = os.path.join('./dilated', ratio)
        os.makedirs(output_dir, exist_ok=True)  # 创建ratio文件夹
        ir_dir = os.path.join(output_dir, 'ir')
        vi_dir = os.path.join(output_dir, 'vi')
        os.makedirs(ir_dir, exist_ok=True)  # 创建ir文件夹
        os.makedirs(vi_dir, exist_ok=True)  # 创建vi文件夹

        plt.imsave(os.path.join(ir_dir, '{:04d}.png'.format(i + 1)), out_img_fuseA, cmap='gray')

        out_img2 = outputB.cpu()
        out_imgB = out_img2.data[0]
        out_imgB = out_imgB.squeeze()

        out_img_fuseB = out_imgB.numpy()

        plt.imsave(os.path.join(vi_dir, '{:04d}.png'.format(i + 1)), out_img_fuseB, cmap='gray')

        print("mask has save")

end = time.time()
print(end - start)
