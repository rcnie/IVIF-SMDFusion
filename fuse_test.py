from __future__ import print_function
import os
import numpy as np
import torch
from PIL import Image
import time
import matplotlib.pyplot as plt
import argparse
#定义超参数
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--lambda2', type=float, default=0.2, help='weight on L1 term in objective')
opt = parser.parse_args()

start = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_paths = {
    'MSRS': {
        'IR': './TestDataset/MSRS/IR/',
        'VIS': './TestDataset/MSRS/VIS/',
        'num_pairs': 361
    },
    'M3FD': {
        'IR': './TestDataset/M3FD/IR/',
        'VIS': './TestDataset/M3FD/VIS/',
        'num_pairs': 150
    },
    'TNO': {
        'IR': './TestDataset/TNO/IR/',
        'VIS': './TestDataset/TNO/VIS/',
        'num_pairs': 40
    }
}

# 创建目录
for dataset, paths in dataset_paths.items():
    if not os.path.exists(f"./dilated/shiyan/fuse_results/{str(opt.lambda2)}/{dataset}"):
        os.makedirs(f"./dilated/shiyan/fuse_results//{str(opt.lambda2)}/{dataset}")
        print(f"Created directory: {dataset}")

for dataset, paths in dataset_paths.items():
    num_pairs = paths['num_pairs']
    for i in range(num_pairs):
        with torch.no_grad():
            fuse_model_path = r'./fuse_parameter/{}/net_model_epoch_500.pth'.format(str(opt.lambda2))
            ae_model_path = r'./fuse_parameter/{}/net_g_auto_out_path_epoch_500.pth'.format(str(opt.lambda2))

            net = torch.load(ae_model_path)
            f_net = torch.load(fuse_model_path)

            imgA1_path = paths['IR'] + f"{i+1}.png"
            imgA2_path = paths['VIS'] + f"{i+1}.png"

            imgA1 = Image.open(imgA1_path)
            imgA2 = Image.open(imgA2_path)

            imgA1 = imgA1.convert('L')
            imgA2 = imgA2.convert('L')

            imgA1 = np.asarray(imgA1)
            imgA2 = np.asarray(imgA2)

            imgA1 = np.atleast_3d(imgA1).transpose(2, 0, 1).astype(float)
            imgA2 = np.atleast_3d(imgA2).transpose(2, 0, 1).astype(float)

            C, Row, Col = imgA1.shape

            imgA1 = imgA1 / float(255)
            imgA2 = imgA2 / float(255)

            imgA1 = torch.from_numpy(imgA1).float()
            imgA2 = torch.from_numpy(imgA2).float()

            imgA1 = imgA1.view(1, 1, Row, Col)
            imgA2 = imgA2.view(1, 1, Row, Col)

            f_net = f_net.to(device)
            img_IR, img_VI = imgA1.to(device), imgA2.to(device)

            tempA06, tempA04, tempA02, sA00, tempA14, tempA12, tempA10, sA10, tempA22, tempA21, tempA20, sA20, \
                tempB06, tempB04, tempB02, sB00, tempB14, tempB12, tempB10, sB10, tempB22, tempB21, tempB20, sB20, \
                outputA3, outputA2, outputA1, outputB3, outputB2, outputB1, outputA, outputB = net(img_IR, img_VI)

            output = f_net(tempA06, tempA04, tempA02, sA00, tempA14, tempA12, tempA10, sA10, tempA22, tempA21, tempA20, sA20,
              tempB06, tempB04, tempB02, sB00, tempB14, tempB12, tempB10, sB10, tempB22, tempB21, tempB20, sB20,
              outputA3, outputA2, outputA1, outputB3, outputB2, outputB1, img_IR, img_VI)


            out1 = output.cpu()
            out_img = out1.data[0]
            out_img = out_img.squeeze()

            out_img_fuse = out_img.numpy()

            plt.imsave(f"./dilated/shiyan/fuse_results/{str(opt.lambda2)}/{dataset}/{i+1}.png", out_img_fuse, cmap='gray')
            print(f"{dataset}: mask {i+1} has been saved")

end = time.time()
print(end - start)