from __future__ import print_function
import argparse
import os
from dataset_mask1 import fusiondata
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from gridnet import NET
import torch.backends.cudnn as cudnn
import torch.optim as optim
from matplotlib import pyplot as plt
import ssim

import datetime

# Training settings   #定义超参数
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--dataset', type=str, default='aetrain', help='facades')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--input_nc', type=int, default=1, help='input image channels')
parser.add_argument('--output_dim', type=int, default=256, help='input image channels')
parser.add_argument('--output_nc', type=int, default=1, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=150, help='weight on L1 term in objective')
parser.add_argument('--A', type=int, default=120, help='weight on L1 term in objective')
parser.add_argument('--lambda2', type=float, default=0.2, help='weight on L1 term in objective')
opt = parser.parse_args()

use_cuda = not opt.cuda and torch.cuda.is_available()
print(torch.cuda.is_available())

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)  #随机种子，参数初始化
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

device = torch.device("cuda" if use_cuda else "cpu")

print('===> Loading datasets')
root_path = os.path.abspath('./DataSet/TrainData/MSRS/')
dataset = fusiondata(root_path)
training_data_loader = DataLoader(dataset=dataset, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True, drop_last=True)

print('===> Building model')
model = NET()
model.train()

# 模块初始化
print('---------- Networks initialized -------------')

MSE = nn.MSELoss()
ssim_loss = ssim.ssim
optimizer = optim.AdamW(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.01)  #优化器
print('-----------------------------------------------')

if not opt.cuda:
    model = model.to(device)

train_loss = []

def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('number of pictures')
    plt.ylabel('loss curve')
    plt.show()

def train(epoch):
    total_ssim = 0
    global prev_loss
    total_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        imgA_V, imgB_V, maskA, maskB,mask1, mask2 = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]#加mask使用的读取数据

        imgA_V = imgA_V.to(device)
        imgB_V = imgB_V.to(device)
        maskA = maskA.to(device)
        maskB = maskB.to(device)

        _,_,H,W = imgA_V.shape


        tempA06, tempA04, tempA02, sA00, tempA14, tempA12, tempA10, sA10, tempA22, tempA21, tempA20, sA20, \
            tempB06, tempB04, tempB02, sB00, tempB14, tempB12, tempB10, sB10, tempB22, tempB21, tempB20, sB20, \
            outputA3, outputA2, outputA1, outputB3, outputB2, outputB1, outputA, outputB = model(maskA, maskB)


        weighta = imgA_V/((imgA_V+imgB_V)+0.00000000000001)
        weightb = imgB_V/((imgA_V+imgB_V)+0.00000000000001)

        optimizer.zero_grad()

        MSEA = MSE(weighta * outputA, weighta * imgA_V)
        MSEB = MSE(weightb * outputB, weightb * imgB_V)

        ssima = 1 - ssim_loss(outputA, imgA_V)
        ssimb = 1 - ssim_loss(outputB, imgB_V)

        lossMSE = MSEA + MSEB
        ssim = ssima + ssimb

        loss = lossMSE + 5 * ssim

        loss.backward()

        optimizer.step()

        train_loss.append(loss.item())
        total_loss += loss.item()
        total_ssim += ssim.item()

    running_loss = total_loss / len(training_data_loader)
    running_ssim = total_ssim / len(training_data_loader)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_info = "===> Epoch[{}]: Loss: {:.4f}, SSIM: {:.4f}, Time: {}".format(epoch, running_loss, 1 - running_ssim,
                                                                             current_time)
    print(log_info)

    # # 将log信息保存到文本文件
    with open("./parameter/log_{}.txt".format(str(opt.lambda2)), "a") as f:
        f.write(log_info + "\n")

def checkpoint(epoch):  #保存参数
    if not os.path.exists("parameter"):
        os.mkdir("parameter")
    if not os.path.exists(os.path.join("parameter", str(opt.lambda2))):
        os.mkdir(os.path.join("parameter", str(opt.lambda2)))
    net_g_model_out_path = "parameter/{}/net_model_epoch_{}.pth".format(str(opt.lambda2), epoch)
    torch.save(model, net_g_model_out_path)
    print("Checkpoint saved to {}".format("parameter " + str(opt.lambda2)))


if __name__ == '__main__':


    for epoch in range(1, opt.nEpochs + 1):
        train(epoch)
        if epoch % 50 == 0:  #第50轮保存一次参数
            checkpoint(epoch)
plot_curve(train_loss)

plt.plot(train_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.savefig('./parameter/{}/figure.png'.format(str(opt.lambda2)))  # 指定保存路径和文件名