from __future__ import print_function
import argparse
import os
from dataset_mask2 import fusiondata
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fusenet import FUSENET
import torch.backends.cudnn as cudnn
import torch.optim as optim
from matplotlib import pyplot as plt
import numpy as np
import datetime
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--dataset', type=str, default='parameter_ceshi', help='facades')
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
parser.add_argument('--seed', type=int, default=12, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=150, help='weight on L1 term in objective')
parser.add_argument('--A', type=int, default=120, help='weight on L1 term in objective')
parser.add_argument('--lambda2', type=float, default=0.2, help='weight on L1 term in objective')

opt = parser.parse_args()

use_cuda = not opt.cuda and torch.cuda.is_available()

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

device = torch.device("cuda" if use_cuda else "cpu")

print('===> Loading datasets')
root_path = "./DataSet/TrainData/MSRS/"
ae_model_path = './parameter/{}/net_model_epoch_500.pth'.format(opt.lambda2)
NET = torch.load(ae_model_path)

dataset = fusiondata(root_path)
training_data_loader = DataLoader(dataset=dataset, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True, drop_last=True)

print('===> Building model')

# model2 = FUSENET()
model2 = FUSENET()
model2.train()

# 模块初始化
print('---------- Networks initialized -------------')
optimizer1 = optim.AdamW(NET.parameters(), lr=opt.lr/10, betas=(opt.beta1, 0.999), weight_decay=0.01)  #微调
optimizer2 = optim.AdamW(model2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.01)

print('-----------------------------------------------')

if not opt.cuda:
    model2 = model2.to(device)
    NET = NET.to(device)

train_loss = []

def plot_curve(data):
    fig = plt.figure()

    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('number of pictures')
    plt.ylabel('loss curve')
    plt.show()

def max_gradint(ir_image, vis_image, out_image):
    conv_op = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    sobel_kernel = np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    sobel_kernel = torch.from_numpy(sobel_kernel)
    sobel_kernel = sobel_kernel.cuda()
    conv_op.weight.data = sobel_kernel
    ir_image_edge_detect1 = conv_op(Variable(ir_image))
    vis_image_edge_detect1 = conv_op(Variable(vis_image))
    out_edge_detect1 = conv_op(Variable(out_image))

    sobel_kernel = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    sobel_kernel = torch.from_numpy(sobel_kernel)
    sobel_kernel = sobel_kernel.cuda()
    conv_op.weight.data = sobel_kernel
    ir_image_edge_detect2 = conv_op(Variable(ir_image))
    vis_image_edge_detect2 = conv_op(Variable(vis_image))
    out_edge_detect2 = conv_op(Variable(out_image))
    ir_image_edge_detect = ir_image_edge_detect1 + ir_image_edge_detect2
    vis_image_edge_detect = vis_image_edge_detect1 + vis_image_edge_detect2
    ir_edge_detect = torch.abs(ir_image_edge_detect)
    vis_edge_detect = torch.abs(vis_image_edge_detect)
    max_edge_detect = torch.maximum(ir_edge_detect, vis_edge_detect)
    out_edge_detect = out_edge_detect1 + out_edge_detect2
    out_edge_detect = torch.abs(out_edge_detect)
    return max_edge_detect, out_edge_detect

def train(epoch):
    for iteration, batch in enumerate(training_data_loader, 1):
        total_batches = 0
        total_loss = 0
        imgA_V, imgB_V, maskA, maskB = batch[0], batch[1], batch[2], batch[3]
        imgA_V = imgA_V.to(device)
        imgB_V = imgB_V.to(device)
        _, _, H, W = imgA_V.shape

        weighta = imgA_V/((imgA_V+imgB_V)+0.00000000000001)
        weightb = imgB_V/((imgA_V+imgB_V)+0.00000000000001)

        tempA06, tempA04, tempA02, sA00, tempA14, tempA12, tempA10, sA10, tempA22, tempA21, tempA20, sA20, \
            tempB06, tempB04, tempB02, sB00, tempB14, tempB12, tempB10, sB10, tempB22, tempB21, tempB20, sB20, \
            outputA3, outputA2, outputA1, outputB3, outputB2, outputB1, outputA, outputB = NET(imgA_V, imgB_V)

        F = model2(tempA06, tempA04, tempA02, sA00, tempA14, tempA12, tempA10, sA10, tempA22, tempA21, tempA20, sA20,
                   tempB06, tempB04, tempB02, sB00, tempB14, tempB12, tempB10, sB10, tempB22, tempB21, tempB20, sB20,
                   outputA3, outputA2, outputA1, outputB3, outputB2, outputB1, imgA_V, imgB_V)


        ##############################################################################s算梯度

        max_gradient_detect, out_gradient_detect = max_gradint(imgA_V, imgB_V, F)

        loss_gradient = (out_gradient_detect - max_gradient_detect).norm(1)
        loss_gradient = loss_gradient / imgA_V.shape[2] / imgA_V.shape[3]

        loss_norm1 = ((weighta * (F - imgA_V)).norm(1) + (weightb * (F - imgB_V)).norm(1))/(H*W)
        w = 5
        loss = loss_norm1 + w * loss_gradient

        #############################################################
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer1.step()
        optimizer2.step()

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        train_loss.append(loss.item())
        total_loss += loss.item()
        total_batches += 1

    running_loss = total_loss / total_batches
    log_info = "===> Epoch[{}]: Average Loss: {:.4f}, Time: {}".format(epoch, running_loss, current_time)
    print(log_info)

    with open(r"./fuse_parameter/log_{}.txt".format(str(opt.lambda2)), "a") as f:
        f.write(log_info + "\n")

def checkpoint(epoch):
    if not os.path.exists("fuse_parameter"):
        os.mkdir("fuse_parameter")
    if not os.path.exists(os.path.join("fuse_parameter", str(opt.lambda2))):
        os.mkdir(os.path.join("fuse_parameter", str(opt.lambda2)))
    net_g_model_out_path = "fuse_parameter/{}/net_model_epoch_{}.pth".format(str(opt.lambda2), epoch)
    torch.save(model2, net_g_model_out_path)
    print("Checkpoint saved to {}".format("fuse_parameter " + str(opt.lambda2)))
    net_g_auto_out_path = "fuse_parameter/{}/net_g_auto_out_path_epoch_{}.pth".format(str(opt.lambda2), epoch)
    torch.save(NET, net_g_auto_out_path)
    print("Checkpoint saved to {}".format("fuse_parameter " + str(opt.lambda2)))

if __name__ == '__main__':

    for epoch in range(1, opt.nEpochs + 1):
        train(epoch)
        if epoch % 100 == 0:
            checkpoint(epoch)
    plot_curve(train_loss)
    # 指定输出文件夹路径
    output_dir = './fuse_parameter/{}'.format(str(opt.lambda2))
    save_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(save_path)

plt.plot(train_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.savefig('./fuse_parameter/{}/figure.png'.format(str(opt.lambda2)))  # 指定保存路径和文件名
