import os
import numpy as np
import torch.utils.data as data
import torch
from PIL import Image
import torchvision.transforms as transforms

from imgaug import augmenters as iaa
sometimes = lambda aug: iaa.Sometimes(0.8, aug)

from add_mask import add_masks_8

def make_dataset(root, train=True):
    dataset = []

    if train:
        IR = os.path.join(root)
        VIS = os.path.join(root)
    for index in range(400):

        img_ir = 'IR1/' + '{:04d}.png'.format(index + 1)
        img_vis = 'VIS/' + '{:04d}.png'.format(index + 1)

        dataset.append([os.path.join(IR, img_ir), os.path.join(VIS, img_vis)])

    return dataset

class fusiondata(data.Dataset):
    def __init__(self, root, transform=None, train=True):
        self.train = train
        self._tensor = transforms.ToTensor()
        if self.train:
            self.train_set_path = make_dataset(root, train)

    def __getitem__(self, idx):
        if self.train:

            imgA_path, imgB_path = self.train_set_path[idx]

            imgA = Image.open(imgA_path)
            imgA = imgA.convert('L')   #L 将图像变成灰度图
            imgA = np.asarray(imgA) #将图片信息转成矩阵
            imgB = Image.open(imgB_path)
            imgB = imgB.convert('L')
            imgB = np.asarray(imgB)

            #############################################
            imgA = np.atleast_3d(imgA).transpose(2, 0, 1).astype(float) #变换通道位置
            imgA = imgA / float(255) #归一化处理
            imgA = torch.from_numpy(imgA).float()

            imgB = np.atleast_3d(imgB).transpose(2, 0, 1).astype(float)
            imgB = imgB / float(255)
            imgB = torch.from_numpy(imgB).float()
            maskA, maskB,mask1, mask2 = add_masks_8(imgA, imgB)
            return imgA, imgB, maskA, maskB,mask1,mask2


    def __len__(self):
        if self.train:
            return 400
        else:
            return 400

