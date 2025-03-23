import torch
import argparse
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Training settings   #定义超参数
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--lambda2', type=float, default=3, help='weight on L1 term in objective')
opt = parser.parse_args()

def add_masks_1(image1, image2, alpha=0.228, rect_size=(5, 5)): ##加固定比例的01mask块
    shape = image1.shape
    _, h, w = shape
    n_rects = int(alpha * h * w / (rect_size[0] * rect_size[1]))

    noise_masks_all = []
    image_masks = []
    dtype = image1.dtype
    for _ in range(2):
        noise_masks = []
        mask = torch.ones((shape[0], h, w), dtype=torch.uint8, device=image1.device)
        while len(noise_masks) < n_rects:
            x = torch.randint(high=w - rect_size[1], size=())
            y = torch.randint(high=h - rect_size[0], size=())
            rect = [x, y, x + rect_size[1], y + rect_size[0]]
            noise_masks_all.append(rect)
            noise_masks.append(rect)
        for rect in noise_masks:
            mask[..., rect[1]:rect[3], rect[0]:rect[2]] = 0
        noise_masks.append(mask)
        image_masks.append(mask.to(dtype))
    # Add masks to images
    noise_mask_1, noise_mask_2 = image_masks[0], image_masks[1]
    masked_image1 = image1 * noise_mask_1
    masked_image2 = image2 * noise_mask_2
    mask1 = np.logical_not(noise_mask_1)
    mask2 = np.logical_not(noise_mask_2)
    return masked_image1, masked_image2, mask1, mask2

def add_masks_2(image1, image2, alpha=0.228, std_dev=25, rect_size=(5, 5)):     ##加固定比例的噪声mask块
    shape = image1.shape
    _, h, w = shape
    n_rects = int(alpha * h * w / (rect_size[0] * rect_size[1]))

    noise_masks_all = []
    image_masks = []
    dtype = image1.dtype
    for _ in range(2):
        noise_masks = []
        mask = torch.ones((shape[0], h, w), dtype=torch.float32, device=image1.device)
        while len(noise_masks) < n_rects:
            x = torch.randint(high=w - rect_size[1], size=())
            y = torch.randint(high=h - rect_size[0], size=())
            rect = [x, y, x + rect_size[1], y + rect_size[0]]
            noise_masks_all.append(rect)
            noise_masks.append(rect)
        for rect in noise_masks:
            noise_value = torch.normal(mean=0, std=torch.tensor(std_dev, dtype=torch.float32))
            mask[..., rect[1]:rect[3], rect[0]:rect[2]] = noise_value
        noise_masks.append(mask)
        image_masks.append(mask.to(dtype))
    # Add masks to images
    noise_mask_1, noise_mask_2 = image_masks[0], image_masks[1]
    masked_image1 = image1 * noise_mask_1
    masked_image2 = image2 * noise_mask_2
    mask1 = np.logical_not(noise_mask_1)
    mask2 = np.logical_not(noise_mask_2)
    return masked_image1, masked_image2, mask1, mask2

def add_masks_1_1(image1, image2):
    alpha = 0.20
    rect_size = (5, 5)
    shape = image1.shape
    _, h, w = shape
    n_rects = int(alpha * h * w / (rect_size[0] * rect_size[1]))

    noise_masks_all = []
    image_masks = []
    dtype = image1.dtype
    for _ in range(2):
        noise_masks = []
        mask = torch.ones((shape[0], h, w), dtype=torch.uint8, device=image1.device)
        while len(noise_masks) < n_rects:
            x = torch.randint(high=w - rect_size[1], size=())
            y = torch.randint(high=h - rect_size[0], size=())
            rect = [x, y, x + rect_size[1], y + rect_size[0]]
            def intersect(rect1, rect2):
                return (rect1[0] < rect2[2] and rect1[2] > rect2[0] and
                        rect1[1] < rect2[3] and rect1[3] > rect2[1])
            if not any(intersect(rect, r) for r in noise_masks_all):
                noise_masks_all.append(rect)
                noise_masks.append(rect)
        for rect in noise_masks:
            mask[..., rect[1]:rect[3], rect[0]:rect[2]] = 0
        noise_masks.append(mask)
        image_masks.append(mask.to(dtype))
    # Add masks to images
    noise_mask_1, noise_mask_2 = image_masks[0], image_masks[1]
    # masked_image1 = torch.bitwise_and(image1, noise_mask_1)
    # masked_image2 = torch.bitwise_and(image2, noise_mask_2)
    masked_image1 = image1 * noise_mask_1
    masked_image2 = image2 * noise_mask_2
    return masked_image1, masked_image2, noise_mask_1, noise_mask_2

def add_masks_3(img1, img2):
    _, height, width = img1.shape
    noise_ratio = 0.20

    b = 0.5
    a = 0.5

    total_pixels = height * width
    num_noisy_pixels = round(total_pixels * noise_ratio)
    # 生成符合高斯分布的噪声
    noise = torch.randn((height, width))
    # 将噪声的值限制在 [0, 1] 范围内
    noise.clamp_(0, 1)

    # 生成噪声 mask
    mask1 = torch.zeros((height, width))
    mask2 = torch.zeros((height, width))

    indices1 = torch.randperm(total_pixels)[:num_noisy_pixels]
    mask1.view(-1)[indices1] = 1

    # 从剩余像素中选择非mask1的位置作为mask2
    available_indices = torch.nonzero(mask1.view(-1) == 0).view(-1)
    indices2 = available_indices[torch.randperm(available_indices.shape[0])[:num_noisy_pixels]]
    mask2.view(-1)[indices2] = 1

    masked_imgA = img1.clone()
    masked_imgB = img2.clone()

    masked_imgA[:, mask1.bool()] = 0
    masked_imgB[:, mask2.bool()] = 0

    return masked_imgA, masked_imgB, mask1, mask2

def add_masks_8(img1, img2):   #加噪声像素点mask
    _, height, width = img1.shape
    noise_ratio = 0.20

    b = 0.5
    a = 0.5

    total_pixels = height * width
    num_noisy_pixels = round(total_pixels * noise_ratio)
    # 生成符合高斯分布的噪声
    noise = torch.randn((height, width))
    # 将噪声的值限制在 [0, 1] 范围内
    noise.clamp_(0, 1)

    # 生成噪声 mask
    mask1 = torch.zeros((height, width))
    mask2 = torch.zeros((height, width))

    indices1 = torch.randperm(total_pixels)[:num_noisy_pixels]
    mask1.view(-1)[indices1] = 1

    # 从剩余像素中选择非mask1的位置作为mask2
    available_indices = torch.nonzero(mask1.view(-1) == 0).view(-1)
    indices2 = available_indices[torch.randperm(available_indices.shape[0])[:num_noisy_pixels]]
    mask2.view(-1)[indices2] = 1

    masked_imgA = img1.clone()
    masked_imgB = img2.clone()

    # 将噪声添加到图像中
    masked_imgA[:, mask1.bool()] = a * noise[mask1.bool()] + b * img1[:, mask1.bool()]
    masked_imgB[:, mask2.bool()] = a * noise[mask2.bool()] + b * img2[:, mask2.bool()]


    return masked_imgA, masked_imgB, mask1, mask2

def add_masks_8_1(img1, img2):
    _, height, width = img1.shape
    mask_ratio = opt.lambda2

    total_pixels = height * width
    num_noisy_pixels = round(total_pixels * mask_ratio)

    # 生成噪声 mask
    mask1 = torch.zeros((height, width))
    mask2 = torch.zeros((height, width))

    indices1 = torch.randperm(total_pixels)[:num_noisy_pixels]
    mask1.view(-1)[indices1] = 1

    # 从剩余像素中选择非mask1的位置作为mask2
    available_indices = torch.nonzero(mask1.view(-1) == 0).view(-1)
    indices2 = available_indices[torch.randperm(available_indices.shape[0])[:num_noisy_pixels]]
    mask2.view(-1)[indices2] = 1

    masked_imgA = img1.clone()
    masked_imgB = img2.clone()

    # 将mask添加到图像中
    masked_imgA[:, mask1.bool()] = img2[:, mask1.bool()]
    masked_imgB[:, mask2.bool()] = img1[:, mask2.bool()]

    return masked_imgA, masked_imgB, mask1, mask2
def add_masks_9(img1, img2):    #加位置不限制的噪声像素点mask
    _, height, width = img1.shape
    noise_ratio = 0.75
    a = 0.5
    b = 0.5
    total_pixels = height * width
    num_noisy_pixels = round(total_pixels * noise_ratio)

    # 生成符合高斯分布的噪声
    noise = torch.randn((height, width))

    # 将噪声的值限制在 [0, 1] 范围内
    noise.clamp_(0, 1)

    # 生成噪声 mask
    mask1 = torch.zeros((height, width))
    mask2 = torch.zeros((height, width))

    indices1 = torch.randperm(total_pixels)[:num_noisy_pixels]
    mask1.view(-1)[indices1] = 1

    indices2 = torch.randperm(total_pixels)[:num_noisy_pixels]
    mask2.view(-1)[indices2] = 1

    masked_imgA = img1.clone()
    masked_imgB = img2.clone()

    # 将噪声添加到图像中
    masked_imgA[:, mask1.bool()] = a * noise[mask1.bool()] + b * img1[:, mask1.bool()]
    masked_imgB[:, mask2.bool()] = a * noise[mask2.bool()] + b * img2[:, mask2.bool()]

    return masked_imgA, masked_imgB, mask1, mask2

def add_masks_10(img1, img2):   #生成固定比例的椒盐噪声
    _, height, width = img1.shape
    noise_ratio = 0.20
    a = 0.5
    b = 0.5
    total_pixels = height * width
    num_noisy_pixels = round(total_pixels * noise_ratio)

    # 生成符合高斯分布的噪声
    noise = torch.zeros((height, width))

    # 生成椒盐噪声 mask
    mask1 = torch.zeros((height, width), dtype=torch.float32)
    mask2 = torch.zeros((height, width), dtype=torch.float32)
    mask1[torch.randint(0, height, (num_noisy_pixels,)), torch.randint(0, width, (num_noisy_pixels,))] = 1
    mask2[torch.randint(0, height, (num_noisy_pixels,)), torch.randint(0, width, (num_noisy_pixels,))] = 1

    masked_imgA = img1.clone()
    masked_imgB = img2.clone()

    # 将噪声添加到图像中
    masked_imgA[:, mask1.bool()] = a * noise[mask1.bool()] + b * img1[:, mask1.bool()]
    masked_imgB[:, mask2.bool()] = a * noise[mask2.bool()] + b * img2[:, mask2.bool()]

    return masked_imgA, masked_imgB, mask1, mask2

def add_masks_11(img1, img2):  #添加伯努利噪声
    _, height, width = img1.shape
    noise_ratio = 0.20
    a = 0.5
    b = 0.5
    total_pixels = height * width
    num_noisy_pixels = round(total_pixels * noise_ratio)

    # 生成伯努利噪声 mask
    mask1 = torch.zeros((height, width))
    mask2 = torch.zeros((height, width))

    indices1 = torch.randperm(total_pixels)[:num_noisy_pixels]
    mask1.view(-1)[indices1] = 1

    indices2 = torch.randperm(total_pixels)[:num_noisy_pixels]
    mask2.view(-1)[indices2] = 1

    masked_imgA = img1.clone()
    masked_imgB = img2.clone()

    # 将噪声添加到图像中
    masked_imgA[:, mask1.bool()] = a * img1[:, mask1.bool()]
    masked_imgB[:, mask2.bool()] = b * img2[:, mask2.bool()]

    return masked_imgA, masked_imgB, mask1, mask2

def add_masks_12(img1, img2):  #生成泊松噪声
    _, height, width = img1.shape
    noise_ratio = 0.20
    a = 0.5
    b = 0.5
    total_pixels = height * width
    num_noisy_pixels = round(total_pixels * noise_ratio)

    # 生成泊松噪声
    noise = torch.poisson(torch.ones((height, width)))

    # 将噪声的值限制在 [0, 1] 范围内
    noise.clamp_(0, 1)

    # 生成噪声 mask
    mask1 = torch.zeros((height, width))
    mask2 = torch.zeros((height, width))

    indices1 = torch.randperm(total_pixels)[:num_noisy_pixels]
    mask1.view(-1)[indices1] = 1

    indices2 = torch.randperm(total_pixels)[:num_noisy_pixels]
    mask2.view(-1)[indices2] = 1

    masked_imgA = img1.clone()
    masked_imgB = img2.clone()

    # 将噪声添加到图像中
    masked_imgA[:, mask1.bool()] = a * noise[mask1.bool()] + b * img1[:, mask1.bool()]
    masked_imgB[:, mask2.bool()] = a * noise[mask2.bool()] + b * img2[:, mask2.bool()]

    return masked_imgA, masked_imgB, mask1, mask2

def add_masks_13(img1, img2):  #生成均匀噪声
    _, height, width = img1.shape
    noise_ratio = 0.20
    a = 0.5
    b = 0.5
    total_pixels = height * width
    num_noisy_pixels = round(total_pixels * noise_ratio)

    # 生成均匀噪声
    low, high = 0.0, 1.0
    noise = torch.FloatTensor(height, width).uniform_(low, high)

    # 生成噪声 mask
    mask1 = torch.zeros((height, width))
    mask2 = torch.zeros((height, width))

    indices1 = torch.randperm(total_pixels)[:num_noisy_pixels]
    mask1.view(-1)[indices1] = 1

    indices2 = torch.randperm(total_pixels)[:num_noisy_pixels]
    mask2.view(-1)[indices2] = 1

    masked_imgA = img1.clone()
    masked_imgB = img2.clone()

    # 将噪声添加到图像中
    masked_imgA[:, mask1.bool()] = a * noise[mask1.bool()] + b * img1[:, mask1.bool()]
    masked_imgB[:, mask2.bool()] = a * noise[mask2.bool()] + b * img2[:, mask2.bool()]

    return masked_imgA, masked_imgB, mask1, mask2


# if __name__ == "__main__":
    img1 = Image.open(r"D:\DataSet\TrainData\MSRS\msrs\ir\ir1.png").convert("L")
    img2 = Image.open(r"D:\DataSet\TrainData\MSRS\msrs\vi\vi1.png").convert("L")
    transform1 = transforms.ToTensor()
    a = transform1(img1)
    b = transform1(img2)
    # a = torch.ones((1,10,10))
    # b = torch.ones((1,10,10))
    print(a.shape)
    # a = a.numpy()
    # a = a[0]
    c, d,_,_ = add_masks_9(a, b)
    # print(type(b))
    #
    # print(b.shape)
    # b = b.numpy()*255
    # print(b)
    # image = Image.fromarray(b[0],mode="L")
    # b = b.astype("uint8")
    # c = c[0]
    # d = d[0]
    # 784,222
    print(b)
    transform = transforms.ToPILImage()
    # #
    image1 = transform(c)
    image2 = transform(d)
    # # b = b *255
    # image = Image.fromarray(b)
    #
    image1.save(r"D:\mask+GridNet\code20230805\加噪图/ir1-50%01mask块.png")
    image2.save(r"D:\mask+GridNet\code20230805\加噪图/vi1-50%01mask块.png")


if __name__ == "__main__":  ##验证
    img1 = Image.open(r"D:\DataSet\TrainData\MSRS\msrs\ir\ir1.png").convert("L")
    img2 = Image.open(r"D:\DataSet\TrainData\MSRS\msrs\vi\vi1.png").convert("L")
    transform1 = transforms.ToTensor()
    # a = transform1(img1)
    # b = transform1(img2)
    a = torch.ones((1,256,256))
    # b= torch.full((1, 10, 10), 3)
    b = torch.ones((1,256,256))
    # b= torch.full((1, 10, 10), 5)
    print(a.shape)
    # a = a.numpy()
    # a = a[0]
    c, d,e,f = add_masks_13(a, b)
    sum_of_pixels1 = torch.sum(e== 1).item()
    sum_of_pixels2 = torch.sum(e== 1).item()
    # print(type(b))
    # mask = mask1 + mask2

    # print(mask1)
    # print(mask2)
    print(c)
    print(d)
    print(e)
    print(f)
    # print(b.shape)
    # b = b.numpy()*255
    # print(b)
    # image = Image.fromarray(b[0],mode="L")
    # b = b.astype("uint8")
    # c = c[0]
    # d = d[0]
    # 784,222
    print(b)
    transform = transforms.ToPILImage()
    # #
    image1 = transform(c)
    image2 = transform(d)
    # # b = b *255
    # image = Image.fromarray(b)
    #
    image1.save(r"D:\mask+GridNet\code20240327\加噪图/ir1-50%01mask块.png")
    image2.save(r"D:\mask+GridNet\code20240327\加噪图/vi1-50%01mask块.png")